# Standard library
import os
import csv
import math
import time
import random
import multiprocessing as mp

from typing import List
from collections import defaultdict
from typing import Optional
# Third-party
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

# PyG / torch-scatter（注意版本匹配）
from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader  # 新版PyG推荐用这个

# RDKit（尽量集中导入，避免重复和路径差异）
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem,
    MACCSkeys,
    Descriptors,
    Crippen,
    rdMolDescriptors,
    rdMolTransforms,
    Descriptors3D,
    Fingerprints,
    RDKFingerprint,
)
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT

# sklearn（只用到shuffle的话这样即可）
from sklearn.utils import shuffle

# from utils import auto_n_jobs_free80
# n_jobs = auto_n_jobs_free80()
# print("n_jobs =", n_jobs)

RDLogger.DisableLog('rdApp.*')

# globleJobs = 16

# ==== Parallel preprocessing helpers ====
# NOTE: This file may be imported by different scripts. We try to use your utils.auto_n_jobs_free80()
# if it exists, otherwise fall back to 80% of total CPU cores.
def auto_n_jobs_free80(
    cpu_ratio: float = 0.8, max_jobs: Optional[int] = None,
    min_jobs: int = 1,) -> int:
    """
    使用“空闲核(近似)的 cpu_ratio（默认80%）”，剩下预留给系统/其他进程。
    - 优先尊重 CPU affinity（比如容器/任务调度限制）
    - 尊重 SLURM 环境变量（如果存在）
    - 用 os.getloadavg()[0] 近似估算“正在占用的核数”
      （Windows 没有 getloadavg -> 当作负载=0，即尽量多用）
    """
    # 1) 允许使用的核数（affinity 优先）
    try:
        n_allowed = len(os.sched_getaffinity(0))  # Linux/部分Unix
    except Exception:
        n_allowed = os.cpu_count() or 1

    # 2) HPC/SLURM 限制（如果存在）
    for env in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(env)
        if v:
            try:
                n_allowed = min(n_allowed, int(v))
            except Exception:
                pass
    n_allowed = max(1, int(n_allowed))

    # 3) 估算当前占用（1min loadavg）
    try:
        load1 = float(os.getloadavg()[0])
    except Exception:
        load1 = 0.0  # Windows 等：没有就当空闲

    used = int(load1 + 0.5)                 # 四舍五入
    used = max(0, min(n_allowed, used))
    free = max(1, n_allowed - used)

    # 4) 取空闲核的 cpu_ratio
    jobs = int(math.floor(free * float(cpu_ratio)))
    jobs = max(int(min_jobs), jobs)
    jobs = min(jobs, n_allowed)

    if max_jobs is not None:
        jobs = min(jobs, int(max_jobs))

    return max(1, int(jobs))



def _limit_threads_for_worker() -> None:
    # Avoid each worker spawning many BLAS/OpenMP threads -> CPU oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch as _torch
        _torch.set_num_threads(1)
    except Exception:
        pass

_MP_CFG = None

def _mp_init(cfg: dict) -> None:
    global _MP_CFG
    _MP_CFG = cfg
    _limit_threads_for_worker()
    try:
        from rdkit import RDLogger as _RDLogger
        _RDLogger.DisableLog('rdApp.*')
    except Exception:
        pass

def _mp_worker(smiles_label):
    smiles, label = smiles_label
    cfg = _MP_CFG
    return _build_data_from_smiles(smiles, label, **cfg)


def _build_data_from_smiles(
    smiles: str,label,*,task: str,conversion: float,fingerprint_list,
    fp_radius: int,ecfp_bits: int,maccs_bits: int,atompair_bits: int,extfp_bits: int,
    torsion_bits: int,avalon_bits: int,extfp_maxPath: int,
    # 3D params (keep same defaults as the current file)
    # FAST_3D: bool = True,
    # NUM_CONFS_FAST: int = 5,
    # PRUNE_RMS_FAST: float = 0.75,
    # MAX_ATTEMPTS_FAST: int = 20,
    # MAX_ITERS_FAST: int = 50,
    # DO_MINIMIZE: bool = True,
):
    """Featurize one molecule into a PyG Data. Safe to run inside multiprocessing workers."""
    import traceback
    z = smiles
    mol = Chem.MolFromSmiles(z)
    if mol is None:   #应该不用管，默认是到不到这里的
        print("[MOL-FAIL] Invalid SMILES, MolFromSmiles 返回 None:", z)
        # Extremely rare: invalid SMILES. Return a tiny dummy graph to keep indices aligned.
        x = torch.zeros((1, 2), dtype=torch.long)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.long)
        if task == 'classification':
            y = torch.tensor(int(label), dtype=torch.long).view(1, -1)
        else:
            y = torch.tensor(float(label) * float(conversion), dtype=torch.float32).view(1, -1)

        fp_dim = 0
        if 'ecfp' in fingerprint_list: fp_dim += int(ecfp_bits)
        if 'maccs' in fingerprint_list: fp_dim += int(maccs_bits)
        if 'ap' in fingerprint_list: fp_dim += int(atompair_bits)
        if 'ext' in fingerprint_list: fp_dim += int(extfp_bits)
        if 'torsion' in fingerprint_list: fp_dim += int(torsion_bits)
        if 'avalon' in fingerprint_list: fp_dim += int(avalon_bits)
        fp_tensor = torch.zeros((fp_dim,), dtype=torch.float32)

        desc = torch.zeros((1, 8), dtype=torch.float32)
        geom = torch.zeros((1, 13), dtype=torch.float32)
        geom_mask = torch.zeros((1, 1), dtype=torch.float32)
        data = Data(x=x, y=y, z=z, edge_index=edge_index, edge_attr=edge_attr)
        data.fp = fp_tensor
        data.desc = desc
        data.geom = geom
        data.geom_mask = geom_mask
        return data

    # ---------- Graph ----------
    N = mol.GetNumAtoms()
    M = mol.GetNumBonds()
    type_idx = []
    chirality_idx = []
    atomic_number = []
    for atom in mol.GetAtoms():
        # ATOM_LIST = list(range(1,119)) -> index is atomic_num - 1
        type_idx.append(int(atom.GetAtomicNum()) - 1)
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())  # 这个没用上

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x_node = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bt = bond.GetBondType()
        bd = bond.GetBondDir()
        edge_feat.append([BOND_LIST.index(bt), BONDDIR_LIST.index(bd)])
        edge_feat.append([BOND_LIST.index(bt), BONDDIR_LIST.index(bd)])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

    if task == 'classification':
        y = torch.tensor(int(label), dtype=torch.long).view(1, -1)
    elif task == 'regression':
        y = torch.tensor(float(label) * float(conversion), dtype=torch.float32).view(1, -1)
    else:
        raise ValueError('task must be either regression or classification')

    # ---------- Fingerprints  计算所需指纹  ----------
    arrs = []
    # 单独 try/except 每一类指纹，打印失败原因，但尽量不影响其它指纹
    if 'ecfp' in fingerprint_list:
        try:
            fp_ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=fp_radius, nBits=ecfp_bits)
            arrs.append(np.array(fp_ecfp, dtype=np.float32))
        except Exception as e:
            print(f"[FP-FAIL][ECFP] SMILES={z} error={repr(e)}")
            traceback.print_exc()
    if 'maccs' in fingerprint_list:
        try:
            fp_maccs = MACCSkeys.GenMACCSKeys(mol)
            arrs.append(np.array(fp_maccs, dtype=np.float32))
        except Exception as e:
            print(f"[FP-FAIL][MACCS] SMILES={z} error={repr(e)}")
            traceback.print_exc()
    if 'ap' in fingerprint_list:
        try:
            fp_ap = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=atompair_bits)
            arrs.append(np.array(fp_ap, dtype=np.float32))
        except Exception as e:
            print(f"[FP-FAIL][AP] SMILES={z} error={repr(e)}")
            traceback.print_exc()
    if 'ext' in fingerprint_list:
        try:
            fp_ext = RDKFingerprint(mol, maxPath=extfp_maxPath, fpSize=extfp_bits)
            arrs.append(np.array(fp_ext, dtype=np.float32))
        except Exception as e:
            print(f"[FP-FAIL][EXT] SMILES={z} error={repr(e)}")
            traceback.print_exc()
    if 'torsion' in fingerprint_list:
        try:
            fp_torsion = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=torsion_bits)
            arrs.append(np.array(fp_torsion, dtype=np.float32))
        except Exception as e:
            print(f"[FP-FAIL][TORSION] SMILES={z} error={repr(e)}")
            traceback.print_exc()
    if 'avalon' in fingerprint_list:
        try:
            # 使用正确的 Avalon FP 计算方法
            fp_avalon = GetAvalonFP(mol, nBits=avalon_bits)
            arrs.append(np.array(fp_avalon, dtype=np.float32))
        except Exception as e:
            print(f"[FP-FAIL][AVALON] SMILES={z} error={repr(e)}")
            traceback.print_exc()

    # 拼接所有选中指纹
    if len(arrs) == 0:
        # 所有指纹都失败的情况，直接给一个长度 0 的向量，后续模型可以只用 graph/desc/geom
        fp_tensor = torch.zeros((0,), dtype=torch.float32)
        print(f"[FP-FAIL][ALL] SMILES={z} -> 所有指纹计算失败，fp 向量长度为 0")
    else:
        fp_tensor = torch.from_numpy(np.concatenate(arrs, axis=0))

    # ---------- Descriptors (8) ------------------------------------------
    try:
        desc_vals = [
            float(Descriptors.MolWt(mol)),
            float(Crippen.MolLogP(mol)),
            float(rdMolDescriptors.CalcTPSA(mol)),
            float(rdMolDescriptors.CalcNumHBD(mol)),
            float(rdMolDescriptors.CalcNumHBA(mol)),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(rdMolDescriptors.CalcNumHeavyAtoms(mol)),
        ]
        desc = torch.tensor(desc_vals, dtype=torch.float32).view(1, -1)  # [1,8]
    except Exception as e:
        # 极少数奇怪分子可能在某个 descriptor 上挂掉
        print(f"[DESC-FAIL] SMILES={z} error={repr(e)}")
        traceback.print_exc()
        desc = torch.zeros((1, 8), dtype=torch.float32)

    # ---------- 3D geom (13) with fallback3D几何特征 ---------------------------
    geom_np = np.zeros((13,), dtype=np.float32)
    geom_valid = 0.0

    FAST_3D = True
    DO_MINIMIZE = True  # 想更快就 False：直接用 ETKDG 坐标，不做力场最小化 把 DO_MINIMIZE=True（只最小化 5 个构象找最低能）
    num_confs = 5 if FAST_3D else 20
    # maxAttempts = 20 if FAST_3D else 50
    # 一些 RDKit 版本的 ETKDG 参数对象不支持 maxAttempts 属性，这里只设置通用、安全的参数
    maxIters = 50 if FAST_3D else 200
    prune_rms = 0.75 if FAST_3D else 0.5

    geom_stage = "init"
    try:
        geom_stage = "AddHs"
        mol3d = Chem.AddHs(Chem.Mol(mol))
        if mol3d is None:
            raise ValueError("MolFromSmiles failed for 3D")

        if FAST_3D:
            geom_stage = "EmbedMultipleConfs-ETKDG"
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            # params.maxAttempts = maxAttempts
            params.enforceChirality = True
            params.pruneRmsThresh = prune_rms
            # 针对当前 RDKit 版本：必须用“位置参数”传入 numConfs 和 params
            cids = AllChem.EmbedMultipleConfs(mol3d, num_confs, params)
            if len(cids) == 0:
                geom_stage = "EmbedMultipleConfs-relaxed-prune"
                params.pruneRmsThresh = -1.0
                cids = AllChem.EmbedMultipleConfs(mol3d, num_confs, params)

            if len(cids) == 0:
                # 再放宽：random coords（更稳，但可能更慢一点）
                geom_stage = "EmbedMultipleConfs-randomCoords"
                params2 = AllChem.ETKDGv3()
                params2.randomSeed = 42
                # params2.maxAttempts = maxAttempts
                params2.useRandomCoords = True
                params2.enforceChirality = True
                params2.pruneRmsThresh = -1.0
                cids = AllChem.EmbedMultipleConfs(mol3d, max(1, num_confs // 2), params2)

            if len(cids) == 0:
                # 最后兜底：单构象
                geom_stage = "EmbedMolecule-single-fallback"
                params3 = AllChem.ETKDGv3()
                params3.randomSeed = 42
                # params3.maxAttempts = maxAttempts
                params3.useRandomCoords = True
                params3.enforceChirality = True
                st = AllChem.EmbedMolecule(mol3d, params3)
                if st != 0:
                    print(f"[WARN] 单构象都失败了 EmbedMultipleConfs failed: {smiles}")
                    # raise ValueError("EmbedMolecule failed (single conf fallback)")
                cids = [0]  # 单构象时 confId=0

            geom_stage = "select-best-conf"
            best_energy = float("inf")
            best_cid = int(cids[0])
            for cid in cids:
                cid = int(cid)
                try:
                    # if DO_MINIMIZE:    改到下面了 改成下面的对best_cid进行的最小化
                    #     AllChem.UFFOptimizeMolecule(mol3d, confId=cid, maxIters=maxIters)
                    ff = AllChem.UFFGetMoleculeForceField(mol3d, confId=cid)
                    e = float(ff.CalcEnergy())
                    if e < best_energy:
                        best_energy = e
                        best_cid = cid
                except Exception as e:
                    print(f"[WARN] 能量筛选 conf UFF 失败: SMILES={smiles} conf_id={cid} error={repr(e)}")
            if DO_MINIMIZE:
                geom_stage = "UFFOptimize-best-conf"
                try:
                    AllChem.UFFOptimizeMolecule(mol3d, confId=best_cid, maxIters=maxIters)
                except Exception as e:
                    print(f"[WARN] UFFOptimizeMolecule 失败: SMILES={smiles} conf_id={best_cid} error={repr(e)}")

            try:
                conf = mol3d.GetConformer(best_cid)
            except ValueError as e:
                print(f"Skipping {best_cid} molecule {smiles} due to error: {e}")
                raise
            # if best_cid < 0 or best_cid >= mol3d.GetNumConformers():
            #     print(f"Best Conformer ID: {best_cid}")
            #     raise ValueError(f"Invalid Conformer ID: {best_cid}")


        else:
            geom_stage = "EmbedMolecule-ETKDG"
            status = AllChem.EmbedMolecule(mol3d, AllChem.ETKDG())
            if status != 0:
                raise ValueError("EmbedMolecule failed")
            if DO_MINIMIZE:
                geom_stage = "UFFOptimize-single-conf"
                AllChem.UFFOptimizeMolecule(mol3d)
            conf = mol3d.GetConformer()

        # bond length stats
        geom_stage = "bond-length-stats"
        bond_lengths = []
        for bond in mol3d.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_lengths.append(rdMolTransforms.GetBondLength(conf, i, j))
        mean_bl = np.mean(bond_lengths) if bond_lengths else 0.0
        std_bl = np.std(bond_lengths) if bond_lengths else 0.0
        max_bl = np.max(bond_lengths) if bond_lengths else 0.0

        # bond angle stats
        geom_stage = "bond-angle-stats"
        bond_angles = []
        for j in range(mol3d.GetNumAtoms()):
            nbrs = [n.GetIdx() for n in mol3d.GetAtomWithIdx(j).GetNeighbors()]
            if len(nbrs) < 2:
                continue
            for a in range(len(nbrs)):
                for b in range(a + 1, len(nbrs)):
                    i, k = nbrs[a], nbrs[b]
                    bond_angles.append(rdMolTransforms.GetAngleDeg(conf, i, j, k))
        mean_angle = np.mean(bond_angles) if bond_angles else 0.0
        std_angle = np.std(bond_angles) if bond_angles else 0.0
        max_angle = np.max(bond_angles) if bond_angles else 0.0

        # dihedral circular stats
        geom_stage = "dihedral-stats"
        dihedrals = []
        for bond in mol3d.GetBonds():
            j = bond.GetBeginAtomIdx()
            k = bond.GetEndAtomIdx()
            nbrs_j = [n.GetIdx() for n in mol3d.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
            nbrs_k = [n.GetIdx() for n in mol3d.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
            for i in nbrs_j:
                for l in nbrs_k:
                    dihedrals.append(rdMolTransforms.GetDihedralDeg(conf, i, j, k, l))

        if dihedrals:
            try:
                ph = np.deg2rad(np.array(dihedrals, dtype=np.float32))
                mean_sin = float(np.mean(np.sin(ph)))
                mean_cos = float(np.mean(np.cos(ph)))
                R = float(np.hypot(mean_sin, mean_cos)) + 1e-8

                # 增加更多数值稳定性检查
                eps = 1e-12
                if not np.isfinite(R) or not np.isfinite(mean_sin) or not np.isfinite(mean_cos):
                    mean_sin = mean_cos = 0.0
                    circ_std = 0.0
                else:
                    R = np.clip(R, eps, 1.0)
                    # 额外检查避免log(0)或负数
                    log_arg = np.clip(-2.0 * np.log(R), 0, None)  # 确保参数非负
                    circ_std = float(np.sqrt(log_arg))
            except Exception as e:
                print(f"[WARN] 二面角统计计算失败: {e}")
                mean_sin = mean_cos = circ_std = 0.0

            # ph = np.deg2rad(np.array(dihedrals, dtype=np.float32))
            # mean_sin = float(np.mean(np.sin(ph)))
            # mean_cos = float(np.mean(np.cos(ph)))
            # R = float(np.hypot(mean_sin, mean_cos)) + 1e-8
            # eps = 1e-12
            # if not np.isfinite(R):
            #     circ_std = 0.0
            # else:
            #     R = np.clip(R, eps, 1.0)
            #     circ_std = float(np.sqrt(-2.0 * np.log(R)))
        else:
            mean_sin = mean_cos = circ_std = 0.0

        geom_stage = "PMI-Rg"
        pmi1 = float(Descriptors3D.PMI1(mol3d))
        pmi2 = float(Descriptors3D.PMI2(mol3d))
        pmi3 = float(Descriptors3D.PMI3(mol3d))

        pos = conf.GetPositions()
        centroid = pos.mean(axis=0)
        d = np.linalg.norm(pos - centroid, axis=1)
        Rg = float(np.sqrt(np.mean(d ** 2)))

        geom_np = np.array([
            mean_bl, std_bl, max_bl,
            mean_angle, std_angle, max_angle,
            mean_sin, mean_cos, circ_std,
            pmi1, pmi2, pmi3,
            Rg
        ], dtype=np.float32)

        geom_valid = 1.0
    except Exception as e:
        # 这里统一打印 3D 失败的阶段和错误，geom 保持为 0，但 geom_mask=0 标记无效
        print(f"[3D-FAIL] SMILES={smiles} stage={geom_stage} error={repr(e)}")
        traceback.print_exc()
        geom_valid = 0.0

    geom = torch.from_numpy(geom_np).float().view(1, -1)  # [1,13]
    geom_mask = torch.tensor([[float(geom_valid)]], dtype=torch.float32)

    data = Data(x=x_node, y=y, z=z, edge_index=edge_index, edge_attr=edge_attr)
    data.fp = fp_tensor
    data.desc = desc
    data.geom = geom
    data.geom_mask = geom_mask

    return data

# 定义可用的指纹名称列表
# 顶部新增
AVAILABLE_FPS = ['ecfp','maccs','ap','ext','torsion','avalon']

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def seed_worker(worker_id):
    import random, numpy as np, torch
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain assert from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def random_scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    index = np.array(list(range(0, len(dataset))))
    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(dataset.smiles_data):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

    n_total_valid = int(np.floor(valid_size * len(dataset)))
    n_total_test = int(np.floor(test_size * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_index, val_index, test_index = index[train_idx], index[valid_idx], index[test_idx]

    # if sort:
    #     train_index = sorted(train_index)
    #     val_index = sorted(val_index)
    #     test_index = sorted(test_index)

    return train_index, val_index, test_index


def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                smiles = row['smiles']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    smiles_data, labels = shuffle(smiles_data, labels, random_state=42)
    print(f"Total samples: {len(smiles_data)} (shuffled with seed=42)")
    return smiles_data, labels


class MolTestDataset(InMemoryDataset):

    # def __init__(self, data_path, target, task, transform=None, pre_transform=None, force_reprocess=False):

    def __init__(self, data_path, target, task, transform=None, pre_transform=None, force_reprocess=False,
                 fingerprint_list=None,process_n_jobs=None,
                 fp_radius=2,
                 ecfp_bits=2048,
                 maccs_bits=167,
                 ap_bits=2048,
                 ext_bits=2048,
                 torsion_bits=2048,
                 avalon_bits=1024,
                 extfp_maxPath=5):
        # super(Dataset, self).__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = task
        self.data_path = data_path
        self.dataset = os.path.splitext(os.path.basename(data_path))[0]
        self._root_dir = os.path.dirname(data_path)
        super().__init__(root=self._root_dir, transform=transform, pre_transform=pre_transform)

        # 用户选择的指纹列表
        self.fingerprint_list = fingerprint_list or AVAILABLE_FPS
        # # 验证用户输入
        # for fp_name in self.fingerprint_list:
        #     if fp_name not in AVAILABLE_FPS:
        #         raise ValueError(f"Unsupported fingerprint: {fp_name}")

        # ---------------- 指纹相关的超参数 ----------------
        self.fp_radius = fp_radius        # ECFP 半径
        self.ecfp_bits = ecfp_bits        # ECFP 位长
        self.maccs_bits = maccs_bits      # MACCS 位长（通常 167）
        self.atompair_bits = ap_bits  # AtomPair 位长
        self.extfp_bits = ext_bits      # ExtFP 位长
        self.extfp_maxPath = extfp_maxPath  # ExtFP 最大路径长度

        self.torsion_bits = torsion_bits
        self.avalon_bits = avalon_bits

        # parallel preprocessing (pt generation)============加的多jobs 加快速度
        self.p_jobs = int(process_n_jobs) if process_n_jobs is not None else auto_n_jobs_free80()
        self.process_n_jobs = max(1, self.p_jobs)
        # ------------------------------------------------
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

        self.conversion = 1
        if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')


    def process(self):
        # Parallelize heavy RDKit preprocessing to speed up .pt generation
        data_list = []
        n_jobs = getattr(self, "process_n_jobs", 1)
        n_jobs = max(1, int(n_jobs))
        # n_jobs = globleJobs

        cfg = dict(
            task=self.task,
            conversion=getattr(self, "conversion", 1.0),
            fingerprint_list=self.fingerprint_list,
            fp_radius=self.fp_radius,
            ecfp_bits=self.ecfp_bits,
            maccs_bits=self.maccs_bits,
            atompair_bits=self.atompair_bits,
            extfp_bits=self.extfp_bits,
            torsion_bits=self.torsion_bits,
            avalon_bits=self.avalon_bits,
            extfp_maxPath=self.extfp_maxPath,
            # FAST_3D=globals().get("FAST_3D", True),
            # NUM_CONFS_FAST=globals().get("NUM_CONFS_FAST", 5),
            # PRUNE_RMS_FAST=globals().get("PRUNE_RMS_FAST", 0.75),
            # MAX_ATTEMPTS_FAST=globals().get("MAX_ATTEMPTS_FAST", 20),
            # MAX_ITERS_FAST=globals().get("MAX_ITERS_FAST", 50),
            # DO_MINIMIZE=globals().get("DO_MINIMIZE", True),
        )
        tasks = list(zip(self.smiles_data, self.labels))
        t0 = time.time()

        if n_jobs <= 1:
            for smi, lab in tasks:
                data_list.append(_build_data_from_smiles(smi, lab, **cfg))
        else:
            import torch.multiprocessing as tmp
            try:
                tmp.set_sharing_strategy("file_system")
            except Exception as e:
                print("[WARN] set_sharing_strategy failed:", e)

            ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
            chunksize = max(1, len(tasks) // (n_jobs * 20))
            print(f"[preprocess] Using {n_jobs} processes (chunksize={chunksize}) to build pt file...")
            with ctx.Pool(processes=n_jobs, initializer=_mp_init, initargs=(cfg,)) as pool:
                for data in pool.imap(_mp_worker, tasks, chunksize=chunksize):
                    data_list.append(data)

        os.makedirs(self.processed_dir, exist_ok=True)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        dt = time.time() - t0
        try:
            valid_rate = float(torch.cat([d.geom_mask for d in data_list], dim=0).mean().item())
            print(
                f'Processed {len(data_list)} samples for dataset "{self.dataset}". geom_valid_rate={valid_rate:.3f}. time={dt / 60:.1f} min. Saved to: {self.processed_paths[0]}')
        except Exception:
            print(
                f'Processed {len(data_list)} samples for dataset "{self.dataset}". time={dt / 60:.1f} min. Saved to: {self.processed_paths[0]}')

        # for index, smiles in enumerate(self.smiles_data):
        #     # mol = Chem.MolFromSmiles(smiles)
        #
        #     z = self.smiles_data[index]
        #     mol = Chem.MolFromSmiles(z)
        #     # mol = Chem.AddHs(mol)
        #
        #     N = mol.GetNumAtoms()
        #     M = mol.GetNumBonds()
        #
        #     type_idx = []
        #     chirality_idx = []
        #     atomic_number = []
        #     for atom in mol.GetAtoms():
        #         type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        #         chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        #         atomic_number.append(atom.GetAtomicNum())  # 这个没用上
        #
        #     x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        #     x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        #     x_node = torch.cat([x1, x2], dim=-1)
        #
        #     row, col, edge_feat = [], [], []
        #     for bond in mol.GetBonds():
        #         start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        #         row += [start, end]
        #         col += [end, start]
        #         edge_feat.append([
        #             BOND_LIST.index(bond.GetBondType()),
        #             BONDDIR_LIST.index(bond.GetBondDir())
        #         ])
        #         edge_feat.append([
        #             BOND_LIST.index(bond.GetBondType()),
        #             BONDDIR_LIST.index(bond.GetBondDir())
        #         ])
        #
        #     edge_index = torch.tensor([row, col], dtype=torch.long)
        #     edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        #     if self.task == 'classification':
        #         y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        #     elif self.task == 'regression':
        #         y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)
        #
        #     # ----------------- 新增：计算四种指纹并拼接 -----------------
        #
        #     # ================= 计算所需指纹 =================
        #     arrs = []
        #     if 'ecfp' in self.fingerprint_list:
        #         fp_ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.fp_radius, nBits=self.ecfp_bits)
        #         arrs.append(np.array(fp_ecfp, dtype=np.float32))
        #     if 'maccs' in self.fingerprint_list:
        #         fp_maccs = MACCSkeys.GenMACCSKeys(mol)
        #         arrs.append(np.array(fp_maccs, dtype=np.float32))
        #     if 'ap' in self.fingerprint_list:
        #         fp_ap = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=self.atompair_bits)
        #         arrs.append(np.array(fp_ap, dtype=np.float32))
        #     if 'ext' in self.fingerprint_list:
        #         fp_ext = RDKFingerprint(mol, maxPath=self.extfp_maxPath, fpSize=self.extfp_bits)
        #         arrs.append(np.array(fp_ext, dtype=np.float32))
        #     if 'torsion' in self.fingerprint_list:
        #         fp_torsion = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
        #             mol, nBits=self.torsion_bits)
        #         arrs.append(np.array(fp_torsion, dtype=np.float32))
        #     if 'avalon' in self.fingerprint_list:
        #         # 使用正确的 Avalon FP 计算方法
        #         fp_avalon = GetAvalonFP(mol, nBits=self.avalon_bits)
        #         arrs.append(np.array(fp_avalon, dtype=np.float32))
        #
        #     # 拼接所有选中指纹
        #     arr_concat = np.concatenate(arrs, axis=0)
        #     fp_tensor = torch.from_numpy(arr_concat)
        #     # ================================================
        #
        #     desc_vals = [
        #         float(Descriptors.MolWt(mol)),
        #         float(Crippen.MolLogP(mol)),
        #         float(rdMolDescriptors.CalcTPSA(mol)),
        #         float(rdMolDescriptors.CalcNumHBD(mol)),
        #         float(rdMolDescriptors.CalcNumHBA(mol)),
        #         float(rdMolDescriptors.CalcNumRings(mol)),
        #         float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        #         float(rdMolDescriptors.CalcNumHeavyAtoms(mol)),
        #     ]
        #     desc = torch.tensor(desc_vals, dtype=torch.float32).view(1, -1)  # [1,8]
        #     # ================================================
        #
        #     # mol3d = Chem.Mol(mol)
        #
        #     mol = Chem.MolFromSmiles(smiles)
        #     mol3d = Chem.AddHs(mol)
        #
        #     # ====================== FAST 3D（单构象）======================
        #     FAST_3D = True
        #     DO_MINIMIZE = True  # 想更快就 False：直接用 ETKDG 坐标，不做力场最小化 把 DO_MINIMIZE=True（只最小化 5 个构象找最低能）
        #
        #     num_confs = 5 if FAST_3D else 20
        #     maxAttempts = 20 if FAST_3D else 50
        #     maxIters = 50 if FAST_3D else 200
        #     prune_rms = 0.75 if FAST_3D else 0.5
        #
        #     geom_np = np.zeros(13, dtype=np.float32)
        #     geom_valid = 0.0
        #
        #     try:
        #         conf_ids = list(AllChem.EmbedMultipleConfs(
        #             mol3d,
        #             numConfs=num_confs,
        #             maxAttempts=maxAttempts,
        #             randomSeed=42,
        #             useRandomCoords=True,
        #             pruneRmsThresh=prune_rms,
        #         ))
        #     except Exception as e:
        #         print("EmbedMultipleConfs failed:", smiles, type(e), e)
        #         conf_ids = []
        #
        #     best_cid = None
        #     best_E = float("inf")
        #
        #     if len(conf_ids) == 0:
        #         print(f"[WARN] EmbedMultipleConfs failed: {smiles}")
        #
        #         # 备用：单构象
        #         try:
        #             params = AllChem.ETKDGv3()
        #             params.randomSeed = 42
        #             params.useRandomCoords = True
        #             st = AllChem.EmbedMolecule(mol3d, params)
        #             if st == 0:
        #                 best_cid = 0  # mol3d 只有一个 confId=0
        #         except Exception as e:
        #             print("[WARN] EmbedMolecule failed 单构象失败:", smiles, type(e), e)
        #             best_cid = None
        #
        #     else:
        #         if not DO_MINIMIZE:
        #             # ✅ 极快：直接取第一个构象
        #             best_cid = conf_ids[0]
        #         else:
        #             # ✅ 找最低能构象（只算能量，不算几何特征）
        #             for cid in conf_ids:
        #                 E = float("inf")
        #                 try:
        #                     if AllChem.MMFFHasAllMoleculeParams(mol3d):
        #                         props = AllChem.MMFFGetMoleculeProperties(mol3d, mmffVariant='MMFF94s')
        #                         ff = AllChem.MMFFGetMoleculeForceField(mol3d, props, confId=cid)
        #                     else:
        #                         ff = AllChem.UFFGetMoleculeForceField(mol3d, confId=cid)
        #
        #                     if ff is not None:
        #                         ff.Minimize(maxIts=maxIters)
        #                         E = float(ff.CalcEnergy())
        #                 except Exception:
        #                     pass
        #
        #                 if E < best_E:
        #                     best_E = E
        #                     best_cid = cid
        #
        #     # ✅ 只对 best_cid 计算一次 13 维几何特征
        #     if best_cid is None:
        #         geom_np = np.zeros(13, dtype=np.float32)
        #         geom_valid = 0.0
        #     else:
        #         try:
        #             conf = mol3d.GetConformer(best_cid)
        #
        #             # --- bond lengths ---
        #             bond_lengths = []
        #             for bond in mol3d.GetBonds():
        #                 i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        #                 bond_lengths.append(rdMolTransforms.GetBondLength(conf, i, j))
        #             mean_bl = float(np.mean(bond_lengths)) if bond_lengths else 0.0
        #             std_bl = float(np.std(bond_lengths)) if bond_lengths else 0.0
        #             max_bl = float(np.max(bond_lengths)) if bond_lengths else 0.0
        #
        #             # --- bond angles ---
        #             bond_angles = []
        #             for j in range(mol3d.GetNumAtoms()):
        #                 nbrs = [n.GetIdx() for n in mol3d.GetAtomWithIdx(j).GetNeighbors()]
        #                 if len(nbrs) < 2:
        #                     continue
        #                 for a in range(len(nbrs)):
        #                     for b in range(a + 1, len(nbrs)):
        #                         i, k = nbrs[a], nbrs[b]
        #                         bond_angles.append(rdMolTransforms.GetAngleDeg(conf, i, j, k))
        #             mean_angle = float(np.mean(bond_angles)) if bond_angles else 0.0
        #             std_angle = float(np.std(bond_angles)) if bond_angles else 0.0
        #             max_angle = float(np.max(bond_angles)) if bond_angles else 0.0
        #
        #             # --- dihedrals ---
        #             dihedrals = []
        #             for bond in mol3d.GetBonds():
        #                 j = bond.GetBeginAtomIdx()
        #                 k = bond.GetEndAtomIdx()
        #                 nbrs_j = [n.GetIdx() for n in mol3d.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
        #                 nbrs_k = [n.GetIdx() for n in mol3d.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
        #                 for i in nbrs_j:
        #                     for l in nbrs_k:
        #                         dihedrals.append(rdMolTransforms.GetDihedralDeg(conf, i, j, k, l))
        #
        #             if dihedrals:
        #                 ph = np.deg2rad(np.array(dihedrals, dtype=np.float32))
        #                 mean_sin = float(np.mean(np.sin(ph)))
        #                 mean_cos = float(np.mean(np.cos(ph)))
        #                 R = float(np.hypot(mean_sin, mean_cos)) + 1e-8
        #                 circ_std = float(np.sqrt(-2.0 * np.log(R)))
        #             else:
        #                 mean_sin = mean_cos = circ_std = 0.0
        #
        #             # --- PMI ---
        #             try:
        #                 pmi1 = float(Descriptors3D.PMI1(mol3d, confId=best_cid))
        #                 pmi2 = float(Descriptors3D.PMI2(mol3d, confId=best_cid))
        #                 pmi3 = float(Descriptors3D.PMI3(mol3d, confId=best_cid))
        #             except TypeError:
        #                 pmi1 = float(Descriptors3D.PMI1(mol3d))
        #                 pmi2 = float(Descriptors3D.PMI2(mol3d))
        #                 pmi3 = float(Descriptors3D.PMI3(mol3d))
        #
        #             # --- Rg ---
        #             pos = conf.GetPositions()
        #             centroid = pos.mean(axis=0)
        #             d = np.linalg.norm(pos - centroid, axis=1)
        #             Rg = float(np.sqrt(np.mean(d ** 2))) if len(d) > 0 else 0.0
        #
        #             geom_np = np.array([
        #                 mean_bl, std_bl, max_bl,
        #                 mean_angle, std_angle, max_angle,
        #                 mean_sin, mean_cos, circ_std,
        #                 pmi1, pmi2, pmi3,
        #                 Rg
        #             ], dtype=np.float32)
        #
        #             geom_valid = 1.0
        #         except Exception as e:
        #             print("[WARN] geom compute failed:", smiles, type(e), e)
        #             geom_np = np.zeros(13, dtype=np.float32)
        #             geom_valid = 0.0
        #
        #     geom = torch.from_numpy(geom_np).float().view(1, -1)  # [1,13]
        #     geom_mask = torch.tensor([[geom_valid]], dtype=torch.float32)  # [1,1]
        #     # ===============================================================
        #
        #
        #     data = Data(x=x_node, y=y,z=z, edge_index=edge_index, edge_attr=edge_attr)
        #
        #     data.fp = fp_tensor  # 新增一个属性：data.fp 形状是 [总指纹维度]
        #     data.desc = desc
        #     data.geom = geom
        #     data.geom_mask = geom_mask
        #
        #     data_list.append(data)
        #
        # print(f'Processed {len(data_list)} samples for dataset "{self.dataset}" (graphs + fingerprints/desc). Saving to: {self.processed_paths[0]}')
        # # 整理数据并保存
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])



    @property
    def raw_file_names(self):
        # 重要：不要返回假的 raw 文件名，否则 PyG 会以为必须存在 raw 文件
        return []

    @property
    def processed_file_names(self):
        return [self.dataset + '_pyg.pt']

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed_dir")

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def __len__(self):
        return len(self.smiles_data)

    # 添加一个新的方法用于标准化desc特征
    def zscore_transform(self, data):
        # desc z-score
        if hasattr(self, 'desc_mean') and hasattr(self, 'desc_std'):
            data.desc = (data.desc - self.desc_mean) / self.desc_std
        # geom z-score
        if hasattr(self, 'geom_mean') and hasattr(self, 'geom_std'):
            data.geom = (data.geom - self.geom_mean) / self.geom_std

        return data

    # 添加一个辅助方法用于计算简化的几何特征
    def _compute_simplified_geom_features(self, mol3d, conf):
        """计算简化的几何特征，用于无法生成3D构象的情况"""
        try:
            # --- bond lengths (估算值) ---
            bond_lengths = []
            for bond in mol3d.GetBonds():
                # 简单估算键长，基于原子类型
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                # 使用典型键长值
                bt = bond.GetBondType()
                if bt == BT.SINGLE:
                    mean_bl = 1.5
                elif bt == BT.DOUBLE:
                    mean_bl = 1.3
                elif bt == BT.TRIPLE:
                    mean_bl = 1.2
                else:  # AROMATIC
                    mean_bl = 1.4
                bond_lengths.append(mean_bl)

            mean_bl = float(np.mean(bond_lengths)) if bond_lengths else 0.0
            std_bl = float(np.std(bond_lengths)) if bond_lengths else 0.0
            max_bl = float(np.max(bond_lengths)) if bond_lengths else 0.0

            # --- 其他特征使用默认值 ---
            # 对于无法计算的特征，使用合理的默认值
            std_angle = 10.0   # 典型键角标准差
            mean_angle = 109.5 # 典型键角均值（四面体角度）
            max_angle = 180.0  # 最大键角

            mean_sin = 0.0
            mean_cos = 1.0
            circ_std = 0.0

            # PMI使用分子量相关估算值
            mol_weight = Descriptors.MolWt(mol3d)
            pmi1 = mol_weight * 0.1
            pmi2 = mol_weight * 0.2
            pmi3 = mol_weight * 0.3

            # 回转半径估算
            Rg = mol_weight * 0.01

            g = np.array([
                mean_bl, std_bl, max_bl,
                mean_angle, std_angle, max_angle,
                mean_sin, mean_cos, circ_std,
                pmi1, pmi2, pmi3,
                Rg
            ], dtype=np.float32)

            return g
        except Exception:
            # 如果简化方法也失败，返回None
            return None


class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_path, target, task, splitting,fingerprint_list=None,
                 fp_radius=2,
                 ecfp_bits=2048,
                 maccs_bits=167,
                 ap_bits=2048,
                 ext_bits=2048,
                 extfp_maxPath=5,
                 torsion_bits=2048,
                 avalon_bits=1024
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        assert splitting in ['random', 'scaffold', 'random_scaffold']
        self.fingerprint_list = fingerprint_list or AVAILABLE_FPS

        self.fp_radius = fp_radius
        self.ecfp_bits = ecfp_bits
        self.maccs_bits = maccs_bits
        self.ap_bits = ap_bits
        self.ext_bits = ext_bits
        self.extfp_maxPath = extfp_maxPath
        self.torsion_bits = torsion_bits
        self.avalon_bits = avalon_bits

    def get_data_loaders(self):
        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task,
                                       fingerprint_list=self.fingerprint_list,
                                       fp_radius=self.fp_radius,
                                       ecfp_bits=self.ecfp_bits,
                                       maccs_bits=self.maccs_bits,
                                       ap_bits=self.ap_bits,
                                       ext_bits=self.ext_bits,
                                       extfp_maxPath=self.extfp_maxPath,
                                       torsion_bits=self.torsion_bits,
                                       avalon_bits=self.avalon_bits


                                       )
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_data_loaders111(self):
        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task,
                                       fingerprint_list=self.fingerprint_list,
                                       fp_radius=self.fp_radius,
                                       ecfp_bits=self.ecfp_bits,
                                       maccs_bits=self.maccs_bits,
                                       ap_bits=self.ap_bits,
                                       ext_bits=self.ext_bits,
                                       extfp_maxPath=self.extfp_maxPath,
                                       torsion_bits=self.torsion_bits,
                                       avalon_bits=self.avalon_bits
                                       )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, drop_last=False
        )
        return test_loader


    def get_train_validation_data_loaders(self, train_dataset,seed=42):
        import os, random, numpy as np, torch
        # os.environ["PYTHONHASHSEED"] = str(seed)   ### 设置全局随机种子 这些都是后加的锁定随机种子 42
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        g = torch.Generator()
        g.manual_seed(seed)

        random.seed(seed)



        if self.splitting == 'random':
            # 设置固定的随机种子
            np.random.seed(seed)  # 设置随机种子为 42，确保每次划分一致
            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size,seed)
        elif self.splitting == 'random_scaffold':

            train_idx, valid_idx, test_idx = random_scaffold_split(train_dataset, self.valid_size, self.test_size, seed)


        # ====== 1) 用训练集样本算 desc、geom 的 mean/std（8维）============================================================
        def fit_desc_geom_zscore(dataset, train_indices):
            # desc: 用所有训练样本
            X_desc = torch.cat([dataset[int(i)].desc for i in train_indices], dim=0)  # [N,8]
            desc_mean = X_desc.mean(dim=0, keepdim=True)
            desc_std = X_desc.std(dim=0, keepdim=True, unbiased=False)
            desc_std = torch.clamp(desc_std, min=1e-6)

            # geom: 只用 geom_mask=1 的训练样本（没有 mask 就退化为用全部）
            geom_list = []
            for i in train_indices:
                d = dataset[int(i)]
                if hasattr(d, "geom_mask"):
                    if float(d.geom_mask.view(-1)[0]) < 0.5:
                        continue
                geom_list.append(d.geom)

            if len(geom_list) == 0:
                # 极端情况：训练集里没有一个有效 3D
                geom_mean = torch.zeros(1, 13)
                geom_std = torch.ones(1, 13)
            else:
                X_geom = torch.cat(geom_list, dim=0)  # [N_valid,13]
                geom_mean = X_geom.mean(dim=0, keepdim=True)
                geom_std = X_geom.std(dim=0, keepdim=True, unbiased=False)
                geom_std = torch.clamp(geom_std, min=1e-6)

            return desc_mean, desc_std, geom_mean, geom_std

        desc_mean, desc_std, geom_mean, geom_std = fit_desc_geom_zscore(train_dataset, train_idx)
        print('desc_mean:', desc_mean, 'desc_std:', desc_std)
        print('geom_mean:', geom_mean, 'geom_std:', geom_std)
        # 将均值和标准差保存到数据集中
        train_dataset.desc_mean = desc_mean
        train_dataset.desc_std = desc_std
        train_dataset.geom_mean = geom_mean
        train_dataset.geom_std = geom_std

        # train_dataset.transform = train_dataset.zscore_transform

        # ====== 2) 给 dataset 挂一个 transform：每次取样本时自动做 z-score =============================================
        # 使用数据集类的方法而不是局部函数
        train_dataset.transform = train_dataset.zscore_transform

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx, generator=g)
        valid_sampler = SubsetRandomSampler(valid_idx, generator=g)
        test_sampler = SubsetRandomSampler(test_idx, generator=g)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=True,
            worker_init_fn=seed_worker, generator=g  ### 这一行也是锁定的随机种子。
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False,
        worker_init_fn=seed_worker, generator=g
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False,
        worker_init_fn=seed_worker, generator=g
        )

        return train_loader, valid_loader, test_loader
