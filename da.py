import pandas as pd

file_path = 'data10m.csv'
df = pd.read_csv(file_path)

print("处理后的文件内容（前五行）：")
print(df.head())

print("\n处理后的文件内容（后五行）：")
print(df.tail())