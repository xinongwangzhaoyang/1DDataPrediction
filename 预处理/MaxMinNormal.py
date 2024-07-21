import pandas as pd
import matplotlib.pyplot as plt
# 此代码对数据进行最大最小归一化
# 读取Excel文件
input_file = './../data.xlsx'  # 替换为你的文件名
df = pd.read_excel(input_file).drop(['SOM', 'name'], axis=1)

# 定义归一化函数
def normalize_row(row):
    min_val = row.min()
    max_val = row.max()
    return (row - min_val) / (max_val - min_val)

# 对每一行进行归一化
df_normalized = df.apply(normalize_row, axis=1)

# 保存归一化后的数据到新的Excel文件
output_file = 'normalized_data.xlsx'  # 替换为你希望保存的文件名
df_normalized.to_excel(output_file, index=False)

# 绘制所有行的归一化前数据
plt.figure(figsize=(14, 10))

for index, row in df.iterrows():
    plt.plot(row, marker='o', label=f'Row {index}', alpha=0.7)

plt.title('Original Data - All Rows')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.tight_layout()
plt.savefig('original_data.png')  # 保存为PNG文件
plt.show()

# 绘制所有行的归一化后数据
plt.figure(figsize=(14, 10))

for index, row in df_normalized.iterrows():
    plt.plot(row, marker='o', label=f'Row {index}', alpha=0.7)

plt.title('Normalized Data - All Rows')
plt.xlabel('Feature Index')
plt.ylabel('Normalized Value')
plt.tight_layout()
plt.savefig('normalized_data.png')  # 保存为PNG文件
plt.show()
