import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('data.csv')
df = df.dropna()

df_section = df[(df.index >= 6000) & (df.index <= 12000)]

# 计算相关系数
corr_matrix = df_section[['x_angle_1', 'y_angle_1', 'z_angle_1']].corr()
print("相关系数矩阵：")
print(corr_matrix)

# 绘制相关性热力图
plt.figure(figsize=(6,4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('X、Y、Z角度相关性热力图')
plt.show()