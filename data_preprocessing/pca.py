# 对三组角度数据进行主成分分析（PCA），输出主成分方差贡献率和方向，并绘制第一主成分随样本序号变化的曲线。
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('data.csv')
df = df.dropna()

df_section = df[(df.index >= 6000) & (df.index <= 12000)].reset_index(drop=True)



angles = df_section[['x_angle_1', 'y_angle_1', 'z_angle_1']]
pca = PCA()
pca.fit(angles)
print("各主成分方差贡献率：", pca.explained_variance_ratio_)
print("主成分方向：\n", pca.components_)

# 只保留第一个主成分
angles_pca = pca.transform(angles)
first_component = angles_pca[:, 0]

plt.figure(figsize=(8,4))
plt.plot(df_section.index, first_component)
plt.title('第一主成分随序号变化曲线')
plt.xlabel('样本序号')
plt.ylabel('第一主成分值')
plt.show()