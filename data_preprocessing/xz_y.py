# 通过线性回归分析 x_angle_1 和 z_angle_1 对 y_angle_1 的关系，并输出回归方程及拟合优度。
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取 CSV 文件
df = pd.read_csv('data.csv')

# 数据预处理示例
df = df.dropna()  # 删除缺失值
# df['某列'] = df['某列'].astype(float)  # 类型转换

df_section = df[(df.index >= 6000) & (df.index <= 12000)].reset_index(drop=True)

# 构建自变量和因变量
X = df_section[['x_angle_1', 'z_angle_1']]
y = df_section['y_angle_1']

# 建立线性回归模型
model = LinearRegression()
model.fit(X, y)

# 输出回归系数和截距
print(f"回归方程: y_angle_1 = {model.coef_[0]:.4f} * x_angle_1 + {model.coef_[1]:.4f} * z_angle_1 + {model.intercept_:.4f}")
print(f"拟合优度 R²: {model.score(X, y):.2f}")