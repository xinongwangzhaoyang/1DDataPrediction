"""偏最小二乘回归（Partial Least Squares Regression, PLSR）是一种统计学和机器学习中的多元数据分析方法
特别适用于处理因变量和自变量之间存在多重共线性问题的情况。该方法最早由瑞典化学家Herman
Wold于上世纪60年代提出，作为一种多变量线性回归分析技术，广泛应用于化学、环境科学、
生物医学、金融等领域，尤其在高维数据和小样本问题中表现出色。"""
# 引入库
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 读取数据
file_path = './../data.xlsx'
data = pd.read_excel(file_path)
# 打印前五行
print(data.head())
# 准备数据
y = data['SOM']
x = data.drop(['SOM','name'] , axis=1).astype('float64')
# 打印x，y的前五行
print(y.head())
print(type(x),x.head())
# 训练集、测试集划分
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state= 42)
#回归模型、参数
pls_model_setup = PLSRegression(scale=True)
param_grid = {'n_components': range(1, 20)}
# GridSearchCV优化参数、训练模型
gsearch = GridSearchCV(pls_model_setup, param_grid)
pls_model = gsearch.fit(x_train, y_train)
#对测试集做预测
pls_prediction = pls_model.predict(x_test)
#计算R2，均方差，打印并绘制散点图
pls_r2 = r2_score(y_test,pls_prediction)
pls_mse = np.sqrt(mean_squared_error(y_test,pls_prediction))
print(pls_r2,pls_mse)
# 绘制散点图
plt.scatter(y_test,pls_prediction)
plt.text(26, 36, f'R2={pls_r2}')
plt.text(26, 35, f'MSE={pls_mse}')
plt.show()



