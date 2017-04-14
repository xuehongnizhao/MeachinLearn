from io import StringIO
import pandas as pd  
import numpy as np
from sklearn.preprocessing import Imputer
csv_data = '''A,B,C,d,e
			1.0,2.0,3.0,4.0,5.0
			5.0,6.0,,8.0,
			0.0,11.0,12.0,13.0,'''

csv_data = str(csv_data)#函数str() 用于将值转化为适于人阅读的形式
df = pd.read_csv(StringIO(csv_data))#read_csv导入特定格式的函数.StringIO 在内存中读写字符串
print(df,'完整数据')
# print(df.isnull().sum())
# print(df.values)# value 是 numpy 的方法 将数据转化为矩阵\

#消除缺失值的特征或者样本

# print(df.dropna(),'删除缺失值的样本或特征')
# print(df.dropna(axis=1),'删除缺失值的样本或特征 去掉列')
# print(df.dropna(how='all'),'删除缺失值的样本或特征 只去掉均缺失的行')
# print(df.dropna(thresh=4),'删除缺失值的样本或特征 去掉非缺失值小于4个的行')
# print(df.dropna(subset=['d']),'删除缺失值的样本或特征 删除特定列缺失的行')

#将空值替换为平均值

imr = Imputer(missing_values='NaN',strategy='mean',axis=1)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data,'添加平均值')
