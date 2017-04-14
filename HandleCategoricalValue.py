import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder #自动转化类别
from sklearn.preprocessing import OneHotEncoder #独热编码
#定义一组分类数据
df = pd.DataFrame([['green','m',10.1,'class1'],
	['red','l',13.5,'class2'],
	['blue','xl',15.3,'class1']])
df.columns = ['color','size','price','classlabel']
print(df)

#映射
size_mapping = {
	'xl':3,
	'l':2,
	'm':1
}
df['size'] = df['size'].map(size_mapping)
print(df)

# #手动对类别进行编码
# class_mapping = {
# 	#这段代码不好理解的话看看闭包
# 	label:idx for idx ,label in enumerate(np.unique(df['classlabel']))
# }

# df['classlabel'] = df['classlabel'].map(class_mapping)

# print(df)

# #自动对类别进行编码 所有类都可以用这种方法编码 但是对于无序的类型 会出现0,1,2的情况这种情况在算法中会导致某个无序特征大于另一个 这是不对的
# class_le = LabelEncoder()

# df['classlabel']=class_le.fit_transform(df['classlabel'].values)
# print(df,'编码后的 classlabel')
# #反编码
# print(class_le.inverse_transform(df['classlabel']),'反编码后的 classlabel 列')

# #c出现问题的无序类
# X = df.values
# color_le = LabelEncoder()
# X[:,0] = class_le.fit_transform(X[:,0])#不一样的写法 转换成矩阵也可以使用 labelEncoder 方法
# print(X,'颜色类')

# #独热编码 用多列表示一个列中的不同情况 '稀疏矩阵'
# ohe = OneHotEncoder(categorical_features=[0]) #categorical_features 设置需要编码的列
# y = ohe.fit_transform(X).toarray()
# print(y)

#以上均可以用一个方式转换
X = pd.get_dummies(df)
print(X)