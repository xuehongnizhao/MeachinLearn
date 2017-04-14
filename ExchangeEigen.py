import pandas as pd
from sklearn.cross_validation import train_test_split#分割数据
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

#读取数据
df_wine = pd.read_csv('wine.text')
#设置列名
df_wine.columns = ['ClassLabel','Alcohol','MalicAcid','Ash','AlcalinityOfAsh','Magnesium','TotalPhenols','Flavanoids','NonflavanoidPhenols','Proanthocyanins','ColorIntensity','Hue','ofDilutedWine','Proline']
#取出特征及结果集
X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
#分成训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#标准化 比较好的选择
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

cov_mat = np.cov(X_train_std.T) #协方差矩阵
eigen_vals,eigen_vecs = np.linalg.eig(cov_mat) #特征值 特征向量
eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals)) ]
eigen_pairs.sort(reverse=True)
print(eigen_pairs)
#选择特征向量时保证特征子集包含90%的方差 这里选择了两个 包含60%
w = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
x_train_pca = X_train_std.dot(w)
colors = ['r','b','g']
markers = ['s','x','o']

for l,c,m in zip(np.unique(y_train),colors,markers):
	plt.scatter(x_train_pca[y_train==l,0],x_train_pca[y_train==l,1],c=c,label=l,marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.show()
