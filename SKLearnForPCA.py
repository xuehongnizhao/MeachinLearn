import pandas as pd
from sklearn.cross_validation import train_test_split#分割数据
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
	#设置标记的形状和颜色
	markers = ('s','v','o','^','x')
	colors = ('red','blue','lightgreen','gray','cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	#坐标系
	x1_min,x1_max = X[:,0].min()-1,X[:,0].max()+1
	x2_min,x2_max = X[:,1].min()-1,X[:,1].max()+1
	#arange(0,2,1) = [0,1,2] resolution 梯度  为什么这么写? meshgrid 创建两个矩阵 表示点坐标 恩应该是这个样子
	xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
	Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1,xx2,Z,slpha=0.4,cmap=cmap)
	plt.xlim(xx1.min(),xx1.max())
	plt.ylim(xx2.min(),xx2.max())

	#样本点
	X_test,y_test = X[test_idx,:],y[test_idx]
	for idx,cl in enumerate(np.unique(y)):	
		#xy 坐标点, alpha 透明度, c 标记颜色, marker 标记形状 label 标记名称plt.legend()设置位置,linewidth 线宽, edgecolors 线颜色
		plt.scatter(x=X[y == cl,0],y=X[y==cl,1],alpha=1,c=cmap(idx),marker=markers[idx],label=cl,linewidth=1,edgecolors='black')
	#高亮测试集合
	if test_idx:
		X_test,y_test = X[test_idx,:],y[test_idx]
		plt.scatter(X_test[:,0],X_test[:,1],alpha=1,linewidth=1,marker='o',s=55,label='test set',edgecolors='black',c='white')

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
X_test_std = stdsc.transform(X_test)

pca = PCA(n_components=2)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca,y_train)
plot_decision_regions(X_train_pca,y_train,lr)
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.legend(loc='best')

plt.figure(2)
plot_decision_regions(X_test_pca,y_test,lr)
plt.legend(loc='best')
plt.show()
