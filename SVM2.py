import numpy as np
from sklearn import datasets#数据源
from sklearn.cross_validation import train_test_split#随机分数据集为训练集及测试集
from sklearn.preprocessing import StandardScaler#标准化模型 优化数据提升效率
from sklearn.metrics import accuracy_score#评估模型好坏
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron#神经元
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier#算法集合
from matplotlib.colors import ListedColormap#设置绘图颜色
import matplotlib.pyplot as plt#绘图库

np.random.seed(0)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)
y_xor = np.where(y_xor,1,-1)
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

svm = SVC(kernel='rbf',random_state=0,gamma=0.1,C=10.0)
svm.fit(X_xor,y_xor)
plot_decision_regions(X_xor,y_xor,classifier=svm)
plt.ylim(-3)
plt.legend(loc='upper left')
plt.show()