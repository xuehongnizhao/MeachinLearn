import numpy as np
from sklearn import datasets#数据源
from sklearn.cross_validation import train_test_split#随机分数据集为训练集及测试集
from sklearn.preprocessing import StandardScaler#标准化模型 优化数据提升效率
from sklearn.metrics import accuracy_score#评估模型好坏
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import Perceptron#神经元
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap#设置绘图颜色
import matplotlib.pyplot as plt#绘图库

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)#划分数据
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)#标准化数据
X_test_std = sc.transform(X_test)

# ppn =  Perceptron(n_iter=40, eta0=0.1,random_state=0)
# ppn.fit(X_train_std,y_train)#训练训练集
# y_pred = ppn.predict(X_test_std)#通过训练好的神经元 ppn 预测测试集
# print(accuracy_score(y_test,y_pred))#评估本次训练的准确率
# lr = LogisticRegression(C=1000.0,random_state=0)
# lr.fit(X_train_std,y_train)
# print(lr.predict_proba(X_test_std[0,:]))#通过训练好的神经元 lr 预测测试集
svm = SVC(kernel='rbf',C=1.0,random_state=0,gamma=0.2)
svm.fit(X_train_std,y_train)
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

X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std,y=y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('length')
plt.ylabel('width')
#左上角表
plt.legend(loc='upper left')
plt.show()