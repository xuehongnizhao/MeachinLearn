#如果联系集的表现比测试集表现好太多 可能是过拟合了
import pandas as pd
from sklearn.cross_validation import train_test_split#分割数据
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from SBS import SBS

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

#knn 相邻的那种方法 教程上的数据和我的不一样 有可能是 sklearn 的KNN 源码更新 不用自己写sbs 了 自动使用 SBS
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train_std,y_train)
print(knn.score(X_train_std,y_train))
print(knn.score(X_test_std,y_test))
sbs = SBS(knn,k_features=1)
sbs.fit(X_train_std,y_train)
print(sbs)
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.5,1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()
k_5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k_5])

knn.fit(X_train_std,y_train)
print(knn.score(X_train_std,y_train))
print(knn.score(X_test_std,y_test))