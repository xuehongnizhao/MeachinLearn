#如果联系集的表现比测试集表现好太多 可能是过拟合了
import pandas as pd
from sklearn.cross_validation import train_test_split#分割数据
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

for c in np.arange(-4,6):
	#训练
	lr = LogisticRegression(penalty='l1',C=10.0**c,random_state=0)
	lr.fit(X_train_std,y_train)
	weights.append(lr.coef_[1])
	params.append(10.0 ** c)
weights = np.array(weights)
for column,color in zip(range(weights.shape[1]),colors):
	plt.plot(params,weights[:,column],label=df_wine.columns[column+1],color=color)
plt.axhline(0,color='black',linestyle='--',linewidth=3)
plt.xlim([10.0**(-5),10.0**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
plt.show()
