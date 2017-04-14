#如果徐联机的表现比测试集表现好太多 可能是过拟合了
import pandas as pd
from sklearn.cross_validation import train_test_split#分割数据
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

lr = LogisticRegression(penalty='l1',c=0.1)
lr.fit(X_train_std,y_train)
print('训练集准确率',lr.score(X_train_std,y_train))
print('测试集准确率',lr.score(X_test_std,y_test))