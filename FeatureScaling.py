import pandas as pd
from sklearn.cross_validation import train_test_split#分割数据
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#读取数据
df_wine = pd.read_csv('wine.text')
#设置列名
df_wine.columns = ['ClassLabel','Alcohol','MalicAcid','Ash','AlcalinityOfAsh','Magnesium','TotalPhenols','Flavanoids','NonflavanoidPhenols','Proanthocyanins','ColorIntensity','Hue','ofDilutedWine','Proline']
#取出特征及结果集
X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
#分成训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#归一化
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test = mms.fit_transform(X_test)

#标准化 比较好的选择
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

print(X_train_norm , '\n',X_train_std)