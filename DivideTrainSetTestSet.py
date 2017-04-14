import pandas as pd
from sklearn.cross_validation import train_test_split#分割数据
#读取数据
df_wine = pd.read_csv('wine.text')
#设置列名
df_wine.columns = ['ClassLabel','Alcohol','MalicAcid','Ash','AlcalinityOfAsh','Magnesium','TotalPhenols','Flavanoids','NonflavanoidPhenols','Proanthocyanins','ColorIntensity','Hue','ofDilutedWine','Proline']
#取出特征及结果集
X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
#分成训练集和测试集
X_train,Xtest,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
print(df_wine)

print(X_train,Xtest,y_train,y_test)

