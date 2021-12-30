import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

#Read CSV File
df=pd.read_csv('C:\\Users\\Hp\\Desktop\\Dataset\\weatherAUS.csv')
pd.set_option('display.max_columns',25)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
df=df[['MinTemp','MaxTemp','Rainfall','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am', 'Humidity3pm' ,'Pressure9am','Pressure3pm',  'Cloud9am',  'Cloud3pm',  'Temp9am',  'Temp3pm', 'RainToday','RainTomorrow']]
df.fillna(method='ffill',inplace=True)
df.fillna(0,inplace=True)
x=df[['MinTemp','MaxTemp','Rainfall','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am', 'Humidity3pm' ,'Pressure9am','Pressure3pm',  'Cloud9am',  'Cloud3pm',  'Temp9am',  'Temp3pm']]
y=df[['RainToday']]
lbl1=LabelEncoder()
lbl2=LabelEncoder()
a=lbl1.fit_transform(y['RainToday'])
y=y.drop(['RainToday'],axis='columns')
y['RainToday']=a
#y['RainTomorrow']=b
lbl3=LabelEncoder()
lbl4=LabelEncoder()
lbl5=LabelEncoder()
a=lbl3.fit_transform(x['WindGustDir'])
b=lbl4.fit_transform(x['WindDir9am'])
c=lbl5.fit_transform(x['WindDir3pm'])
x=x.drop(['WindGustDir','WindDir9am','WindDir3pm'],axis='columns')
x['WindGustDir']=a
x['WindDir9am']=b
x['WindDir3pm']=c
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=109)
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
accuracy1=metrics.accuracy_score(y_test, y_pred)

y=df[['RainTomorrow']]
a=lbl1.fit_transform(y['RainTomorrow'])
y=y.drop(['RainTomorrow'],axis='columns')
y['RainTomorrow']=a
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=109)
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
accuracy2=metrics.accuracy_score(y_test, y_pred)
print('Accuracy for RainToday is ',accuracy1*100,'%')
print('Accuracy for RainTomorrow is ',accuracy2*100,'%')
