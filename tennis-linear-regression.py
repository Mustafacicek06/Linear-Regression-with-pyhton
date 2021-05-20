# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriFrame = pd.read_csv("odev_tenis.csv")

veriFrame

# sonuc olarak play mı değil mi istediğimiz için alıyorum
play = veriFrame.iloc[:,-1:].values
windy = veriFrame.iloc[:,4:5].values

outlook = veriFrame.iloc[:,0:1].values

# tek tek label encoder yapmak yerine şunu yapabiliriz
# hepsine birden label encoder yapar
from sklearn import preprocessing
veriFrame2 = veriFrame.apply(preprocessing.LabelEncoder().fit_transform)

#windy'e onehotencoder yapmamız gerek
o = veriFrame2.iloc[:,:1]

from sklearn import preprocessing

ohe = preprocessing.OneHotEncoder()
o = ohe.fit_transform(o).toarray()
o


outData = pd.DataFrame(data=o,index=range(14),columns=["o","r","s"])

lastData = pd.concat([outData,veriFrame.iloc[:,1:3]],axis=1)
lastData = pd.concat([veriFrame2.iloc[:,-2:],lastData],axis=1)
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(lastData.iloc[:,:-1],lastData.iloc[:,-1:],test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression


regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)



import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int),values=lastData.iloc[:,:-1],axis=1)

X_l = lastData.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(lastData.iloc[:,-1:].astype(float),X_l.astype(float)).fit()
print(model.summary())



"""
from sklearn import preprocessing

lEncoder = preprocessing.LabelEncoder()

# ne anlama geliyor :,-1 anlamadım
play[:,-1] = lEncoder.fit_transform(veriFrame.iloc[:,-1])


# windy i de sayısal verilere dönüştürüyorum

encoderWindy = preprocessing.LabelEncoder()
windy[:,0] = encoderWindy.fit_transform(veriFrame.iloc[:,3])


encoderOutlook = preprocessing.LabelEncoder()

outlook[:,0] = encoderOutlook.fit_transform(veriFrame.iloc[:,0])


# onehotencoder

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()



# numpy dizilerini dataframeye dönüştürüyoruz birleştirmek için



yeni = veriFrame.drop("windy",axis=1)
yeni = yeni.drop("play",axis=1)
yeni = yeni.drop("outlook",axis=1)

playdf= pd.DataFrame(data=play , index=range(14),columns=["play"])
windydf = pd.DataFrame(data=windy, index=range(14),columns=["windy"])
outlookdf = pd.DataFrame(data=outlook, index=range(14),columns=["sunny","overcast","rainy"])

sonuc = pd.concat([outlookdf,yeni],axis=1)
sonuc2 = pd.concat([sonuc,windydf],axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test , y_train , y_test = train_test_split(sonuc2,playdf,test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression


regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


# backward elimination 
import statsmodels.api as sm

X = np.append(arr= np.ones((14,1)).astype(int),values=sonuc2 ,axis=1,)

X_l = sonuc2.iloc[:,[0,1,2,3,4,5]].values

X_l = np.array(X_l,dtype=int)


model = sm.OLS(playdf.astype(float),X_l.astype(float)).fit()

print(model.summary())

#####################################################

X_l = sonuc2.iloc[:,[0,1,2]].values

X_l = np.array(X_l,dtype=int)


model = sm.OLS(playdf.astype(float),X_l.astype(float)).fit()

print(model.summary())




"""




