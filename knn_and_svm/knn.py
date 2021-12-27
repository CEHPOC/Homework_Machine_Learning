from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
import time
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


raw_data=pd.read_csv('heart.csv')
raw_dataX=raw_data.drop(['sex','cp','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal','target'],axis=1).to_numpy()
raw_dataY=raw_data['target'].to_numpy()
#raw_dataX, raw_dataY=make_blobs(n_samples=500,n_features=2,centers=3)
#raw_dataX, raw_dataY=make_circles(n_samples=500,noise=0.15)
#raw_dataX, raw_dataY=make_moons(n_samples=500,noise=0.5)
#plt.scatter(raw_dataX[:,0],raw_dataX[:,1],c=raw_dataY)

scaler=StandardScaler()
scaler.fit(raw_dataX)
scaled_data=scaler.transform(raw_dataX)

scaled_data=pd.DataFrame(scaled_data)
x=scaled_data
y=raw_dataY
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x,y,test_size=0.35)

'''
error_rates=[]
for i in np.arange(1, 51):
    new_model = KNeighborsClassifier(n_neighbors = i)
    new_model.fit(x_training_data, y_training_data)
    new_predictions = new_model.predict(x_test_data)
    error_rates.append(np.mean(new_predictions != y_test_data))
plt.plot(error_rates)
'''

modelSVM = SVC()
start_time=time.time()
modelSVM.fit(x_training_data,y_training_data)
SVMtrainingtime=time.time()-start_time
start_time=time.time()
predictionsSVM=modelSVM.predict(x_test_data)
SVMpredicttime=time.time()-start_time

model = KNeighborsClassifier(n_neighbors=17)
start_time=time.time()
model.fit(x_training_data,y_training_data)
KNNtrainingtime=time.time()-start_time
start_time=time.time()
predictions=model.predict(x_test_data)
KNNpredicttime=time.time()-start_time
x_test_data=x_test_data.to_numpy()

error_ratesSVM=0
for i in range(predictionsSVM.size):
    if predictionsSVM[i] != y_test_data[i]:
        error_ratesSVM+=1
error_ratesSVM=error_ratesSVM/predictionsSVM.size

error_ratesKNN=0
for i in range(predictions.size):
    if predictions[i] != y_test_data[i]:
        error_ratesKNN+=1
error_ratesKNN=error_ratesKNN/predictions.size

plt.subplot(1,2,1)
plt.scatter(x_test_data[:,0],x_test_data[:,1],c=predictionsSVM)
plt.text(-2,-2.5,'error rates SVM:'+str(math.floor(error_ratesSVM*1000)/1000),fontsize=6)
plt.text(-2,-2.6,'SVM training time:'+str(math.floor(SVMtrainingtime*1000)/1000),fontsize=6)
plt.text(-2,-2.7,'SVM predict time:'+str(math.floor(SVMpredicttime*1000)/1000),fontsize=6)
plt.subplot(1,2,2)
plt.scatter(x_test_data[:,0],x_test_data[:,1],c=predictions)
plt.text(-2,-2.5,'error rates KNN:'+str(math.floor(error_ratesKNN*1000)/1000),fontsize=6)
plt.text(-2,-2.6,'KNN training time:'+str(math.floor(KNNtrainingtime*1000)/1000),fontsize=6)
plt.text(-2,-2.7,'KNN predict time:'+str(math.floor(KNNpredicttime*1000)/1000),fontsize=6)

print("error rates SVM: "+str(error_ratesSVM))
print("error rates KNN: "+str(error_ratesKNN))
print(SVMtrainingtime)
print(SVMpredicttime)
print(KNNtrainingtime)
print(KNNpredicttime)

plt.show()

