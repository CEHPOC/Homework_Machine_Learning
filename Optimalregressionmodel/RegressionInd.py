from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics import r2_score

def func(x,y,degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(x)
    poly.fit(X_poly, y)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model.predict(poly.fit_transform(x))


x=np.arange(-10,10)
y=[random.randint(-10000,10000) for i in x]

xexp=np.exp(x)



model=LinearRegression()
model2=LinearRegression()
xexp=pd.DataFrame(xexp)
x=pd.DataFrame(x)
y=pd.DataFrame(y)

model.fit(x,y)
model2.fit(xexp,y)
plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
plt.scatter(x,y,c='blue')
plt.plot(x.to_numpy(),model.predict(x),c='red')
plt.subplot(2,3,2)
plt.scatter(x,y,c='blue')
plt.plot(x.to_numpy(),func(x,y,2),c='green')
plt.subplot(2,3,3)
plt.scatter(x,y,c='blue')
plt.plot(x.to_numpy(),func(x,y,3),c='pink')
plt.subplot(2,3,4)
plt.scatter(x,y,c='blue')
plt.plot(x.to_numpy(),func(x,y,4),c='brown')
plt.subplot(2,3,5)
plt.scatter(x,y,c='blue')
plt.plot(x.to_numpy(),func(x,y,5),c='black')
plt.subplot(2,3,6)
plt.scatter(x,y,c='blue')
plt.plot(x.to_numpy(),model2.predict(xexp),c='yellow')
print(r2_score(y,model2.predict(xexp)))
print(r2_score(y,model.predict(x)))
print(r2_score(y,func(x,y,2)))
print(r2_score(y,func(x,y,3)))
print(r2_score(y,func(x,y,4)))
print(r2_score(y,func(x,y,5)))

max=0
maxi=0
if r2_score(y,model.predict(x)) > max:
    maxi=1
    max=r2_score(y,model.predict(x))
if r2_score(y,func(x,y,2)) > max:
    maxi=2
    max=r2_score(y,func(x,y,2))
if r2_score(y,func(x,y,3)) > max:
    maxi=3
    max=r2_score(y,func(x,y,3))
if r2_score(y,func(x,y,4)) > max:
    maxi=4
    max=r2_score(y,func(x,y,4))
if r2_score(y,func(x,y,5)) > max:
    maxi=5
    max=r2_score(y,func(x,y,5))
if r2_score(y,model2.predict(xexp)) > max:
    maxi='exp'

print("Оптимальная регрессия степени "+str(maxi))

plt.show()

