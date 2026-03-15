import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Load dataset
#X=studya hours
#y=marks
X = np.array([1, 2, 3, 4, 5]).reshape(-1,1)
Y = np.array([50, 55, 60, 65, 70])

# 2. Train the model
model = LinearRegression()
model.fit(X, Y)

# 3. Find coefficient and intercept
c = model.coef_
i = model.intercept_

print("Coefficient:", c)
print("Intercept:", i)

#predict 6 hours 
hours=np.array([[6]])
predict=model.predict(hours)

print("predicte value :",predict)