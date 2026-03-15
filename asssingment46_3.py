import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Load dataset
#X=studya hours
#y=marks
X = np.array([
    [1,7],
    [2,6],
    [3,7],
    [4,6],
    [5,8]
])
Y = np.array([50, 55, 60, 65, 70])

# 2. Train the model
model = LinearRegression()
model.fit(X, Y)

# 3. Find coefficient and intercept
c = model.coef_
i = model.intercept_

print("Coefficient:", c)
print("Intercept:", i)

#coeeficent of both features
print("coeffecnt of study hours:",c[0])
print("coeffecnt of study hours:",c[1])
print("intercept :",i)