import numpy as np
import matplotlib.pyplot as plt
x = np.array([10,9,2,15,10,16,11,16])
y = np.array([95,80,10,50,45,98,38,93])


b1 = ((x-x.mean()) * (y-y.mean())).sum() / ((x-x.mean()) ** 2).sum()

b0 = y.mean() - b1 * x.mean()
y_pred = b0 + b1 * x
print(b1,b0)
plt.plot(x,y_pred,'orange')
plt.scatter(x,y)
plt.xlabel("No of Hours")
plt.ylabel("Risk Factor")
plt.show()