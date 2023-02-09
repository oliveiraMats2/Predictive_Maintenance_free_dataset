import numpy as np

from sklearn.linear_model import LogisticRegression
from scipy.special import expit

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

# time = 10 #sec
samples = 1
slot = 100 * samples
sensores = 1

periodo = 120
noise = 8

data = np.ones((slot, sensores))

data = np.transpose(np.concatenate(10*[np.random.normal(0, noise, slot),
                                    np.array([np.random.normal(x, noise, 1) for x in range(120)]).squeeze(1),
                                    np.random.normal(periodo, noise, slot),
                                    np.array([np.random.normal(x, noise, 1) for x in range(120,
                                                                                           -1, -1)]).squeeze(1)]))
# data = np.transpose(np.random.normal(0, noise, slot))

# data = preprocessing.normalize(data)
print(data.shape)
x = np.linspace(-5, 5, num=100)[:, None]
# y = -0.5 + 2.2*x +0.3*x**3+ 2*np.random.randn(100,1)

tam = 4000

y = data[:tam]

x = np.linspace(0, 10, num=tam)[:, None]

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

plt.scatter(x, y)
plt.plot(x[:, 0], y_pred, 'r')
plt.legend(['Predicted line', 'Observed data'])
plt.show()
