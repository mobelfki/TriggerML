import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors._base
import sys


Ex = np.random.normal(0, 1, 10000)
Ey = np.random.normal(0, 1, 10000)


Ex_flat = np.random.uniform(Ex.min(), Ex.max(), 10000)
Ey_flat = np.random.uniform(Ey.min(), Ey.max(), 10000)


E = (Ex**2 + Ey**2)**0.5
E_flat = (Ex_flat**2 + Ey_flat**2)**0.5

plt.figure()

hist_settings = {'bins': 100, 'range':[0, 10], 'density': True, 'histtype': 'step'}
plt.hist(Ex, **hist_settings)
plt.hist(Ex_flat, **hist_settings)

plt.legend(['Normal', 'Uniform'], loc='best')
plt.show()
