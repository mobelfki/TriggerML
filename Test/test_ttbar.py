import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from hep_ml import reweight as rw
from sklearn.preprocessing import StandardScaler, QuantileTransformer, Normalizer

def getZ(file):

	arr = np.load(file, allow_pickle=True)
	Y   = arr['Z']
	Y   = np.asarray(Y).astype('float32')
	return Y.reshape(-1, Y.shape[1])
	
Z_ttbar = getZ('/media/mbelfkir/diskD/Array/user.mobelfki.00001.e3569_s3126_r12406.SCells.V04_OUTPUT/Train/Array_0.npz')
for i in range(5):
	Z_ttbar = np.append(Z_ttbar, getZ('/media/mbelfkir/diskD/Array/user.mobelfki.00001.e3569_s3126_r12406.SCells.V04_OUTPUT/Train/Array_%i.npz'%(i)),  axis=0)
	
	
