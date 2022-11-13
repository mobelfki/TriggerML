import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from hep_ml import reweight as rw
from sklearn.preprocessing import StandardScaler, QuantileTransformer, Normalizer

def getY(file):

	arr = np.load(file, allow_pickle=True)
	Y   = arr['Y'][:,0:3]
	Y   = np.asarray(Y).astype('float32')
	return Y.reshape(-1, 3)

	
Y_ZH = getY('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Train/Array_0.npz')
for i in range(42):
	Y_ZH = np.append(Y_ZH, getY('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Train/Array_%i.npz'%(i)),  axis=0)

Y_Ztautau = getY('/media/mbelfkir/diskD/Array/user.mobelfki.361108.e3601_s3126_r12406.SCells.V04_OUTPUT/Train/Array_0.npz')
for i in range(40):
	Y_Ztautau = np.append(Y_Ztautau, getY('/media/mbelfkir/diskD/Array/user.mobelfki.361108.e3601_s3126_r12406.SCells.V04_OUTPUT/Train/Array_%i.npz'%(i)),  axis=0)
	
Y_ttbar = getY('/media/mbelfkir/diskD/Array/user.mobelfki.00001.e3569_s3126_r12406.SCells.V04_OUTPUT/Train/Array_0.npz')
for i in range(5):
	Y_ttbar = np.append(Y_ttbar, getY('/media/mbelfkir/diskD/Array/user.mobelfki.00001.e3569_s3126_r12406.SCells.V04_OUTPUT/Train/Array_%i.npz'%(i)),  axis=0)	
	
	


'''
Y  = Y_ZH

Yy = Y[:,1]
Yx = Y[:,2]

phi = np.arctan2(Yx, Yy).reshape(-1, 1)

quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
scaler = Normalizer()

phip = quantile_transformer.fit_transform(phi)

phis = scaler.fit_transform(phip).reshape(-1)

plt.figure()
hist_settings = {'bins': 80, 'range':[-4, 4], 'density': True, 'histtype': 'step', 'linewidth':1.2}
plt.hist(phi, **hist_settings)
plt.hist(phis, **hist_settings)
plt.legend(['Origin', 'Rotate'], loc='best')
plt.show()

E = (Yy**2 + Yx**2)**(0.5)

Yp = E*np.sin(phis)
Xp = E*np.cos(phis)


plt.figure()
hist_settings = {'bins': 100, 'range':[-500, 500], 'density': True, 'histtype': 'step', 'linewidth':1.2}
plt.hist(Yx/1000, **hist_settings)
plt.hist(Xp/1000, **hist_settings)
plt.legend(['Origin', 'Target'], loc='best')
plt.show()

exit()


plt.figure()

hist_settings = {'bins': 500, 'range':[-500, 500], 'density': True, 'histtype': 'step'}

plt.hist(Y_ZH[:,1]/1000, **hist_settings)
plt.hist(Y_Ztautau[:,1]/1000, **hist_settings)
#plt.hist(Y_ttbar[:,0]/1000, **hist_settings)
plt.legend(['ZH', 'Ztautau', 'ttbar'], loc='best')
plt.show()

	


plt.figure()

hist_settings = {'bins': 400, 'range':[0, 400], 'density': True, 'histtype': 'step'}


et  = (Y[:,0]**2 + Y[:,1]**2)**0.5
phi = np.arctan2(Y[:,0], Y[:,1])

plt.hist(et/1000, **hist_settings)

plt.show()
exit()
'''



Y = Y_ZH



#Y = np.append(Y, Y_Ztautau, axis=0)
#Y = np.append(Y, Y_ttbar, axis=0)

print(Y.shape)

#Y = Y[:, 0]

#Y = np.concatenate((Y, np.arctan2(Y[:,1], Y[:,2]).reshape(-1,1)), axis=1).reshape(-1, 4) 


Yy = Y[:,1]
Yx = Y[:,2]


#Y_f = scaler.transform(Y[:,0:3])

#Yy = Y_f[:,1]
#Yx = Y_f[:,2]

org = (Yy**2 + Yx**2)**0.5

#org = Y_f[:, 1:3]

N = org.shape

trg = np.random.uniform(org.min(), org.max(), N)

#trg_x = np.random.uniform(Yx.min(), Yx.max(), N)
#trg_y = np.random.uniform(Yy.min(), Yy.max(), N)

#trg = (trg_x**2 + trg_y**2)**0.5
#trg_phi = np.arctan2(trg_y, trg_x)

#trg = np.concatenate((trg_e.reshape(-1, 1) , trg_y.reshape(-1, 1), trg_x.reshape(-1, 1), trg_phi.reshape(-1, 1)), axis=1).reshape(-1, 4)

#print(trg)

#GBrw = rw.GBReweighter(n_estimators=80, learning_rate=0.3)
GBrw = rw.BinsReweighter(n_bins = 400, n_neighs=0.05)
#GBrw_y = rw.BinsReweighter(n_bins = 40, n_neighs=0.2)

GBrw.fit(org, trg)

w = GBrw.predict_weights(org)

#w /= np.median(w)

#w *= W
#w_y = w_y / np.sum(w_y)


hist_settings = {'bins': 150, 'range':[0, 500], 'density': True, 'histtype': 'step', 'linewidth':1.2}
plt.figure()
plt.hist(org/1000, **hist_settings)
plt.hist(trg/1000, **hist_settings)
plt.hist(org/1000, weights=w, **hist_settings)
plt.legend(['Ref', 'Target', 'Reweighted'], loc='best')
plt.xlabel('Truth MET [GeV]')
plt.ylabel('Fraction of Events')
plt.savefig('ZH_MET.pdf')
plt.show()
plt.close()


ist_settings = {'bins': 100, 'range':[-200, 200], 'density': True, 'histtype': 'step', 'linewidth':1.2}
plt.figure()
plt.hist(Yy/1000, **hist_settings)
plt.hist(Yy/1000, weights=w, **hist_settings)
plt.legend(['Before Reweighting', 'Target', 'Reweighted'], loc='best')
plt.xlabel('Truth MET [GeV]')
plt.ylabel('Fraction of Events')
plt.savefig('ZH_MET.pdf')
plt.show()
plt.close()

exit()

org = org/1000
mask_low = org < 10.0 
mask_medium = ( org > 70.) * (org < 150.)
mask_high = org > 300.0

print(mask_high)

print('====='*10)
print('Full MET ')
print('====='*10)
print(w)
print('%.4f +/- %.4f'%(w.mean(), w.std()))
print('[ %.4f , %.4f ]'%(w.min(), w.max()))
print('[ %.4f , %.4f ]'%(org.min(), org.max()))


print('====='*10)
print('Low MET ')
print('====='*10)
mask = mask_low
print(w[mask])
print('%.4f +/- %.4f'%(w[mask].mean(), w[mask].std()))
print('[ %.4f , %.4f ]'%(w[mask].min(), w[mask].max()))
print('[ %.4f , %.4f ]'%(org[mask].min(), org[mask].max()))



print('====='*10)
print('Medium MET ')
print('====='*10)
mask = mask_medium
print(w[mask])
print('%.4f +/- %.4f'%(w[mask].mean(), w[mask].std()))
print('[ %.4f , %.4f ]'%(w[mask].min(), w[mask].max()))
print('[ %.4f , %.4f ]'%(org[mask].min(), org[mask].max()))


print('====='*10)
print('High MET ')
print('====='*10)
mask = mask_high
print(w[mask])
print('%.4f +/- %.4f'%(w[mask].mean(), w[mask].std()))
print('[ %.4f , %.4f ]'%(w[mask].min(), w[mask].max()))
print('[ %.4f , %.4f ]'%(org[mask].min(), org[mask].max()))

exit()
hist_settings = {'bins': 350, 'range':[0, 600], 'density': True, 'histtype': 'step', 'linewidth':1.2}
plt.figure()


plt.hist(org/1000, weights=w, **hist_settings)
plt.hist(org/1000, **hist_settings)
plt.hist(trg/1000, **hist_settings)
plt.legend(['Weighted', 'Origin', 'Target'], loc='best')
plt.show()
plt.close()


#Y_f = scaler.transform(Y[:,1:3])

Yy = Y[:,1]
Yx = Y[:,2]

plt.figure()
hist_settings = {'bins': 100, 'range':[-500, 500], 'density': True, 'histtype': 'step', 'linewidth':1.2}
plt.hist(Yy/1000, weights=w, **hist_settings)
plt.hist(Yy/1000, **hist_settings)
#plt.hist(trg/1000, **hist_settings)
plt.legend(['Weighted', 'Origin', 'Target'], loc='best')
plt.show()


plt.figure()
hist_settings = {'bins': 100, 'range':[-500, 500], 'density': True, 'histtype': 'step', 'linewidth':1.2}
plt.hist(Yx/1000, weights=w, **hist_settings)
plt.hist(Yx/1000, **hist_settings)
#plt.hist(trg/1000, **hist_settings)
plt.legend(['Weighted', 'Origin', 'Target'], loc='best')
plt.show()

'''
exit()
phi = np.arctan2(Yy, Yx)

phi_t = np.arctan2(trg_y, trg_x)

plt.figure()
hist_settings = {'bins': 40, 'range':[-4, 4], 'density': True, 'histtype': 'step', 'linewidth':1.2}
plt.hist(phi, weights=w, **hist_settings)
plt.hist(phi, **hist_settings)
plt.hist(phi_t, **hist_settings)
plt.legend(['Weighted', 'Origin', 'Target'], loc='best')
plt.show()
'''

org = org/1000
mask_low = org < 10.0 
mask_medium = ( org > 70.) * (org < 150.)
mask_high = org > 300.0

print(mask_high)

print('====='*10)
print('Full MET ')
print('====='*10)
print(w)
print('%.4f +/- %.4f'%(w.mean(), w.std()))
print('[ %.4f , %.4f ]'%(w.min(), w.max()))
print('[ %.4f , %.4f ]'%(org.min(), org.max()))


print('====='*10)
print('Low MET ')
print('====='*10)
mask = mask_low
print(w[mask])
print('%.4f +/- %.4f'%(w[mask].mean(), w[mask].std()))
print('[ %.4f , %.4f ]'%(w[mask].min(), w[mask].max()))
print('[ %.4f , %.4f ]'%(org[mask].min(), org[mask].max()))



print('====='*10)
print('Medium MET ')
print('====='*10)
mask = mask_medium
print(w[mask])
print('%.4f +/- %.4f'%(w[mask].mean(), w[mask].std()))
print('[ %.4f , %.4f ]'%(w[mask].min(), w[mask].max()))
print('[ %.4f , %.4f ]'%(org[mask].min(), org[mask].max()))


print('====='*10)
print('High MET ')
print('====='*10)
mask = mask_high
print(w[mask])
print('%.4f +/- %.4f'%(w[mask].mean(), w[mask].std()))
print('[ %.4f , %.4f ]'%(w[mask].min(), w[mask].max()))
print('[ %.4f , %.4f ]'%(org[mask].min(), org[mask].max()))


