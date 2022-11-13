import numpy as np
seed = 412
np.random.seed(seed)
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, AveragePooling2D, GlobalMaxPooling2D, BatchNormalization, LayerNormalization, LSTM, concatenate, Input, Conv2D, MaxPooling2D, Flatten, PReLU, LeakyReLU, LayerNormalization
from tensorflow.keras import initializers
from tensorflow.keras.utils import plot_model
from sklearn.utils import class_weight, shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
tf.random.set_seed(seed)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def getModel():

	input_layer_0 = Input(shape=(64, 30, 1), name='Lr0')
	input_layer_1 = Input(shape=(64, 118, 1), name='Lr1')
	input_layer_2 = Input(shape=(64, 114, 1), name='Lr2')
	input_layer_3 = Input(shape=(64, 28, 1), name='Lr3')
	
	input_layers = [input_layer_0, input_layer_1, input_layer_2, input_layer_3]
	
	#input_layers = Input(15, name='Z')
	
	#x = input_layers
	
	def conv(x):
		x = Conv2D(32, kernel_size=(5, 5))(x)
		x = MaxPooling2D((3, 3))(x)
		x = Conv2D(32, kernel_size=(5, 5))(x)
		x = MaxPooling2D((3, 3))(x)
		x = Flatten()(x)
		return x
		
	layers = []
	
	for x in input_layers:
		layers.append(conv(x))
		
	x = concatenate(layers, name='conc_test')
	
	def DNN(x):
	
		#x = Dense(1024, activation='elu', name='dense1_test')(x)
		#x = Dense(1024, activation='relu', name='dense2_test')(x)
		
		x = Dense(100, activation='selu', bias_initializer='zeros', kernel_initializer='he_normal', name='dense3_test')(x)
		x = LayerNormalization()(x)
		
		x = Dense(100, activation='selu', bias_initializer='zeros', kernel_initializer='he_normal', name='dense4_test')(x)
		x = LayerNormalization()(x)
		
		x = Dense(100, activation='selu', bias_initializer='zeros', kernel_initializer='he_normal', name='dense5_test')(x)
		x = LayerNormalization()(x)
		
		#x = Dense(128, activation='relu', name='dense6_test')(x)
		#x = LayerNormalization()(x)
		
		#x = Dense(64, activation='elu', bias_initializer='zeros', name='dense7_test')(x)
		#x = LayerNormalization()(x)
		#x = Dense(64, activation='relu', name='dense8_test')(x)
		
		#x = Dense(16, activation='elu', bias_initializer='zeros', name='dense9_test')(x)
		#x = LayerNormalization()(x)
		#x = Dense(16, activation='relu', name='dense10_test')(x)
		
		return x
	
	x = DNN(x)
	
	output = Dense(2, activation = 'linear', kernel_initializer='he_normal', bias_initializer='zeros', name='output')(x)
	
	return Model(inputs=input_layers, outputs=[output])
				

def myappend(items):

	n = len(items)
	
	X = items[0]
	for i in range(1, n):
		X = np.append(X, items[i], axis=0)
		
	return X

def getXYZ(file):

	arr = np.load(file, allow_pickle=True)
	Z   = arr['Z']
	Y   = arr['Y'][:,0]
	X0  = arr['Lr0']
	X1  = arr['Lr1']
	X2  = arr['Lr2']
	X3  = arr['Lr3']
	
	Y   = Z[:, -3:-1]
	Z   = Z.reshape(-1, Z.shape[1])
	
	Y   = Y.reshape(-1, 2) 
	
	Z   = Z[:,-21:-6]
	
	X0 = X0 / X0.max(axis= (1,2))[:, np.newaxis, np.newaxis]
	X1 = X1 / X1.max(axis= (1,2))[:, np.newaxis, np.newaxis]
	X2 = X2 / X2.max(axis= (1,2))[:, np.newaxis, np.newaxis]
	X3 = X3 / X3.max(axis= (1,2))[:, np.newaxis, np.newaxis]
	
	X0 = np.asarray(X0).astype('float32')
	X1 = np.asarray(X1).astype('float32')
	X2 = np.asarray(X2).astype('float32')
	X3 = np.asarray(X3).astype('float32')
	
	Y = np.asarray(Y).astype('float32')
	Z = np.asarray(Z).astype('float32')
	 		
	
	return X0, X1, X2, X3, Y, Z

	
X1_lr0, X1_lr1, X1_lr2, X1_lr3, Y1, Z1 = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Train/Array_0.npz')
X2_lr0, X2_lr1, X2_lr2, X2_lr3, Y2, Z2 = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Train/Array_1.npz')
X3_lr0, X3_lr1, X3_lr2, X3_lr3, Y3, Z3 = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Train/Array_2.npz')
X4_lr0, X4_lr1, X4_lr2, X4_lr3, Y4, Z4 = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Train/Array_3.npz')


NClass = 3
Class1 = Y1.shape[0]+Y2.shape[0]
Class2 = Y3.shape[0]
Class3 = Y4.shape[0]

Total = Class1+Class2+Class3

class1_weight = float( Total/(Class1*NClass)) * np.ones(Y1.shape[0]+Y2.shape[0],)
class2_weight = float( Total/(Class2*NClass)) * np.ones(Y3.shape[0],)
class3_weight = float( Total/(Class3*NClass)) * np.ones(Y4.shape[0],)


#XT_lr0, XT_lr1, XT_lr2, XT_lr3, YT, ZT = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Train/Array_4.npz')
XT_lr0, XT_lr1, XT_lr2, XT_lr3, YT, ZT = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.361108.e3601_s3126_r12406.SCells.V04_OUTPUT/Train/Array_4.npz')

X0 = myappend([X1_lr0, X2_lr0, X3_lr0, X4_lr0])
X1 = myappend([X1_lr1, X2_lr1, X3_lr1, X4_lr1])
X2 = myappend([X1_lr2, X2_lr2, X3_lr2, X4_lr2])
X3 = myappend([X1_lr3, X2_lr3, X3_lr3, X4_lr3])

Y  = myappend([Y1, Y2, Y3, Y4])
Z  = myappend([Z1, Z2, Z3, Z4])

W  = myappend([class1_weight, class2_weight, class3_weight]).reshape(-1,1)



scaler = StandardScaler().fit(Z)

Z = scaler.transform(Z)

scaler_Y = StandardScaler().fit(Y)

Y = scaler_Y.transform(Y)

model = getModel()

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, loss=['huber'])

print(model.summary())
plot_model(model, 'model.pdf', show_shapes=True)

#X0, X1, X2, X3, Y = shuffle(X0, X1, X2, X3, Y, random_state=seed)

N = 10
#print(y)

history = model.fit( x = {'Lr0': X0, 'Lr1': X1, 'Lr2': X2, 'Lr3': X3}, y=Y, epochs=N, batch_size=32, validation_split=0.33)
#history = model.fit(x=Z, y=Y, epochs=N, batch_size=512, validation_split=0.25)

plt.figure()
plt.plot(range(N), history.history['loss'])
plt.plot(range(N), history.history['val_loss'])
plt.legend(['train', 'val'])
plt.savefig('loss_test.pdf')
plt.close()

X_v = [XT_lr0, XT_lr1, XT_lr2, XT_lr3]

#X_v = ZT
Y_v = YT

#X_v = scaler.transform(X_v)
	
Y_pre = model.predict(X_v)

Y_pre = scaler_Y.inverse_transform(Y_pre)

Y_pre  = (Y_pre[:,0]**2 + Y_pre[:,1]**2)**(0.5)
Y_true = (Y_v[:,0]**2 + Y_v[:,1]**2)**(0.5)


plt.figure()
	
hist_settings = {'bins':100, 'range':[0, 400], 'histtype':'step', 'density':True}
plt.hist(Y_pre/1000, **hist_settings)
plt.hist(Y_true/1000, **hist_settings)
plt.legend(['Predicted', 'Test'])
plt.savefig('test.pdf')
#plt.show()
plt.close()



'''

	
	
	# For debuging
	model = test_model()
	
	print(model.summary())
	
	
	#exit()
	
	plot_model(model, 'model_test.pdf', show_shapes=True)
	
	
	optimizer                    = getoptimizer(args.optimizer, 1e-5)
	
	#arr1 = np.load('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Train/Array_1.npz', allow_pickle=True)
	#arr2 = np.load('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Val/Array_1.npz', allow_pickle=True)
	#arr3 = np.load('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Test/Array_1.npz', allow_pickle=True)
	
	X1, Y1, Z1 = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Train/Array_0.npz')
	X2, Y2, Z2 = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Train/Array_1.npz')
	X3, Y3, Z3 = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Train/Array_2.npz')
	X4, Y4, Z4 = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Train/Array_3.npz')
	
	
	X1V, Y1V, Z1V = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Val/Array_0.npz')
	X2V, Y2V, Z2V = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Val/Array_1.npz')
	X3V, Y3V, Z3V = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Val/Array_2.npz')
	X4V, Y4V, Z4V = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Val/Array_3.npz')
	
	
	X1T, Y1T, Z1T = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.361108.e3601_s3126_r12406.SCells.V04_OUTPUT/Train/Array_0.npz')
	X2T, Y2T, Z2T = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Test/Array_1.npz')
	X3T, Y3T, Z3T = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Test/Array_2.npz')
	X4T, Y4T, Z4T = getXYZ('/media/mbelfkir/diskD/Array/user.mobelfki.345058.e6004_e5984_s3126_r12406.SCells.V04_OUTPUT/Test/Array_3.npz')
	
	
	
	loss                         = getloss(args.loss)
	
	N = 1
	
	
	reweighter = reweight.GBReweighter(n_estimators=30, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4})
	
	
	Y1_uniform = np.random.uniform(Y1.min(), Y1.max(), Y1.shape)
	Y1V_uniform = np.random.uniform(Y1V.min(), Y1V.max(), Y1V.shape)
	Y1_gauss   = np.random.normal(Y1.mean(), Y1.std(), Y1.shape)
	
	
	model.compile(optimizer=optimizer, loss=loss)
	
	
	#m = Y1.mean()
	#s = Y1.std()
	
	#Y1 = (Y1 - m)/s
	#Y1V = (Y1V - m)/s
	
	X = np.append(X1, X2, axis=0)
	X = np.append(X, X3, axis=0)
	X = np.append(X, X4, axis=0)
	
	Y = np.append(Y1, Y2, axis=0)
	Y = np.append(Y, Y3, axis=0)
	Y = np.append(Y, Y4, axis=0)
	
	Z = np.append(Z1, Z2, axis=0)
	Z = np.append(Z, Z3, axis=0)
	Z = np.append(Z, Z4, axis=0)
	
	print(Y.shape)
	
	
	history = model.fit(X, Y, epochs=N, batch_size=128, shuffle=True, validation_data=(X1V, Y1V))
	
	plt.figure()
	
	plt.plot(range(N), history.history['loss'])
	plt.plot(range(N), history.history['val_loss'])
	plt.legend(['train', 'val'])
	plt.savefig('loss_test.pdf')
	plt.close()
	
	X_v = X2T [:]
	Y_v = Y2T [:]
	
	Y_pre = model.predict(X_v)
	
	print(Y_pre[:10])
	
	l = tf.reduce_mean(loss(Y_v, Y_pre))
	
	print(l)
	
	#print(Y1_uniform.mean())
	
	#Y_pre = Y_pre*s + m
	print(Y_pre.mean())
	
	print(Y_v.mean())
	
	print(((Y_pre - Y_v)/Y_v).mean())
	
	plt.figure()
	
	hist_settings = {'bins':100, 'range':[0, 8], 'histtype':'step', 'density':True}
	plt.hist(Y_pre, **hist_settings)
	plt.hist(Y_v, **hist_settings)
	plt.hist(Y1, **hist_settings)
	plt.hist(Y1V, **hist_settings)
	plt.legend(['Predicted', 'Test', 'Train', 'Val'])
	plt.savefig('test.pdf')
	#plt.show()
	plt.close()
	
	exit()
	
	
	data = tf.data.Dataset.from_tensor_slices((X, Y))
	data = data.shuffle(buffer_size=16, seed=seed).batch(16)

	all_loss = []
	for epoch in range(N):
		epoch_loss = []
		i = 0	
		for step, (x, y) in enumerate(data):
			with tf.GradientTape() as tape:
				train_y = model(x, training=True) # compute predicted y
				train_loss = loss(y, train_y)
			grads = tape.gradient(train_loss, model.trainable_variables)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			epoch_loss.append(train_loss)
			i+=1
		all_loss.append(tf.reduce_mean(epoch_loss))		
		print("Epoch %i loss = %.4f"%(epoch+1, tf.reduce_mean(epoch_loss)))	
	
	
	plt.figure()
	
	plt.plot(range(N), all_loss)
	plt.legend(['train'])
	plt.savefig('loss_test_static.pdf')
	plt.close()
	
	
	X_v = X [25:35]
	Y_v = Y [25:35]
	
	Y_pre = model.predict(X_v)
	
	#Y_pre = Y_pre*y_s + y_m
	print(Y_pre.std())
	
	print(Y_v.std())
	
	print(((Y_pre - Y_v)/Y_v).mean())
	
	print(Y_pre)
	print(Y_v)
	
	#exit()
	
'''

'''


def train_test_model(args, model, data): 

	dsid                         = args.train_on	
	batch_size                   = args.batch_size
	loss                         = getloss(args.loss)
	train_metric, val_metric     = getmetric(args.metric)
	optimizer                    = getoptimizer(args.optimizer, args.learning_rate)
	data_generator               = data.tf_data_generator
	processfunction              = data.getprocessfunction(args.processFunction)
	
	model, train_df = train_on_loop(model, dsid, batch_size, loss, train_metric, val_metric, optimizer, args.epochs, data_generator, processfunction, args.debug, args)
	
	test_df = test_on_loop(model, dsid, batch_size, data_generator, processfunction, args.debug)
			
	save_result(args, model, train_df, test_df)	
					



		
	
	
'''


	









