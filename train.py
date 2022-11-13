#!/usr/bin/env python
import numpy as np
seed = 412
np.random.seed(seed)
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from DataProcessing.DataProcessing import SuperCells
from Models.models import *
from tensorflow.keras.utils import plot_model
from Helps.helps import *
from Fit.fit import *
from argparse import ArgumentParser
import json
import os
from pickle import dump
#from numpy.random import seed as random_seed
#random_seed(seed)
tf.random.set_seed(seed)
import matplotlib.pyplot as plt 
from hep_ml import reweight

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def getArgs():
	
	args = ArgumentParser(description="Argumetns")
	args.add_argument('-c', '--config', action='store', help='json config file')
	return read_json(args)

def make_model(args):

	supercells = SuperCellsModel(args.model_name, args.layers_name, args.layers_shape, args.n_outputs, args.Conv_layers, args.Pool_layers, args.Dense_layers)

	return supercells.get_model()
	


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
def getXYZ(file):

	arr1 = np.load(file, allow_pickle=True)
	X1 = arr1['Lr2']
	Z1 = np.array(arr1['Z'], np.float64)
	Y1 = arr1['Y'][:,0]/Z1[:,-19]
	Y1 = Y1.reshape(-1, 1)
	X1 = X1 / X1.max(axis= (1,2))[:, np.newaxis, np.newaxis]
	X1 = X1.reshape(-1, X1.shape[1], X1.shape[2])
	
	#Y1 = Z1[:, -7]/Z1[:,-19]
	
	Z = np.concatenate((Z1[:, -16].reshape(-1, 1), Z1[:, -13].reshape(-1, 1), Z1[:, -10].reshape(-1, 1), Z1[:,-19].reshape(-1, 1)), axis=-1)
	
	#Y1 *= 4.0
	
	return X1, Y1, Z
		       	 
	
def plot_loss(args, data):
	
	plt.figure()
	plt.plot(data.epoch+1, data.train_loss) 
	plt.plot(data.epoch+1, data.val_loss)
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['Training loss', 'Validation loss'], )
	plt.savefig('%s/status/loss.pdf'%(args.output_dir))
	plt.close()	
	

def save_result(args, model, train_df, test_df, transformer, reweighter):

	try:
		os.makedirs(args.output_dir)
		os.makedirs(args.output_dir+"/status/")
		os.makedirs(args.output_dir+"/results/")
		os.makedirs(args.output_dir+"/model/")
		print(args.output_dir)
		
	except:
		print('dir ' + args.output_dir + ' already exists')
        
		try:
			os.makedirs(args.output_dir+'/status/')
			os.makedirs(args.output_dir+'/results/')
			os.makedirs(args.output_dir+'/model/')

		except:
			print('dir status + model already exists')
       
	train_df.to_hdf(args.output_dir+'/status/data.h5', key='df', mode='w')
	test_df.to_hdf(args.output_dir+'/results/data.'+str(args.train_on)+'.h5', key='df', mode='w')
	
	plot_loss(args, train_df)
    		
	model_json = model.to_json()
	with open(args.output_dir+'/model/'+args.model_name+'.json', "w") as json_file:
		json_file.write(model_json)
	model.save_weights(args.output_dir+'/model/'+args.model_name+'.h5')
	dump(transformer, open(args.output_dir+'/model/transformer.pkl', 'wb'))
	dump(reweighter, open(args.output_dir+'/model/reweighter.pkl', 'wb'))
	dump(model, open(args.output_dir+'/model/'+args.model_name+'.pkl', 'wb'))
	
def fit(args, model, data):
		
	fit = Fit(model, data, args)
	fit.compile()
	fit.execute()
	train_df, test_df, model = fit.share()
	save_result(args, model, train_df, test_df, data.transformer, data.reweighter)
	
def main():

	args = getArgs()
	Supercells = SuperCells(args)
	for dsid in args.dsid:
		Supercells.load(dsid)
	for dsid, dsids in args.to_merge.items():
			Supercells.merge(int(dsid), dsids)
	
	print(Supercells.DSIDfiles[999999])	
	Supercells.fitTransformerAndReweighter()
	
	print(Supercells.transformer.mean_)
	model = make_model(args)
	print(model.summary())
	
	plot_model(model, args.output_dir+'/model.pdf', show_shapes=True)
	#train_test_model(args, model, Supercells)   # fit model in static mode (for gradient debuging)  moved to Test/hard_test.py
	fit(args, model, Supercells) # fit model in dynamic mode (x2 faster) 
	
if __name__ == '__main__':
	
	main()




            	
