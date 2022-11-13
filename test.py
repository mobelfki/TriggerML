#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
from DataProcessing.DataProcessing import SuperCells
from Models.models import *
from tensorflow.keras.utils import plot_model
from Helps.helps import *
from Fit.fit import *
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import json
import os
seed = 215
import numpy as np
np.random.seed(seed)
from numpy.random import seed as random_seed
random_seed(seed)
tf.random.set_seed(seed)
from pickle import load

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def getArgs():
	
	args = ArgumentParser(description="Argumetns")
	args.add_argument('-d', '--dir', action='store', help='json config file')
	args.add_argument('-e', '--epoch', action='store', help='model epoch')
	args.add_argument('-s', '--test_on', action='store', default=['361021'], help='data to test on')
	args.add_argument('-m', '--merge', action='store', default=['410471'], help='data to merge train val and test')
	args.add_argument('-p', '--from_pkl', action='store', default=False, help='use model from pickle')
	return args

def make_model(args):

	supercells = SuperCellsModel(args.model_name, args.layers_name, args.layers_shape, args.n_outputs, args.Conv_layers, args.Pool_layers, args.Dense_layers)
	
	return supercells.get_model()
	
def getRatio(args):
	
	dics = {"cell_met": "cell_et", "scell_xy": ["scell_ex", "scell_ey"]}
	
	try: 
		return dics[args.learn_ratio_to]	
	except:
		print('target not found')
		exit()
	
	

def save_result(args, model, test_df):

	try:
		os.makedirs(args.output_dir)

		os.makedirs(args.output_dir+"/results/")
		os.makedirs(args.output_dir+"/model/")
		print(args.output_dir)
		
	except:
		print('dir ' + args.output_dir + ' already exists')
        
		try:
			os.makedirs(args.output_dir+'/results/')
			os.makedirs(args.output_dir+'/model/')

		except:
			print('dir status + model already exists')
       
	test_df.to_hdf(args.output_dir+'/results/data.'+str(args.train_on)+'.h5', key='df', mode='w')	
    		
	model_json = model.to_json()
	with open(args.output_dir+'/model/'+args.model_name+'.json', "w") as json_file:
		json_file.write(model_json)
	model.save_weights(args.output_dir+'/model/'+args.model_name+'.h5')
	
def addNNResult(args, data):

	
	data['NN_ex'] = data.predicted_x
	data['NN_ey'] = data.predicted_y
	data['NN_et'] = data.predicted_met
	data['NN_phi']= data.predicted_phi
	
	return data
        
def test(args, model, data):
		
	fit = Fit(model, data,  args)
	fit.compile()
	test_df = fit.eval('Test')
	
	if str(args.train_on) in args.merge:
		print('Start merging...')
		train_df = fit.eval('Train')
		val_df   =   fit.eval('Val')
		test_df = test_df.append(train_df, ignore_index=True)
		test_df = test_df.append(val_df, ignore_index=True)
		
	test_df = addNNResult(args, test_df)
	save_result(args, model, test_df)
	
def load_weight(args, model):
	
	if args.from_pkl:
		pmodel = load( open(args.dir+'/model/'+args.model_name+'.pkl', 'rb'))
		return pmodel
		 
	weight_file = ''
	if args.epoch == -1: 
		weight_file = args.dir+'/model/'+args.model_name+'.h5'
	
	else: 
		weight_file = args.dir+'/model_weight_epoch_'+str(args.epoch)+'.h5'	
	
	model.load_weights(weight_file)
	
	return model

def load_transformer(args, data):

	transformer = load( open(args.dir+'/model/transformer.pkl', 'rb'))
	
	data.transformer = transformer
	
	return data	
	
def main():

	args = getArgs()
	model_config = args.parse_args().dir+'model_config.json'
	args.add_argument('--config', default=model_config)
	args = read_json(args)
	
	Supercells = SuperCells(args)
	for dsid in args.dsid:
		Supercells.load(dsid)
	
	Supercells = load_transformer(args, Supercells)
	
	print(Supercells.transformer.mean_)	
	model = make_model(args)
	model = load_weight(args, model)
	print("Testing model_config: %s ..."%(model_config))	
	print("Using epoch: %i"%int(args.epoch))
	for sample in args.test_on:
		args.train_on = int(sample)
		print("Test on Sample: %i"%int((args.train_on)))
		test(args, model, Supercells)
	
	
if __name__ == '__main__':
	
	main()




            	
