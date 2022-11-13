import tensorflow as tf
from tensorflow import keras
from DataProcessing.DataProcessing import SuperCells
from Models.models import SuperCellsModel
from tensorflow.keras.utils import plot_model
from Helps.helps import train_on_loop
import numpy as np

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
print(tf.__version__)
SuperCells = SuperCells()
SuperCells.load(361108)
SuperCells.load(1)

def func(x, batch_size): 
	dic = {}
	for item in SuperCells.InputsVars:
		if item in ['Z', 'Y']: continue
		dic.update({item: x[item]})
	data = tf.data.Dataset.from_tensor_slices((dic, x['Y'])).shuffle(buffer_size=1024).batch(batch_size)	
	return data

dataset = SuperCells.tf_data_generator(361108, 'Train')

#num = 0 
'''
for d in dataset:
	
	y, x = func(d, 1)
	print(len(x['Lr0']))
	
	data = tf.data.Dataset.from_tensor_slices((x,y))
	
	data = data.shuffle(1).batch(3)
	
	for step, (x, y) in enumerate(data):
		
		print(step) 
		print(x['Lr0'].shape)
#
#exit()
'''

LayersNames = ['Lr0', 'Lr1', 'Lr2', 'Lr3', 'Lr4', 'Lr5', 'Lr6_R0', 'Lr6_R1', 'Lr7_R0', 'Lr7_R1', 'Lr8_R0', 'Lr8_R1', 'Lr21', 'Lr22', 'Lr23']
LayersShapes = {'Lr0': (64, 30, 1,),
	 'Lr1': (64, 118, 1,),
	 'Lr2': (64, 114, 1,),
	 'Lr3': (64, 28, 1,),
	 'Lr4':(64, 6, 1,),
	 'Lr5':(64, 84, 1,),
	 'Lr6_R0':(64, 88, 1,),
	 'Lr6_R1': (32, 8, 1,),
	 'Lr7_R0':(64, 20, 1,),
	 'Lr7_R1':(32, 8, 1,),
	 'Lr8_R0':(64, 20, 1,),
	 'Lr8_R1':(32, 8, 1,),
	 'Lr21':(16, 24, 1,),
	 'Lr22':(16, 16, 1,),
	 'Lr23':(16, 8, 1,)}
	 
Noutputs = 2
Conv = {'Lr0': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ], 
		
	'Lr1': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ], 
		
	'Lr2': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ], 
		
	'Lr3': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ], 
		 
	'Lr4': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ], 
		
	'Lr5': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ], 
		
	'Lr6_R0': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ], 
		
	'Lr6_R1': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ], 
		
	'Lr7_R0': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ], 
		
	'Lr7_R1': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ], 
		
	'Lr8_R0': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ], 
		
	'Lr8_R1': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ], 
		
	'Lr21': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ], 
		
	'Lr22': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ], 
		
	'Lr23': [{'neurons': 64, 'kernel': (3,3), 'activation': 'relu', 'initializer': 'lecun_normal'} ] 
	}
	
Pool = {'Lr0': [{'kernel': (2,2)} ],
		
	'Lr1': [{'kernel': (2,2)} ],
		 
	'Lr2': [{'kernel': (2,2)} ],
		 
	'Lr3': [{'kernel': (2,2)} ],
		
	'Lr4': [{'kernel': (2,1)} ],
		
	'Lr5': [{'kernel': (2,2)} ],
		
	'Lr6_R0': [{'kernel': (2,2)} ],
		
	'Lr6_R1': [{'kernel': (2,1)} ],
		
	'Lr7_R0': [{'kernel': (2,2)} ],
		
	'Lr7_R1': [{'kernel': (2,1)} ],
		
	'Lr8_R0': [{'kernel': (2,2)} ],
		
	'Lr8_R1': [{'kernel': (2,1)} ],
		
	'Lr21': [{'kernel': (2,2)} ],
		
	'Lr22': [{'kernel': (2,2)} ],
		
	'Lr23': [{'kernel': (2,1)} ]
		 }
Denses = [{'type': 'Dense', 'neurons': 64, 'activation': 'selu', 'initializer': 'lecun_normal'}, 
		{'type': 'Dropout', 'rate': 0.1}
		]

model = SuperCellsModel("test", LayersNames, LayersShapes, Noutputs, Conv, Pool, Denses).get_model()
#model.compile(loss='RMS', optimizer='adam')

print(model.summary())

plot_model(model, 'model.png', show_shapes=True)

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.MeanSquaredError()
metrict = keras.metrics.MeanSquaredError()
metricv = keras.metrics.MeanSquaredError()

train_on_loop(model, 361108, 1, loss_fn, metrict, metricv, optimizer, 2, SuperCells.tf_data_generator, SuperCells.processSuperCells)
            	
