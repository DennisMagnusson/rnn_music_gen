from __future__ import division

import parse_midi
import generate_midi

import numpy as np
#from sklearn import preprocessing
import h5py

from os import listdir

import math

from keras.models import Sequential
from keras.layers import Recurrent, LSTM, GRU
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers.advanced_activations import SReLU, ThresholdedReLU
#from keras.layers.embeddings import Embedding

"""
Okay, so the songs are empty. That's a problem. Could it be due to problems with np.reshape?
Likely, have a look into it.

Could also be in the training.
Maybe print those arrays and have a look.
"""

def to_dataset(r):
	m = 0
	for i in range(len(r)):
		if len(r[i]) > m:
			m = len(r[i])
	return np.zeros(len(r), m, 131)

def normalize(r):
	max_time = 0
	max_tempo = 0
	min_tempo = float(0)
	for i in range(len(r)):
		for t in range(len(r[i])):
			for u in range(1, 130):
				r[i][t][u] = float(r[i][t][u] / 127.0)
			if max_time <  r[i][t][0]:
				max_time = float(r[i][t][0])
			if max_tempo < r[i][t][130]:
				max_tempo= float(r[i][t][130])
			if min_tempo > r[i][t][130]:
				min_tempo= float(r[i][t][130])

	for i in range(len(r)):
		for t in range(len(r[i])):
			r[i][t][0] = float(r[i][t][0] / max_time)
			r[i][t][130] = float((r[i][t][130]-min_tempo)/(max_tempo-min_tempo))

	return r, max_time, max_tempo, min_tempo

def remove_duplicates(r):
	i = 1
	while i < len(r):
		if r[i] == r[i-1]:
			r.pop(i)
			i = i-1
		i = i+1

	return r

def denormalize(r, max_time, max_tempo, min_tempo):
	for t in range(len(r)):
		for u in range(1, 130):
			k = r[t][u]
			v = int(math.floor(k*127))
			#Make sure you do l = l[0] if there is an error here
			r[t][u] = int(0) if v < 15.0 else v
		time = r[t][0]
		r[t][0] = int(0) if time < 0.0 else int(math.floor(time*max_time))
		tempo = r[t][130]
		r[t][130] = int(0) if tempo < 0.0 else \
					int(math.floor(tempo*(max_tempo-min_tempo)+min_tempo))
	r = remove_duplicates(r)
	return r

def create_model(loss='mean_squared_error', optimizer='sgd'):
	#The super awesome new and improved one
	#l = int(x.shape[1])
	model = Sequential()
	#model.add(LSTM(256, return_sequences=True, input_dim=131, forget_bias_init='one', activation="tanh", dropout_U=0.4))
	model.add(LSTM(256, return_sequences=True, input_dim=131, forget_bias_init='one', activation="tanh", dropout_U=0.3))

	model.add(Dropout(0))
	model.add(LSTM(131, return_sequences=False, forget_bias_init='one', activation="tanh"))
	#Add a Dense layer with ReLU?
	model.add(Dense(131))#Change to threashholded ReLU or SReLU
	model.add(Activation(ThresholdedReLU(theta=0.1)))
	#model.compile(loss=loss, optimizer='rmsprop')
	model.compile(loss=loss, optimizer=optimizer)
	
	"""	
	#OLD ONE
	l = int(x.shape[1])
	model = Sequential()
	model.add(LSTM(512, return_sequences=True, input_shape=x.shape[1:], forget_bias_init='one', activation="tanh", dropout_U=0.4))
	#model.add(Dropout(0.6))#JUST A TEST
	model.add(Dropout(0))
	#model.add(LSTM(512, return_sequences=True))
	#model.add(Dropout(0.4))
	model.add(LSTM(131, return_sequences=True, forget_bias_init='one', activation="tanh"))
	model.compile(loss=loss, optimizer='rmsprop')
	"""	
	return model

def create_dataset(norm=True):
	songs = []
	files = listdir("music/")
	for i in files:
		songs.append(parse_midi.parse("music/"+i))
	if norm:
		return normalize(songs)
	else:
		return songs

def filter_data(songs, size):#Reomves songs over a specific size
	i = 0
	while i < len(songs):
		if len(songs[i]) > size:
			songs.pop(i)
			i -= 1
		i += 1

	return songs

def to_midi(r, norm=True, max_time=0, max_tempo=0, min_tempo=0):
	l = r.tolist()
	l = l[0]
	if norm:
		l = denormalize(l, max_time=max_time, max_tempo=max_tempo, min_tempo=min_tempo)
	mid = generate_midi.generate(l)
	return mid

def clamp(r):#Some weird behaviour here, there are still negative numbers
	r = r.tolist()
	for k in range(131):
		if l[k] < 10/127:
			l[k] = 0.0
	return np.array(r)
	

def predict(x, model, length=1000):#With the new, badass way of doing things
	#r = x#Fill with something
	#TODO Replace x with tempo. Fill rest with zeros. Maybe
	for i in range(length):
		x = x.reshape(1, i+1, 131)
		#nxt = clamp(model.predict(x))
		nxt = model.predict(x)
		#if nxt == [0]*131:
			#return x
		x = np.append(x, nxt)
		#r.append(model.predict(r))

	return x

def train(model, songs, delta=5):
	maxlen = 0
	for s in songs:
		if len(s) > maxlen:
			maxlen = len(s)

	for i in range(1, maxlen-1, delta):
		x = []
		y = []
		for s in songs:
			if(len(s) <= i+1): continue
			#for k in range(i+1):
				#x.append(s[k])
			x.append(s[0:i])
			y.append(s[i+1])
		if(len(x) == 0): return
		x = np.array(x)
		y = np.array(y)
		print x.shape
		model.train_on_batch(x, y)

	print "done"


songs, max_time, max_tempo, min_tempo = create_dataset()
#model = create_model()

#The cool, new shizzz

#TODO: Something seems to be wrong with the datatypes
#Okay, I got this: Since the lengths are varying dtype=list
#That's why it is 2d
#How to fix?
#	1. np.zeros (not a good idea)
#	2. train all songs at t=whatever (might work, but harder on the processing)
#	
"""
#This is the old method
for s in songs:
	#Make a np array and fill it with other arrays
	x = []
	y = []
	for u in range(1, len(s)-1, delta):#Taking some steps to reduce memory
		l = []
		for k in range(u+1):#Should it be +1 or not?
			l.append(s[k])#TODO Length is fucked up.
		#tmp = np.array(l)
		x.append(l)
		#tmp = np.array(x)
		y.append(s[u+1])
		
	#x = np.array(x, ndmin=3, dtype=np.float32)#Maybe change to FP16
	#print x
	#x = np.array(x, ndmin=3)#Maybe change to FP32
	#x = np.asfarray(x)#Maybe change to FP32
	x = np.asfortranarray(x)
	y = np.array(y, dtype=np.float16)
	#x = x.reshape((x.shape[0]/131, None, 131))
	#y = y.reshape((y.shape[0]/131, 131))
	print x.shape #Should be 3d
	print y.shape
	model.train_on_batch(x, y)
	#model.fit(x, y, batch_size=128, nb_epoch=1, verbose=1)

#Giving #2 a try.
maxlen = 0
for s in songs:
	if len(s) > maxlen:
		maxlen = len(s)

for i in range(1, maxlen-1, delta):
	x = []
	y = []
	for s in songs:
		if(len(s) < i): continue
		#for k in range(i+1):
			#x.append(s[k])
		x.append(s[0:i])
		y.append(s[i+1])
	x = np.array(x)
	y = np.array(y)
	print x.shape
	model.train_on_batch(x, y)

print "done"
"""

"""
#x = np.array(x)
#y = np.array(x)

##############################OLD STUFF, remove when done
max_len = 0
for i in range(len(songs)):
	if len(songs[i]) > max_len:
		max_len = len(songs[i])


#A test
x = np.zeros((len(songs), max_len+1, 131), dtype=float)
y = np.zeros((len(songs), max_len+1, 131), dtype=float)
for o in range(len(songs)):
	for t in range(len(songs[o])):
		for th in range(len(songs[o][t])):
			x[o, t+1, th] = float(songs[o][t][th])
			y[o, t, th] = float(songs[o][t][th])
print "x.shape", x.shape
print "y.shape", y.shape
"""
