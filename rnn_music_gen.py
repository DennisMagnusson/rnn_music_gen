from __future__ import division

import parse_midi
import generate_midi

import numpy as np
from sklearn import preprocessing
import h5py

from os import listdir

import math

from keras.models import Sequential
from keras.layers import Recurrent, LSTM, GRU
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers.embeddings import Embedding

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

def create_model(loss='mean_squared_error'):
	#The super awesome new and improved one
	#l = int(x.shape[1])
	model = Sequential()
	model.add(LSTM(256, return_sequences=True, input_dim=131, forget_bias_init='one', activation="tanh", dropout_U=0.4))
	model.add(Dropout(0))
	model.add(LSTM(131, return_sequences=False, forget_bias_init='one', activation="tanh"))
	model.compile(loss=loss, optimizer='rmsprop')
	
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

def to_midi(r, norm=True, max_time=0, max_tempo=0, min_tempo=0):
	l = r.tolist()
	l = l[0]
	if norm:
		l = denormalize(l, max_time=max_time, max_tempo=max_tempo, min_tempo=min_tempo)
	mid = generate_midi.generate(l)
	return mid

def predict(x, model):#With the new, badass way of doing things
	r = x#Fill with something
	for i in range(10000):
		nxt = model.predict(r)
		if nxt == [0]*131:
			return r
		r.append(model.predict(r))

	return r

songs, max_time, max_tempo, min_tempo = create_dataset()

#The cool, new shizzz
model = create_model()
max_len = 0#Maybe use if necessary
delta = 5

for s in songs:

	x = []
	y = []
	o = 0
	for u in range(1, len(s)-1, delta):#Taking some steps to reduce memory
		o += 1
		l = []
		for k in range(u):
			l.append(s[k])#TODO Length is fucked up.

		x.append(l)
		y.append(i[u+1])
		
	x = np.array(x)
	y = np.array(y)
	#x = x.reshape((x.shape[0]/131, None, 131))
	#y = y.reshape((y.shape[0]/131, 131))
	print x.shape #Should be 3d
	print y.shape
	model.train_on_batch(x, y)
	#model.fit(x, y, batch_size=128, nb_epoch=1, verbose=1)
	#x = np.array([], dtype=np.float16)
	#y = np.array([], dtype=np.float16)
print "done"



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
