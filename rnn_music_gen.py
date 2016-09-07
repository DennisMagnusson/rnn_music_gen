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
from keras.optimizers import RMSprop

def max_index(r, x):#TODO Modify to decrease the probability of notes
	m = 0
	k = 0
	prev = x[len(x)-1].index(1)
	if len(x) > 2:
		prev2 = x[len(x)-2].index(1)
		if len(x) > 3:
			prev3 = x[len(x)-3].index(1)
		else:
			prev3 = -1
	else:
		prev2 = -1
		prev3 = -1

	for i in range(len(r)):
		if i == prev or i == prev2 or i == prev3: continue
		if r[i] > m:
			m = r[i]
			k = i
	
	if r[prev] > 2*m:#To prevent getting stuck on one note
		k = prev
		print "Exception"
	elif r[prev2] > 1.5*m:#To prevent getting stuck on two notes
		k = prev2
		print "hm"
	elif r[prev3] > 1.25*m:
		k = prev3
		print "y"
	
	print k	
	return k

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
			elif min_tempo > r[i][t][130]:
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

def create_model(loss='binary_crossentropy'):
	model = Sequential()

	model.add(LSTM(512,
	        dropout_W=0.4,
	        dropout_U=0.4,
	        return_sequences=True,
	        input_dim=88,
	        forget_bias_init='one',
	        activation="tanh",
	        init='normal',
	        inner_init='glorot_normal'))

	model.add(LSTM(256,
	        dropout_U=0.4,
	        return_sequences=False,
	        forget_bias_init='one',
	        activation="tanh",
	        init='normal',
	        inner_init='glorot_normal'))

	model.add(Dense(88,
	        activation="softmax",
	        init='normal'))

	optimizer = RMSprop(lr=0.001)
	model.compile(loss=loss, optimizer=optimizer)
	
	return model

def create_dataset(norm=False, size=999999):
	songs = []
	files = listdir("music/")
	for i in files:
		s = parse_midi.parse("music/"+i)
		if len(s) <= size:
			songs.append(s)
	
	return normalize(songs) if norm else songs

def filter_data(songs, size):#Reomves songs over a specific size
	i = 0
	while i < len(songs):
		if len(songs[i]) > size:
			songs.pop(i)
			i -= 1

		i += 1

	return songs

def to_midi(r, norm=False, max_time=0, max_tempo=0, min_tempo=0):
	l = r.tolist()
	l = l[0]
	if norm:
		l = denormalize(l, max_time=max_time, max_tempo=max_tempo, min_tempo=min_tempo)

	mid = generate_midi.generate(l)
	return mid

def clamp(r, x):
	r = r.tolist()[0]
	x = x.tolist()[0]
	i = max_index(r, x)
	r = [0]*88
	r[i] = 1
	
	return np.array(r)
	

def predict(x, model, length=1000, clmp=True):
	for i in range(length):
		nxt = model.predict(x)
		if clmp:
			nxt = clamp(nxt, x)

		x = np.append(x, nxt)
		x = x.reshape(1, i+2, 88)

	return x

def train(model, songs, delta=5, length=999999):
	maxlen = 0
	for s in songs:
		if len(s) > maxlen:
			maxlen = len(s)

	if maxlen > length:
		maxlen = length

	for i in range(1, maxlen-1, delta):
		x = []
		y = []
		for s in songs:
			if(len(s) <= i+1): continue
			x.append(s[0:i])
			y.append(s[i+1])
		if(len(x) == 0): return
		x = np.array(x)
		y = np.array(y)
		if ((i-1) % 10) == 0:
			print i 

		model.train_on_batch(x, y)

	print "done"
