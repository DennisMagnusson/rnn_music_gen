from __future__ import division

import parse_midi
import generate_midi

import numpy as np
from sklearn import preprocessing
import h5py

from os import listdir

import math

from keras.models import Sequential
from keras.layers import Recurrent, LSTM
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers.embeddings import Embedding

"""
time is not always increasing

gets a fuckton of tiny ones. Remove those.
"""


def to_dataset(r):
	m = 0
	for i in range(len(r)):
		if len(r[i]) > m:
			m = len(r[i])
	return np.zeros(len(r), m, 131)

def to_rel_timesteps(r):#Temp. workaround. Change the midi to rel timesteps in future
	for i in range(1, len(r)):
		r[i][0] = float(r[i][0]-r[i-1][0]) #Returns negative values. wtf?
	return r

def to_abs_timesteps(r):#Maybe not a temp workaround
	t = 0
	for i in range(len(r)-1):
		t = t+r[i][0]
		r[i][0] = int(t)
	return r

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
			if r[i][t][0] > max_time:
				print "max_time", max_time, "r[i][t][0]", r[i][t][0]
			r[i][t][0] = float(r[i][t][0] / max_time)
			r[i][t][130] = float((r[i][t][130]-min_tempo)/(max_tempo-min_tempo))
			#print r[i][t][130]

	return r, max_time, max_tempo, min_tempo

def remove_duplicates(r):
	#for i in range(1, len(r)):
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
	model = Sequential()
	model.add(LSTM(256, return_sequences=True, input_shape=x.shape[1:]))
	#model.add(Dropout(0.6))
	model.add(Dropout(0))#Just a test TODO
	model.add(LSTM(131, return_sequences=True))
	model.compile(loss=loss, optimizer='rmsprop')#Works
	return model

def create_dataset(norm=True):
	songs = []
	files = listdir("music/")
	for i in files:
		songs.append(parse_midi.parse("music/"+i))
		#songs.append(to_rel_timesteps(parse_midi.parse("music/"+i)))
	if norm:
		return normalize(songs)
	else:
		return songs

def generate_x(x):
	x = x[0]
	x = x.reshape(1, 4381, 131)
	return x

def to_midi(r, norm=True, max_time=0, max_tempo=0, min_tempo=0):
	l = r.tolist()
	l = l[0]
	if norm:
		l = denormalize(l, max_time, max_tempo, min_tempo)
	l = to_abs_timesteps(l)
	for i in range(len(l)):
		for k in range(len(l[i])):
			l[i][k] = int(l[i][k])
	mid = generate_midi.generate(l)
	return mid

songs, max_time, max_tempo, min_tempo = create_dataset()
#songs = create_dataset()

#songs = create_dataset(norm=True)
max_len = 0
for i in range(len(songs)):
	if len(songs[i]) > max_len:
		max_len = len(songs[i])
"""
the old one, where x is an empty arr of zeros.
x = np.zeros((len(songs), max_len, 131), dtype=float)
y = np.zeros((len(songs), max_len, 131), dtype=float)
for o in range(len(songs)):
	for t in range(len(songs[o])):
		for th in range(len(songs[o][t])):
			#y[o, t, th] = songs[o][t][th]
			y[o, t, th] = float(songs[o][t][th])#TODO Normalize

"""

#A test
x = np.zeros((len(songs), max_len+1, 131), dtype=float)
y = np.zeros((len(songs), max_len+1, 131), dtype=float)
for o in range(len(songs)):
	for t in range(len(songs[o])):
		for th in range(len(songs[o][t])):
			#x[o, t, th] = float(songs[o][t][th])
			#y[o, t+1, th] = float(songs[o][t][th])#TODO Normalize... or not
			x[o, t+1, th] = float(songs[o][t][th])#Oh for fucks sake which is it?TODO
			y[o, t, th] = float(songs[o][t][th])#TODO Normalize... or not


print x.shape


#model.save_weights("model", overwrite=True)
