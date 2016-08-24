import midi
from operator import itemgetter
from math import floor
#import numpy as np

"""
MIGHT BE FUCKED UP
Some notes do not end
Maybe it is possible in a midi file
Pedal has two endings in a row
Look through the anti-fix
#ACTUALLY, probably not.
"""

def nonzero(r):
	n = 0
	for i in r:
		if i != 0:
			n = n+1
	return n

def nonzero_index(r):
	for i in range(1, len(r)):
		if r[i] != 0:
			return i

#Only needed for three bytes
def to_base_256(i):
	r = []
	for k in range(2, -1, -1):
		r.insert(0, int(floor(i/256**k)))
		i = i%256**k
	return r

def expand(r):
	#TODO test
	#FIXME Something wrong with tempo. Ticks are wrong
	#Wrong in the conversion? Or just the order of things?
	#for i in range(len(r)):
	i = 0
	while i < len(r):#Infinite loop
		if nonzero(r[i]) > 2:
			time = r[i][0]
			for u in range(1, len(r[i])):
				if r[i][u] != 0:
					arr = [0]*len(r[i])
					arr[0] = time
					arr[u] = r[i][u]
					r[i][u] = 0
					r.append(arr)
					if nonzero(r[i]) == 2:
						break
		i = i+1	

	return r

def anti_fix(r):#AKA re-fuck up
	#The notes
	#FIXME The ticks are wrong Probably in expand
	for i in range(1, len(r[0])-1):#Skip time and tempo events
		a = -1
		u = 0
		while u < len(r):
			v = r[u][i]#Velocity
			if(v != 0):
				a = v
				while True:
					u = u+1
					if u >= len(r):
						print "Shouldn't happen"#Note has no end
						break
					if r[u][i] == a:
						r[u][i] = 0
					elif r[u][i] == 0:
						r[u][i] = -1
						break
			u = u+1

	#Tempo Events
	#for i in range(len(r)):
	#TODO test, likely not working
	#FIXME Infinite loop
	"""
	i = 0
	while i < len(r):
		a = 0
		tempo = r[i][130]
		if tempo != 0:
			a = tempo
			while True:
				i = i+1
				if i >= len(r):
					break
				if r[i][130] == a:
					r[i][130] = 0
				else:
					break
			i = i+1
		else:
			i = i+1
	"""
	print "anti-fix done"	
	return r	

			

def generate(r):
	print "Starting"
	#r = r.tolist()
	r = expand(anti_fix(r))
	#r.sort(key=itemgetter(0))
	
	#Cycle through and create events
	tempo = []
	melody = []
	track = []
	for i in r:
		#if last_tempo == i[130]:

		n = nonzero_index(i)
		t = i[0]
		#TODO Fix for the -1 stuff
		if n > 0 and n < 129:
			v = i[n] if i[n] != -1 else 0
			track.append(midi.NoteOnEvent(tick=t, data=[n-1, v]))
		if n == 129:
			v = 127 if i[n] != -1 else 0
			track.append(midi.ControlChangeEvent(tick=t, data=[64, v]))
		if n == 130:
			track.append(midi.SetTempoEvent(tick=t, data=to_base_256(i[n])))

	#Insert other events
	track.insert(0, midi.KeySignatureEvent())
	track.insert(0, midi.TimeSignatureEvent(data=[3, 3, 12, 8]))#Change in future
	#tempo.insert(0, midi.SmpteOffsetEvent(data=[96, 0, 3, 0, 0]))
	#tempo = midi.Track(tempo)
	#tempo.tick_relative = False
	#tempo.make_ticks_rel()
	#tempo.append(midi.EndOfTrackEvent())

	track.insert(0, midi.TextMetaEvent(tick=0, text='bdca426d104a26ac9dcb070447587523', data=[98, 100, 99, 97, 52, 50, 54, 100, 49, 48, 52, 97, 50, 54, 97, 99, 57, 100, 99, 98, 48, 55, 48, 52, 52, 55, 53, 56, 55, 53, 50, 51]))#wtf?
	track.insert(0, midi.ControlChangeEvent(data=[91, 127]))
	track.insert(0, midi.ControlChangeEvent(data=[10, 64]))
	track.insert(0, midi.ControlChangeEvent(data=[7, 100]))
	track.insert(0, midi.ProgramChangeEvent())	
	#melody = midi.Track(melody)
	track.append(midi.EndOfTrackEvent())
	track = midi.Track(track)
	#melody.tick_relative = False
	#melody.make_ticks_rel()
	mid = midi.Pattern(format=1, resolution=480, tracks=[track])
	print "Fucking neato"
	return mid
