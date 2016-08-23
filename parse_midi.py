import midi
from operator import itemgetter, attrgetter
import numpy as np

"""
ControlChangeEvent data=[64, 127 or 0] is pedal

Matrix structure:
0 = time
1-128 notes pressed (velocity)
129 pedale pressed  (bool)
130 tempo from settempoevent

#Should I put the smpteoffsets, time sigs and key sigs there?
########
When generating file can everything be in one single thing?
"""

def to_int(arr):
	a = 0
	for i in range(len(arr)-1, -1, -1):
		a += arr[i]*256**i
	return a

def get_data(pattern, d, event):
	data = []
	for i in range(len(pattern[d])):
		if isinstance(pattern[d][i], event):
			data.append(to_int(pattern[d][i].data))

	return data
"""
def remove_duplicates(r):
	for i in range(1, len(r)):
		if r[i] == r[i-1]:
			r.pop[i]
			print "test"
	return r
"""

def compress(r):
	#for i in range(len(r)):
	i = 1
	while i < len(r):
		if r[i][0] == r[i-1][0]:#If time is equal
			#for u in range(1, len(r[1])):
			for u in range(1, 131):#This is prop less confusing
				if r[i][u] != 0 and r[i-1][u] == 0:
					r[i-1][u] = r[i][u]
			r.pop(i)
			i = i-1
		i = i+1

	return r

def fix(r):#AKA fill
	#Fill empty zeroes in the tempo things
	for i in range(1, len(r[1])-1):#Skip time and tempo events
		a = 0
		for u in range(len(r)):
			v = r[u][i]
			if v == 0 and a == 0:
				continue
			if v == -1:
				r[u][i] = 0
				a = 0
			elif v == 0:
				r[u][i] = a
			else:
				a = v
				r[u][i]
	"""
	Recently removed. Removes the fix for tempoevents
	tempo = 0
	for i in range(len(r)):
		if r[i][130] != 0:
			tempo = r[i][130]
		else:
			r[i][130] = tempo
"""
	return r

def parse(filename):
	p = midi.read_midifile(filename)
	#p.make_ticks_abs()#Holy fuck, that's awesome
	#TODO Test this shit
	#Only works with midi format 0... for now
	"""
	mid = p
	i = 1
	while i < len(mid):
		for u in mid[i]:
		mid[0].append(u)
	#mid[0].append(mid[i])
		mid.pop(i)
	#####FIXME TODO TODO FIXME Sort
	#track = mid[0]
	#track.sort()
	#mid[0] = track
	#Okay, def not working.
	#mid[0].sort()
	mid[0] = midi.Track(events=(sorted(mid[0], key=attrgetter('tick'))), tick_relative=False)
	p = mid
	p.make_ticks_rel()
	"""
	####Should work. Fix other errors TODO	
	r = []
	"""
	for i in range(len(p[0])):
		if type(p[0][i]) == midi.SetTempoEvent:
			frame = [0]*131
			frame[0] = p[0][i].tick
			frame[130] = to_int(p[0][i].data)
			r.append(frame)
	"""
	for u in range(len(p)):
		for i in range(len(p[u])):
			if type(p[u][i]) == midi.NoteOnEvent:
				frame = [0]*131
				frame[0] = p[u][i].tick
				#Velocity
				v = p[u][i].data[1]
				frame[p[u][i].data[0]+1] = v if v != 0 else -1
				r.append(frame)

			elif type(p[u][i]) == midi.ControlChangeEvent: #The Pedal
				if p[u][i].data[0] != 64:
					continue
				frame = [0]*131
				frame[0] = p[u][i].tick
				v = p[u][i].data[1]
				frame[129] = 1 if v != 0 else -1
				r.append(frame)
			elif type(p[0][i]) == midi.SetTempoEvent:
				frame = [0]*131
				frame[0] = p[0][i].tick
				frame[130] = to_int(p[0][i].data)
				r.append(frame)
	
	#r.sort(key=itemgetter(0))
	#Maybe remove compress?
	#r = compress(fix(r))#Fix, compress and remove duplicates
	r = fix(r)
	#r.sort(key=itemgetter(0))#TODO this is just a test... and it seems to work
	return r
