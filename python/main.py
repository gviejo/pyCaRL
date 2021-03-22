import numpy as np
import scipy
import glob
import sys,os
import h5py as hd
from time import time
from functions import *
from pylab import *
from scipy.ndimage import gaussian_filter1d
import cv2
from numba import jit

@jit(nopython=True)
def softmax(a, beta):
	return (np.exp(beta*a.T)/(np.sum(np.exp(beta*a), 1))).T

@jit(nopython=True)
def computeDist(cpos):
	return np.sqrt(np.sum(np.power(cpos[:,:,None] - cpos[:,:,None].T, 2), 1)).mean(1)

@jit(nopython=True)
def getnextpos(cpos, n, a, pa):
	tmp = cpos.reshape(n,2,1)
	tmp2 = np.repeat(tmp, a).reshape((n,2,a))
	tmp3 = tmp2 + pa
	return tmp3

@jit(nopython=True)
def selectAction(p, n, na):
	act = np.zeros(n, dtype = np.int64)
	for k in range(n):
		act[k] = np.searchsorted(np.cumsum(p[k]), np.random.random(), side="right")
	return act

@jit(nopython=True)
def getdist(cpos, n, m):
	tmp = cpos.reshape(n,2,1)
	dist = np.sqrt(np.sum(np.power(tmp - tmp.T, 2), 1))
	dist = np.exp(-dist * 0.05)*m
	np.fill_diagonal(dist, 0)
	return dist

@jit(nopython=True)
def getr(fl, ds):
	return np.tanh(fl*0.1) - np.tanh(ds)

# @jit(nopython=True)
def updatev(v, cpos, alpha, r, gamma, npos, n):
	id1 = (np.arange(n),cpos[:,0],cpos[:,1])
	id2 = (np.arange(n),npos[0],npos[1])
	v[np.arange(n),cpos[:,0],cpos[:,1]] = v[np.arange(n),cpos[:,0],cpos[:,1]] + alpha * (r + gamma * v[np.arange(n),npos[0],npos[1]] - v[np.arange(n),cpos[:,0],cpos[:,1]])
	return v



def play(data, gain = 1, magnification = 2, looping = True, fr = 30):
	maxmov = np.nanmax(data)
	end = False
	T = data.shape[0]
	while looping:
		for i in range(T):
			frame = data[i]
			if magnification != 1:
				frame = cv2.resize(frame, None, fx = magnification, fy = magnification, interpolation = cv2.INTER_LINEAR)
			cv2.imshow('frame', frame * gain / maxmov)
			if cv2.waitKey(int(1. / fr * 1000)) & 0xFF == ord('q'):
				looping = False
				end = True
				break
		if end:
			break
	cv2.waitKey(100)
	cv2.destroyAllWindows()
	for i in range(10):
		cv2.waitKey(100)
	return


path = '../data'
# files = glob.glob(os.path.join(path, '*.avi'))
# # read the data
# video_info, videos, dims = get_video_info(files)
# hdf_mov       = get_hdf_file(videos, video_info, dims)
# duration    = video_info['duration'].sum() 

file = hd.File(path + '/video_V4.hdf5', 'r+')
dims = file.attrs['dims']
data = file['movie']
data = data[()].reshape(len(data),dims[0],dims[1])

data = data[:,130:400,230:450]
T = data.shape[0]
dims = data.shape[1:]

# a = data[:,84,109]

faslow = gaussian_filter1d(data, sigma = 60)
fafast = gaussian_filter1d(data, sigma = 5)

fa = fafast - faslow

#play(fa, 1, 2)



# @jit(nopython=True)
# def cutnextpos(npos, d1, d2):
# 	idx = np.where(npos<0)
# 	npos[idx] = 0
# 	# npos[:,0,:][npos[:,0,:]>=d1] = d1-1
# 	# npos[:,1,:][npos[:,1,:]>=d2] = d2-1
# 	return npos


# Initiate a pool of agents with random position
n = 20
cpos = np.vstack((np.random.randint(fa.shape[1], size = n),
				np.random.randint(fa.shape[2], size = n))).T

gamma = 0.1
alpha = 0.1
beta = 0.1

v = np.zeros(([n]+list(dims)))

possible_actions = np.array([[0,0],[-1,0],[0,-1],[+1,0],[0,+1]])
possible_actions = np.tile(np.atleast_3d(possible_actions).T, (n, 1, 1))
na = possible_actions.shape[-1]

nextpos2v = np.tile(np.atleast_2d(np.arange(0,n)).T, possible_actions.shape[2])
p2nextact = np.tile(np.atleast_2d(np.arange(0,possible_actions.shape[2])),(n,1))
maxfa = fa.max((1,2))


# pos = [xyag]
for i in range(10000):
	t1 = time.time()
	# print(i)
	for j in range(T):
		
		if j%250 == 0:
			v[np.abs(v)<1e-2] = 0.0
			v = v - 0.05 * v

		nextpos = getnextpos(cpos, n, na, possible_actions)

		nextpos[nextpos<0] = 0
		nextpos[:,0,:][nextpos[:,0,:]>=dims[0]] = dims[0]-1
		nextpos[:,1,:][nextpos[:,1,:]>=dims[1]] = dims[1]-1
		

		p = softmax(v[nextpos2v,nextpos[:,0],nextpos[:,1]], beta)

		act = selectAction(p, n, na)

		npos = nextpos[np.arange(n),:,act].T

		# dist = getdist(cpos, n, maxfa[j])

		fl = fa[j][cpos[:,0],cpos[:,1]]

		# r = getr(fl, dist.max(0))
		r = getr(fl, np.zeros(n))

		v = updatev(v, cpos, alpha, r, gamma, npos, n)

		cpos = npos.T

	# v[v<0] = 0.0

	
	print(i, time.time() - t1)



figure()
subplot(121)
imshow(fa.max(0))
for i in range(len(v)):
	idx = np.unravel_index(np.argmax(v[i]), v[i].shape)
	plot(idx[1], idx[0], 'o')
subplot(122)
imshow(v.mean(0))
for i in range(len(v)):
	idx = np.unravel_index(np.argmax(v[i]), v[i].shape)
	plot(idx[1], idx[0], 'o')


figure()
for i in range(n):
	subplot(5,6,i+1)
	imshow(v[i])

show()