import numpy as np
from time import time
import scipy
import glob
import yaml
import sys,os
import h5py as hd
from time import time
from functions import *
from pylab import *
from scipy.ndimage import gaussian_filter1d
import cv2

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


path = '/home/guillaume/CaRL/data'
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

dims = data.shape[1:]

# a = data[:,84,109]

faslow = gaussian_filter1d(data, sigma = 60)
fafast = gaussian_filter1d(data, sigma = 5)

fa = fafast - faslow

play(fa, 1, 2)


# Initiate a pool of agents with random position

xyagents = np.random.randint()



