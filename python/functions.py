import numpy as np
import cv2
import itertools
#from . import sima_functions as sima 
import warnings
import pandas as pd
import re
import av
from tqdm import tqdm
import os
import sys
import h5py as hd
from IPython.core.debugger import Pdb
from copy import copy




def get_video_info(files):
    """ In order to get the name, duration, start and end of each video
    
    Parameters:
    -files : the pathe where ther is all the video (.avi)
    
    Returns:
    -videos : dictionnary of the videos from the miniscopes
    -video_info : DataFrame of informations about the video
    -dimension (h,w) of each frame """

    video_info  = pd.DataFrame(index = np.arange(len(files)), columns = ['file_name', 'start', 'end', 'duration'])
    videos      = dict.fromkeys(files) # dictionnary creation
    for f in files:
        num                                 = int(re.findall(r'\d+', f)[-1])
        video_info.loc[num,'file_name']     = f
        video                               = av.open(f)
        stream                              = next(s for s in video.streams if s.type == 'video') 
        video_info.loc[num, 'duration']     = stream.duration
        videos[f]                           = video

    video_info['start']     = video_info['duration'].cumsum()-video_info['duration']
    video_info['end']       = video_info['duration'].cumsum()
    video_info              = video_info.set_index(['file_name'], append = True)

    return video_info, videos, (stream.format.height, stream.format.width)



def get_hdf_file(videos, video_info, dims, **kwargs):
    """
    In order to convert the video into a HDF5 file.
    Parameters : 
    -videos : dictionnary of the videos from the miniscopes
    -video_info : DataFrame of informations about the video
    -dims : dimension (h,w) of each frame
    
    Returns :
    -file : HDF5 file"""
    hdf_mov     = os.path.split(video_info.index.get_level_values(1)[0])[0] + '/' + 'video_V4.hdf5'
    file        = hd.File(hdf_mov, "w")
    movie       = file.create_dataset('movie', shape = (video_info['duration'].sum(), np.prod(dims)), dtype = np.float32, chunks=True)

    for v in tqdm(videos.keys()):
        offset  = int(video_info['start'].xs(v, level=1))
        stream  = next(s for s in videos[v].streams if s.type == 'video')        
        tmp     = np.zeros((video_info['duration'].xs(v, level=1).values[0], np.prod(dims)), dtype=np.float32)
        for i, packet in enumerate(videos[v].demux(stream)):
            frame           = packet.decode()[0].to_ndarray(format = 'bgr24')[:,:,0].astype(np.float32)
            tmp[i]          = frame.reshape(np.prod(dims))
            if i+1 == stream.duration : break                        
            
        movie[offset:offset+len(tmp),:] = tmp[:]
        del tmp
    del movie 

    file.attrs['folder'] = os.path.split(video_info.index.get_level_values(1)[0])[0]
    file.attrs['filename'] = hdf_mov
    file.attrs['dims'] = dims
    return file
