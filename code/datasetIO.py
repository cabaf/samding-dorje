'''
   _____________________________________________________________________
   datasetIO.py
   ====================
   Includes several functions related with accesing dataset info.

   Supported datasets:
     - Hollywood2
     - HMDB51
     - Olympic Sports

   Copyright @ Fabian Caba H.
   _____________________________________________________________________
'''
import os.path
import sys
import numpy as np
from scipy.io import loadmat
import re

def get_all_video_ids(dataset_info_file, dataset_path, **args):
    """
    ____________________________________________________________________
       get_all_video_ids:
         Search and return all video ids on a dataset. Useful for 
         extracting features.
         args:
           dataset_info_file: full path of matfile containing info about
           the train/set splits on the dataset. It must have the
           following struct:
             data(set_id).train.(className)
             data(set_id).test.(className)
           dataset_path: full path for the dataset root directory.
         return:
           full_path_video_list: returns ids for the entire dataset in 
             a string list.
    ____________________________________________________________________
    """
    ############################################################################
    opts = {"verbose":False}
    if len(args)==1 and isinstance(args[0], dict):
        opts.update(args[1])
    if not os.path.isfile(dataset_info_file):
        print "Error: Dataset file info not found."
        sys.exit(0)

    dataset_info = loadmat(dataset_info_file)["data"][0]
    n_sets = len(dataset_info)
    if opts["verbose"]:
        print "Getting al video ids..."
    video_list = []
    for set_id in range(n_sets):
        this_train_set = loadmat(dataset_info_file)["data"][0][set_id]\
                                                   ["train"][0][0]
        this_test_set = loadmat(dataset_info_file)["data"][0][set_id]\
                                                  ["test"][0][0]
        this_video_list = np.empty((1, 1))
        for action_id in this_train_set.dtype.names:
            this_video_list = np.vstack((this_video_list, 
                                         this_train_set[action_id]))
            this_video_list = np.vstack((this_video_list, 
                                         this_test_set[action_id]))
        this_video_list = np.delete(this_video_list, 0, 0)
        this_video_list = np.unique(this_video_list)
        video_list += list(this_video_list)
    video_list = np.unique(np.array(video_list))
    full_path_video_list = []
    for vidx in video_list:
        full_path_video_list += get_video_full_path(vidx, dataset_path)
    ############################################################################
    return full_path_video_list
    ############################################################################

def get_video_full_path(video_id, dataset_path, ext="avi"):
    video_id = re.escape(video_id)
    command = "find {0} -name '*{1}.{2}'".format(dataset_path, video_id, ext)
    file_path = os.popen(command).read()[:-1]
    # Sanity check
    file_path = file_path.rsplit("\n")
    for this_found in file_path:
        if not os.path.isfile(this_found):
            print "Error: File not found {}".format(this_found)
            file_path = ''
    return file_path
