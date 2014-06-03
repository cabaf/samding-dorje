'''
   _____________________________________________________________________
   featureExtraction.py
   ====================
   Includes several functions related with the feature extraction stage.

   Copyright @ Fabian Caba H.
   _____________________________________________________________________
'''

import os
import os.path
import subprocess
from multiprocessing import Pool
import itertools
import sys

def extract_features(video_files, output_path, idt_bin, args):
    """
    ____________________________________________________________________
       extract_features:
         Calls binary for computing improved trajectories on a 
         video list.
         args:
           video_files: List of full path of video files.
           output_path: Where will be located the output trajectories.
           idt_bin: Improved dense trajectories binary full path.
           args(optional): It should contain the following dict 
             dictionary key/value:
             "trajectory_length": Length of tracking trajectries.
               (default: L=15)
             "sampling_stride": the stride for dense sampling feature 
               points. (default: W=5 pixels)
             "neighborhood_size": The neighborhood size for computing 
               the descriptor. (default: N=32 pixels)
             "spatial_cells": The number of cells in the nxy axis.
               (default: nxy=2 cells)
             "temporal_cells": The number of cells in the nt axis. 
               (default: nt=3 cells)
             "verbose": Display information (True/False).
             "n_jobs": Number of jobs for running in parallel. 
               (default: n_jobs=2)
         return:
           failed_computation: List containing files which feature 
             extraction fails.
    ____________________________________________________________________
    """
    ############################################################################
    opts = {"trajectory_length":15, "sampling_stride":5, "neighborhood_size":32,
            "spatial_cells":2, "temporal_cells":3, "verbose":False,
            "n_jobs": 2}
    if args is not None:
        if isinstance(args, dict):
            opts.update(args)
        else:
            print "Error: Third argument should be a dictionary."
            sys.exit(0)
    failed_computation = []
    if not os.path.isfile(idt_bin):
        print "Error: Improved dense trajectory binary not found."
        sys.exit(0)
    if not os.path.isdir(output_path):
        if opts["verbose"]:
            print "Creating directory: {}".format(output_path)
        os.mkdir(output_path)
    if opts["verbose"]:
        print "# Feature Extraction: Improved Trajectories using\
               Fundamental Matrix."
    idt_args = itertools.izip(video_files, itertools.repeat(output_path),
                             itertools.repeat(idt_bin), itertools.repeat(opts))
    f_pool = Pool(opts["n_jobs"])
    for idx, failed_computation in enumerate(\
      f_pool.imap_unordered(daemon_improved_trajectories_call, idt_args),1):
        sys.stderr.write('\rPercentage video processed: %0.2f%%' % \
                                              (100*idx/(0.0+len(video_files))))
    print "\nFinish!\n"
    f_pool.close()
    f_pool.join()
    ############################################################################
    return failed_computation
    ############################################################################

def improved_trajectories_call(video_file, output_path, idt_bin, opts):
    if not os.path.isfile(video_file):
        if opts["verbose"]:
            print "Failed computation: {}".format(video_file)
        return video_file
    command = "{0} '{1}' -L {2} -W {3} -N {4} -s {5} -t {6} -o {7}".format(
              idt_bin, video_file, opts["trajectory_length"],
              opts["sampling_stride"], opts["neighborhood_size"],
              opts["spatial_cells"], opts["temporal_cells"], output_path)
    os.system(command)
    if opts["verbose"]:
        print "Processed video {}".format(video_file)
    return None

def daemon_improved_trajectories_call(args):
    return improved_trajectories_call(*args)

def parse_dense_trajectories(raw_feature_file, *args):
    """
    ____________________________________________________________________
       parse_dense_trajectories: 
         Read dense trajectories info and descriptors from raw feature 
         file. Please check the output format specified in:
         http://lear.inrialpes.fr/~wang/dense_trajectories
         args:
           raw_feature_file: Full path for txt file containing the
             output of dense trajectory execution.
           output_path (Optional): If provided descriptors and track 
             info will be divided and stored in different files.
           opts: Give the following parameters if you change default 
             dense trajectory extraction.
             "trajectory_length": Length of tracking trajectries.
               (default: L=15)
             "sampling_stride": the stride for dense sampling feature 
               points. (default: W=5 pixels)
             "neighborhood_size": The neighborhood size for computing 
               the descriptor. (default: N=32 pixels)
             "spatial_cells": The number of cells in the nxy axis.
               (default: nxy=2 cells)
             "temporal_cells": The number of cells in the nt axis. 
               (default: nt=3 cells)
           *** If you provide opts please specify output_path (None if
           storage not needed.)
         return:
           track_info: np array containing information about trajectory.
           descriptors: List containing np arrays for each descriptor. 
             The order is: [[Trajectory], [HOG], [HOF], [MBHx], [MBHy]]           
    ____________________________________________________________________
    """
    ############################################################################
    opts = {"trajectory_length":15, "sampling_stride":5, "neighborhood_size":32,
            "spatial_cells":2, "temporal_cells":3}
    output_path = None
    failed_computation = None
    if len(args)==2:
        if isinstance(args[1], dict):
            opts.update(args[1])
        else:
            print "Error: Third argument should be a dictionary."
    if len(args)>0:
        if isinstance(args[0], str):
            output_path = args[0]
        else:
            print "Error: Third argument should be an string."
    if not os.path.isfile(raw_feature_file):
        print "Error: Raw feature file not found."
        failed_computation = raw_feature_file
    if output_path is not None:
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        video_id = os.path.basename(raw_feature_file).split('.')[0]
        descriptors_name = ["Trajectory", "HOG", "HOF", "MBHx", "MBHy"]
    descriptors_length = np.array([2*opts["trajectory_length"],
                            8*opts["spatial_cells"]**2*opts["temporal_cells"],
                            9*opts["spatial_cells"]**2*opts["temporal_cells"],
                            8*opts["spatial_cells"]**2*opts["temporal_cells"],
                            8*opts["spatial_cells"]**2*opts["temporal_cells"]])
    data = np.loadtxt(raw_feature_file)
    track_info = data[:,:9]
    if output_path is not None:
        output_name = os.path.join(output_path,
                    "{0}_{1}.hdf5".format("TrackInfo", video_id))
        if not os.path.isfile(output_name):
            dump = h5py.File(output_name)
            dump.create_dataset("TrackInfo", data=track_info)
            dump.close()        
    start_idx = 9
    descriptors = []
    for idx, this_length in enumerate(descriptors_length):
        end_idx = start_idx + this_length
        this_descriptor = data[:,start_idx:end_idx]
        start_idx = end_idx
        if output_path is not None:
            output_name = os.path.join(output_path,
                    "{0}_{1}.hdf5".format(descriptors_name[idx], video_id))
            if os.path.isfile(output_name):
                continue
            dump = h5py.File(output_name)
            dump.create_dataset(descriptors_name[idx], data=this_descriptor)
            dump.close()
            continue
        descriptors.append(this_descriptor)
    ############################################################################
    if output_path is not None:
        return failed_computation
    return track_info, descriptors
    ############################################################################

def format_features(raw_feature_file_list, output_path, *args):
    """
    ____________________________________________________________________
       parse_dense_trajectories: 
         Read dense trajectories info and descriptors from raw feature 
         file. Please check the output format specified in:
         http://lear.inrialpes.fr/~wang/dense_trajectories
         args:
           raw_feature_file_list: List containing txt files 
             full path of raw features.
           output_path: Full path where formated feature files will be 
             located.
           opts (Optional): Give the following parameters if you change 
             default dense trajectory extraction.
             "trajectory_length": Length of tracking trajectries.
               (default: L=15)
             "sampling_stride": the stride for dense sampling feature 
               points. (default: W=5 pixels)
             "neighborhood_size": The neighborhood size for computing 
               the descriptor. (default: N=32 pixels)
             "spatial_cells": The number of cells in the nxy axis.
               (default: nxy=2 cells)
             "temporal_cells": The number of cells in the nt axis. 
               (default: nt=3 cells)
             "n_jobs": Number of jobs for running in parallel. 
               (Default n_jobs=2)
           *** If you provide opts please specify output_path (None if
           storage not needed.)
         return:
           track_info: np array containing information about trajectory.
           descriptors: List containing np arrays for each descriptor. 
             The order is: [[Trajectory], [HOG], [HOF], [MBHx], [MBHy]]           
    ____________________________________________________________________
    """
    ############################################################################
    opts = {"trajectory_length":15, "sampling_stride":5, "neighborhood_size":32,
            "spatial_cells":2, "temporal_cells":3, "n_jobs":2}    
    if len(args)==1:
        if isinstance(args[0], dict):
            opts.update(args[0])
        else:
            print "Error: Third argument should be a dictionary."
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    format_args = itertools.izip(raw_feature_file_list, 
                                 itertools.repeat(output_path),
                                 itertools.repeat(opts))
    f_pool = Pool(opts["n_jobs"])
    for idx, failed_computation in enumerate(\
      f_pool.imap_unordered(daemon_parse_dense_trajectories, format_args),1):
        sys.stderr.write('\rPercentage video processed: %0.2f%%' % \
                                (100*idx/(0.0+len(raw_feature_file_list))))
    print "\nFinish!\n"
    f_pool.close()
    f_pool.join()
    print "Formating fails for this files: \n", failed_computation
    return failed_computation

def daemon_parse_dense_trajectories(args):
    return parse_dense_trajectories(*args)
