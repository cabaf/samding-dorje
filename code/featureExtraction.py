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

def extract_features(video_files, output_path, idt_bin, **args):
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

