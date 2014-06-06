'''
   _____________________________________________________________________
   codebookFormation.py
   ====================
   Includes several functions for feature encoding, pooling and 
     normalization.

   Copyright @ Fabian Caba H.
   _____________________________________________________________________
'''
import pickle
from scipy.cluster.vq import vq as Quantize
import numpy as np
import h5py
import os
from scipy.io import savemat
import itertools
from multiprocessing import Pool
import sys

def vq_encoding(feat_filename, codebook_filename, data_key, output_path):
    """
       vq_encoding:
       TODO
    """
    raw_path = os.path.join(output_path, "rawVq")
    if not os.path.isdir(raw_path):
        os.makedirs(raw_path)
    encoded_path = os.path.join(output_path, "encodedVq")
    if not os.path.isdir(encoded_path):
        os.makedirs(encoded_path)
    file_basename = os.path.basename(feat_filename).split("{0}_".format(data_key))[1]
    track_info_filename = os.path.join(os.path.dirname(feat_filename),
                                       "{0}_{1}".format("TrackInfo", file_basename))
    dump = h5py.File(track_info_filename)
    # Normalized x,y,t feature position.
    track_info = np.array(dump["TrackInfo"])[:,7:10]
    dump.close()
    dump = h5py.File(feat_filename)
    feat = np.array(dump[data_key])
    dump.close()
    cb = pickle.load(open(codebook_filename))
    feat_code, feat_dist = Quantize(feat, cb)
    vq_feat = np.column_stack((track_info[:], feat_code))
    # Raw vq storage
    raw_filename = os.path.join(raw_path,
                                "{0}_{1}.mat".format(data_key, file_basename.split(".")[0]))
    if not os.path.isfile(raw_filename):
        savemat(raw_filename, dict(vq_feat=vq_feat))
    word_histogram, bin_edges = np.histogram(feat_code, 
                                         bins=range(cb.shape[0] + 1))
    encoded_filename = os.path.join(encoded_path,
                                    "{0}_{1}.hdf5".format(data_key, file_basename.split(".")[0]))
    if not os.path.isfile(encoded_filename):
        dump = h5py.File(encoded_filename)
        dump.create_dataset("feature_vector", data=word_histogram)
        dump.close()
    return word_histogram

def daemon_vq_encoding(args):
    return vq_encoding(*args)

def parallel_vq_encoding(feat_filename_list, codebook_filename, output_path,
                         data_key, num_jobs=2):
    f_args = itertools.izip(feat_filename_list, itertools.repeat(codebook_filename),
                            itertools.repeat(data_key), itertools.repeat(output_path))
    f_pool = Pool(num_jobs)
    for idx, failed_computation in enumerate(\
                                         f_pool.imap_unordered(daemon_vq_encoding, f_args),1):
        sys.stderr.write('\rPercentage video processed: %0.2f%%' % \
                         (100*idx/(0.0+len(feat_filename_list))))
    print "\nFinish!\n"
    f_pool.close()
    f_pool.join()
    return True
