'''
   _____________________________________________________________________
   codebookFormation.py
   ====================
   Includes several functions related with visual embedding creation.

   Copyright @ Fabian Caba H.
   _____________________________________________________________________
'''
import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from yael.ynumpy import gmm_learn

def kmeans_voc(visual_world, dictionary_size, *conf):
    """
    ____________________________________________________________________
       kmeans_voc:
         Construct a visual dictionary with k-means algorithm using an
         euclidean distance.
         args:
           visual_world: NxM np array where N is the number of visual
             examples and M is the length of dimensionality.
           dictionary_size: Number of visual words K (kmeans's centers).
           conf(optional): Dictionary including some configurations.
             "kmeans_verbose": Display progress or not (True,False)
             "kmeans_iterations": Max number of iterations for k-means.
             "kmeans_ntrial": Number of re-starts for robust estimation.
             "kmeans_njobs": Number of jobs for parallel speed up.
         return:
           v_words: KxM np array coordinates of clusters centers.
    ____________________________________________________________________
    """
    ############################################################################
    opts = {"kmeans_verbose":False, "kmeans_iterations":300, "kmeans_ntrial":8, 
            "kmeans_njobs":2}
    if len(conf) > 0:
        if isinstance(conf[0], dict):
            opts.update(conf[0])
        else:
            print "Warning: Options not override, third argument should be a "\
                  "dictionary."
    km = KMeans(n_clusters=dictionary_size, verbose=opts["kmeans_verbose"],
                n_init=opts["kmeans_ntrial"], 
                max_iter=opts["kmeans_iterations"], n_jobs=opts["kmeans_njobs"])
    km.fit(visual_world)
    v_words = km.cluster_centers_
    ############################################################################
    return v_words
    ############################################################################

def gmm_voc(visual_world, n_gaussians, *conf):
    """
    ____________________________________________________________________
       gmm_voc:
         Construct a gaussian mixture model to describe the distribution
         over feature space.
         args:
           visual_world: NxM np array where N is the numer of visual
             examples and M is the legth of dimensionality.
           n_gaussians: Number of gaussians to model feature space.
         return:
           v_words: KxM np array where K is the number of gaussians and 
             M is the length of dimensionality.
    ____________________________________________________________________
    """
    ############################################################################
    # Required both: c-contigous and float32 array.
    visual_world = np.ascontiguousarray(visual_world.astype(np.float32))
    n_gaussians = np.int(n_gaussians)
    v_words = gmm_learn(visual_world, n_gaussians)
    ############################################################################
    return v_words
    ############################################################################

def codebook_generation(visual_world, n_words, *conf):
    """ 
    ____________________________________________________________________
       codebook_generation:
         Compute codebook for set of files. It could be based on kmeans
         or based on GMM.
         args:
           visual_world: NxM np array where N is the number of visual
             examples and M is the length of dimensionality.
           n_words: Number of visual words.
           conf: It must contain the following dictionary key/value:
             "codebook_type": "gmm" or "kmeans".
             "pca_filename" (REQUIRED if PCA): Full path where learned 
               model will be stored.
             "whiten": The components_ vectors are divided by the 
             singular values to ensure uncorrelated outputs with unit 
             component-wise variances (Default: True)
             "reduction_rate": Rate to reduce feature dimensionality.
               (Default: 1 -- No reduction.)
             ******(See options in help of *_voc functions.)********
         return:
           v_words: np array containing computed codebook.
    ____________________________________________________________________
    """
    ############################################################################
    opts = {"codebook_type":"kmeans"}
    if len(conf) > 0:
        if isinstance(conf[0], dict):
            opts.update(conf[0])
        else:
            print "Warning: Opts not override. See help."
    if opts["codebook_type"] is "kmeans":
        v_words = kmeans_voc(visual_world, n_words, opts)
    elif opts["codebook_type"] is "gmm":
        try:
            output_filename = opts["pca_filename"]
            visual_world = apply_pca_to_visual_world(visual_world,
                                                     output_filename, opts)
        except:
            print "Warning: No PCA performed. You must indicate where save model."                    
        v_words = gmm_voc(visual_world, n_words)
    else:
        print "Error: Unknown codebook type."
        v_words = None
    ############################################################################
    return v_words
    ############################################################################

def read_and_sample_features(file_name, sample_ratio, file_key):
    """
    ____________________________________________________________________
       read_and_sample_features:
       args:
         file_name: full path where feature is stored.
         sample_ratio: Ratio for amount of features sampled from original 
           file. 1 means not sampling.
         file_key: Dataset name, for hdf5 storage.
       return:
         features: NxM np array containing N sample features. M is the 
           length of feature dimension.
    ____________________________________________________________________
    """
    if not os.path.isfile(file_name):
        print "Error: File not found."
        return None
    dump = h5py.File(file_name)
    features = np.array(dump[file_key])
    np.random.shuffle(features)
    if sample_ratio<1:
        features = features[:np.int(features.shape[0]*sample_ratio),:]
    return features

def daemon_read_and_sample_features(args):
    return read_and_sample_features(*args)

def apply_pca_to_visual_world(visual_world, output_filename, *conf):
    """
    ____________________________________________________________________
       apply_pca_to_visal_world:
         Learn a pca model from input data, transform visual training 
         world, and write learned model on disk.
         args:
           visual_world: NxM np array where N is the number of visual
             examples and M is the length of dimensionality.             
           output_filename: Full path where learned model will be stored.           
           conf (Optional): Configuration parameters for learning and 
             applying PCA.
             "whiten": The components_ vectors are divided by the 
             singular values to ensure uncorrelated outputs with unit 
             component-wise variances (Default: True)
             "reduction_rate": Rate to reduce feature dimensionality.
             (Default: 1 -- No reduction.)
    ____________________________________________________________________
    """
    opts = {"whiten":True, "reduction_rate":1}
    if len(conf)>0:
        if isinstance(conf[0], dict):
            opts.update(conf[0])
        else:
            print "Warning: Opts not override. See help."
    output_path = os.path.dirname(output_filename)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # Learn PCA model for visual world.
    pca_dim = np.int(visual_world.shape[1]/opts["reduction_rate"])
    pca_model = RandomizedPCA(n_components=pca_dim, whiten=opts["whiten"])
    pca_model.fit(visual_world)
    # Apply learned model to training data.
    visual_world = pca_model.transform(visual_world)
    # Write on disk learned model.    
    pickle.dump(pca_model, open(output_filename, "w"))
    return visual_world
