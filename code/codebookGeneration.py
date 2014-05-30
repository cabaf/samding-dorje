'''
   _____________________________________________________________________
   codebookFormation.py
   ====================
   Includes several functions related with visual embedding creation.

   Copyright @ Fabian Caba H.
   _____________________________________________________________________
'''
import numpy as np
from sklearn.cluster import KMeans
from yael.ynumpy import gmm_learn

def kmeans_voc(visual_world, dictionary_size, **conf):
    """
    ____________________________________________________________________
       kmeans_voc:
         Construct a visual dictionary with k-means algorithm using an
         euclidean distance.
         args:
           visual_world: NxM np array where N is the number of visual
             examples and N is the length of dimensionality.
           dictionary_size: Number of visual words K (kmeans's centers).
           conf(optional): Dictionary including some configurations.
             "kmeans_verbose": Display progress or not (True,False)
             "kmeans_iterations": Max number of iterations for k-means.
             "kmeans_ntrial": Number of re-starts for robust estimation.
         return:
           v_words: KxM np array coordinates of clusters centers.
    ____________________________________________________________________
    """
    ############################################################################
    opts = {"kmeans_verbose":False, "kmeans_iterations":300, "kmeans_ntrial":8}
    if conf is not None:
       opts.update(conf)
    km = KMeans(n_clusters=dictionary_size, verbose=opts["kmeans_verbose"],
                n_init=opts["kmeans_trial"], max_iter=opts["kmeans_iterations"])
    km.fit(visual_world)
    v_words = km.cluster_centers_
    ############################################################################
    return v_words
    ############################################################################

def gmm_voc(visual_world, n_gaussians, **conf):
    """
    ____________________________________________________________________
       gmm_voc:
         Construct a gaussian mixture model to describe the distribution
         over feature space.
         args:
           visual_world: NxM np array where N is the numer of visual
             examples and N is the legth of dimensionality.
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

def codebook_generation(visual_world, n_words, conf):
    """
    ____________________________________________________________________
       codebook_generation:
         Compute codebook for set of files. It could be based on kmeans
         or based on GMM.
         args:
           visual_world: NxM np array where N is the numer of visual
             examples and N is the legth of dimensionality.
           n_words: Number of visual words.
           conf: It shouls contain the following dictionary key/value:
             "codebook_type": "gmm" or "kmeans".
             "codebook_sample_ratio": ratio for sampling features [0-1].
             "codebook_pca_whiten": Flag to conduct pca,
               useful for GMM codebook formation (True/False).
             "codebook_pca_reduction": Reduction rate of feature
               dimensionality when PCA is applied.
             "codebook_pca_path": Full path that will be used to write
               learned PCA model.
             "codebook_verbose": Display information (True/False).
         return:
           v_words: np array containing computed codebook.
    ____________________________________________________________________
    """
    ############################################################################
    opts = {"codebook_type":"gmm"}
    if conf is not None:
       opts.update(conf)
    if opts["codebook_type"] is "kmeans":
        v_words = kmeans_voc(visual_world, n_words, conf)
    elif opts["codebook_type"] is "gmm":
        v_words = gmm_voc(visual_world, n_words)
    else:
        print "Error: Unknown codebook type."
        v_words = None
    ############################################################################
    return v_words
    ############################################################################
