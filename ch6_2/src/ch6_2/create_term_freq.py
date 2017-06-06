#!/usr/bin/env python
__author__ = 'p_duckworth'
import os, sys, csv
import time
import scipy
import numpy as np
import cPickle as pickle
import getpass

def load_term_frequency(directory, run):

    # ****************************************************************************************************
    # load word counts
    # ****************************************************************************************************
    r = "QSR_path/run_%s" % run
    directory = os.path.join(directory, r)
    print "directory: ", directory

    if not os.path.exists(os.path.join(directory, 'TopicData')): os.makedirs(os.path.join(directory, 'TopicData'))
    if not os.path.exists(os.path.join(directory, 'oldaData')): os.makedirs(os.path.join(directory, 'oldaData'))

    with open(directory+"/codebook_data.p", 'r') as f:
        loaded_data = pickle.load(f)
        # (global_codebook, all_graphlets, uuids, labels) = pickle.load(f)
    # print ">>", len(global_codebook), len(all_graphlets), len(uuids), len(labels)
    code_book = loaded_data[0]
    graphlets = loaded_data[1]
    codebook_lengh = len(code_book)

    uuids, wordids, wordcts = [], [], []
    true_videos, true_labels = [], []
    num_of_vids = len(os.listdir(directory))
    for task in xrange(1, num_of_vids+1):

        if task in [196, 211]: continue
        video = "vid%s.p" % task
        d_video = os.path.join(directory, video)
        if not os.path.isfile(d_video): continue
        with open(d_video, 'r') as f:
            try:
                (uuid, label, ids, histogram) = pickle.load(f)
            except:
                "failed to load properly: \n %s" % (video)

        wordids.append(ids)
        wordcts.append(histogram)
        true_videos.append(video)
        true_labels.append(label)
    print "#videos: ", len(wordids), len(wordcts)
    online_data = wordids, wordcts, true_labels, true_videos

    # ****************************************************************************************************
    # Term-Freq Matrix
    # ****************************************************************************************************
    term_freq = term_frequency_mat(codebook_lengh, wordids, wordcts)

    return term_freq, online_data, code_book, graphlets


def term_frequency_mat(codebook_lengh, wordids, wordcnts):
    term_freq = []
    for counter, (ids, cnts) in enumerate(zip(wordids, wordcnts)):
        # print ids, cnts
        vec = np.array([0] * codebook_lengh)
        for i, cnt in zip(ids, cnts):
            # print i, cnt
            vec[i] = cnt
        # print "vec: ", vec
        term_freq.append(vec)
        # print "tf: ", term_freq
    feature_space = np.vstack(term_freq)

    # for e, i in enumerate(term_freq):
    #     print e, sum(i)
    #     if sum(i) == 0:
    #         print ">", wordids[e], wordcnts[e]
    # pdb.set_trace()
    return feature_space


def high_instance_code_words(term_frequency, code_book, graphlets, low_instance):
    """This essentially takes the feature space created over all events, and removes any
    feature that is not witnessed a minimum number of times (low_instance param).
    """
    ## Number of rows with non zero element :
    keep_rows = np.where((term_frequency != 0).sum(axis=0) > low_instance)[0]
    ## Sum of the whole column: term_frequency.sum(axis=0) > low_instance
    remove_inds = np.where((term_frequency != 0).sum(axis=0) <= low_instance)[0]

    print "orig feature space: %s. remove: %s. new space: %s." % (len(term_frequency.sum(axis=0)), len(remove_inds), len(keep_rows))

    #keep only the columns of the feature space which have more than low_instance number of occurances.
    selected_features = term_frequency.T[keep_rows]
    new_term_frequency = selected_features.T
    print "new feature space shape: ", new_term_frequency.shape

    # # Code Book (1d np array of hash values)
    new_code_book = code_book[keep_rows]
    print "  new code book len: ", len(new_code_book)

    # # Graphlets book (1d np array of igraphs)
    new_graphlets = graphlets[keep_rows]
    print "  new graphlet book len: ", len(new_graphlets)

    print "removed low (%s) instance graphlets" % low_instance
    print "shape = ", new_term_frequency.shape

    return new_term_frequency, new_code_book, new_graphlets
