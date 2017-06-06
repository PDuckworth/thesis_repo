#!/usr/bin/env python
__author__ = 'p_duckworth'
import os, sys, csv
import time
import scipy
import numpy as np
import cPickle as pickle
import getpass

def term_frequency(directory, run):

    # ****************************************************************************************************
    # load word counts
    # ****************************************************************************************************
    uuids, wordids, wordcts = [], [], []
    directory += "/QSR_path/run_%s" % run
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

    return term_freq, online_data


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
