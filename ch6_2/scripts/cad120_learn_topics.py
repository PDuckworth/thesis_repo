#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import cPickle as pickle
import math
import numpy as np
import getpass
# from onlineldavb import OnlineLDA
import pdb
import lda
import copy
import pyLDAvis
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from scipy.stats import threshold

import ch6_2.create_term_freq as fr
import ch6_2.LSA as lsa
import ch6_2.svm as supervised


def comare_genomes(data1, data2):
    t = np.arange(0.0, len(data1), 1)
    print "\ndata1", min(data1), max(data1), sum(data1)/float(len(data1))
    print "data2", min(data2), max(data2), sum(data2)/float(len(data2))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    # ax1.plot(t, data1, 'b^')
    ax1.vlines(t, [0], data1)
    ax1.set_xlabel('code words', fontsize=20)
    ax1.set_ylim([0,0.1])
    # ax1.set_title('Latent Concept 1', fontsize=25)
    ax1.set_title('Topic 1', fontsize=25)
    ax1.grid(True)

    ax2 = fig.add_subplot(212)
    # ax1.plot(t, data2, 'b^')
    ax2.vlines(t, [0], data2)
    ax2.set_xlabel('code words', fontsize=20)
    ax2.set_ylim([0, 0.1])
    # ax2.set_title('Latent Concept 2', fontsize=25)
    ax2.set_title('Topic 2', fontsize=25)
    ax2.grid(True)
    plt.show()


def genome_rel(data1, i, ymax=0):
    t = np.arange(0.0, len(data1), 1)
    print "\ndata1", min(data1), max(data1), sum(data1)/float(len(data1))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.vlines(t, [0], data1)
    ax1.set_xlabel('code words', fontsize=20)
    # ax1.set_ylim([min(data1), max(data1)])
    ax1.set_title('Topic %s' %i, fontsize=25)
    ax1.grid(True)
    plt.show()

def genome(data1, i, ymax=0.06):
    t = np.arange(0.0, len(data1), 1)
    print "\ndata1", min(data1), max(data1), sum(data1)/float(len(data1))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.vlines(t, [0], data1)
    ax1.set_xlabel('code words', fontsize=20)
    ax1.set_ylim([0,ymax])
    ax1.set_title('Topic %s' %i, fontsize=25)
    ax1.grid(True)
    plt.show()

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

def get_dic_codebook(code_book, graphlets, create_graphlet_images=False):
    """code book already contains stringed hashes. """
    dictionary_codebook = dict(zip(code_book, graphlets))

    # dictionary_codebook = {}
    # for hash, graph in zip(code_book, graphlets):
    #     dictionary_codebook[g_name] = graph
    if create_graphlet_images:
        image_path = '/home/' + getpass.getuser() + '/Dropbox/Programming/topics_to_language/LDAvis_images'
        create_codebook_images(dictionary_codebook, image_path)
    return dictionary_codebook

def object_nodes(graph):
    object_nodes = []
    num_of_eps = 0
    spatial_nodes = []
    temp_nodes = []
    for node in graph.vs():
        if node['node_type'] == 'object':
            #if node['name'] not in ["hand", "torso"]:
            object_nodes.append(node['name'])
        if node['node_type'] == 'spatial_relation':
            spatial_nodes.append(node['name'])
            num_of_eps+=1
        if node['node_type'] == 'temporal_relation':
            temp_nodes.append(node['name'])
    return object_nodes, spatial_nodes, temp_nodes

def print_results(true_labels, pred_labels, num_clusters):
    (h, c, v) =  metrics.homogeneity_completeness_v_measure(true_labels, pred_labels)

    print "#Topics=%s (%s). v-measure: %0.3f. homo: %0.3f. comp: %0.3f. MI: %0.3f. NMI: %0.3f. Acc: %0.3f" \
      % (num_clusters, len(pred_labels), v, h, c,
        metrics.mutual_info_score(true_labels, pred_labels),
        metrics.normalized_mutual_info_score(true_labels, pred_labels),
        metrics.accuracy_score(true_labels, pred_labels))

    row_inds = {}
    yaxis=[]
    set_of_true_labs = list(set(true_labels))
    for cnt, i in enumerate(set_of_true_labs):
        yaxis.append(i)
        row_inds[i] = cnt
    # print "row inds",  row_inds

    res = {}
    mat = np.zeros( (len(set_of_true_labs), num_clusters))

    for i, j in zip(true_labels, pred_labels):
        row_ind = row_inds[i]
        mat[row_ind][j] +=1
        try:
            res[i].append(j)
        except:
            res[i] = [j]

    # for cnt, i in enumerate(mat):
    #     print "label: %s: %s" % (set_of_true_labs[cnt], i)

    # pdb.set_trace()

    # Find assignments:
    # for true, preds in res.items():
    #     print true, max(set(preds), key=preds.count)

    norm_mat = []
    import pdb; pdb.set_trace()

    for i in mat:
        norm_mat.append([float(j)/float( sum(i) ) for j in i ])

    # f, ax = plt.subplots(nrows=1, ncols=1)
    # f.suptitle("Topic Confusion Matrix: using segmented clips")
    # plt.xlabel('Learnt Topics')
    # plt.ylabel('Ground Truth Labels')
    # plt.setp((ax), xticks=range(len(mat[0])), yticks=xrange(len(set_of_true_labs)))
    # plt.yticks(xrange(len(set_of_true_labs)), set_of_true_labs, color='red')
    # ax.imshow(norm_mat, interpolation='nearest')
    # plt.show()

    reorder = np.argmax(mat, axis=1)
    reorder = [3, 8, 1 , 5, 0, 2, 4, 7, 6, 9]

    classes = ["a","b","c","d","e","f","g","h","i","j"]
    plt.clf()
    cmap=plt.cm.Blues
    plt.imshow([norm_mat[i] for i in reorder], interpolation='nearest', cmap=cmap)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=18) #, rotation=90)
    plt.yticks(tick_marks, [yaxis[i] for i in reorder], color='red', fontsize=18)

    plt.title("LDA Topic Confusion Matrix: CAD120", fontsize=24)
    plt.xlabel('Learned Topics', fontsize=20)
    plt.ylabel('Ground Truth Labels', fontsize=20)

    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # width, height = reordered.shape
    # for x in xrange(width):
    #     for y in xrange(height):
    #         ax2.annotate("%0.2f" % reordered[x][y], xy=(y, x), horizontalalignment='center', verticalalignment='center')

    # plt.show()

    return (num_clusters, len(pred_labels),
       metrics.v_measure_score(true_labels, pred_labels),
       metrics.homogeneity_score(true_labels, pred_labels),
       metrics.completeness_score(true_labels, pred_labels),
       metrics.mutual_info_score(true_labels, pred_labels),
       metrics.normalized_mutual_info_score(true_labels, pred_labels),
       metrics.accuracy_score(true_labels, pred_labels),
       mat, set_of_true_labs)

def run_topic_model(X, loaded_data, n_iters, n_topics, create_graphlet_image, (alpha, eta), class_thresh=0):

    code_book, graphlets_, uuids, true_labels = loaded_data
    graphlets = get_dic_codebook(code_book, graphlets_, create_graphlet_images)
    print "sum of all data: X.shape: %s and X.sum: %s" % (X.shape, X.sum())

    model = lda.LDA(n_topics=n_topics, n_iter=n_iters, random_state=1, alpha=alpha, eta=eta)
    model.fit(X)

    feature_freq = (X != 0).sum(axis=0)
    doc_lengths = (X != 0).sum(axis=1)

    print "phi: %s. theta: %s. nd: %s. vocab: %s. Mw: %s" \
        %( model.topic_word_.shape, model.doc_topic_.shape, doc_lengths.shape, len(graphlets.keys()), len(feature_freq))

    vis_data = pyLDAvis.prepare(model.topic_word_, model.doc_topic_, doc_lengths, graphlets.keys(), feature_freq)
    html_file = "/home/"+getpass.getuser()+"/Dropbox/Programming/topics_to_language/topic_model_cad.html"
    pyLDAvis.save_html(vis_data, html_file)
    print "PyLDAVis ran. output: %s" % html_file
    return model

def investigate_topics(model, loaded_data, labels, videos, prob_of_words, language_indices, _lambda, n_top_words = 30):
    """investigate the learned topics
    Relevance defined: http://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf
    """
    topic_word = model.topic_word_
    doc_topic = model.doc_topic_
    code_book, graphlets_, uuids, miss_labels = loaded_data
    print "\n1"
    true_labels = labels
    vocab = [hash for hash in list(code_book)]
    graphs = loaded_data[1]

    # ****************************************************************************************************
    # Relevance
    # ****************************************************************************************************
    # print("\ntype(topic_word): {}".format(type(topic_word)))
    # print("shape: {}".format(topic_word.shape))
    print "objects in each topic: "

    topics = {}
    relevant_objects = {}
    for i, topic_dist in enumerate(topic_word):
        objs = []
        top_words_in_topic = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        # print('Topic {}: {}'.format(i, ' '.join( [repr(ii) for ii in top_words_in_topic] )))
        for k in top_words_in_topic:
            j = graphlets_[np.where(code_book == k)[0][0]]
            objs.extend(object_nodes(j)[0])

        topics[i] = objs

        relevant_objects[i] = list(set(objs))
        # print('Topic {}: {}'.format(i, relevant_objects[i]))
        # print top_words_in_topic

    # #Each document's most probable topic
    restricted_labels, restricted_videos = [], []
    pred_labels = []

    for n in xrange(doc_topic.shape[0]):
        #print [p for p in doc_topic[n] if p >= 0.0]  # each document probabilities to each topic
        if max(doc_topic[n]) > class_thresh:
            # print true_labels[n]
            # print doc_topic[n]
            # print doc_topic[n].argmax()
            # doc_topic[n][doc_topic[n].argmax()] = 0
            restricted_labels.append(true_labels[n])
            restricted_videos.append(videos[n])
            topic_most_pr = doc_topic[n].argmax()
            pred_labels.append(topic_most_pr)

        #if dbg: print("doc: {} topic: {}".format(n, topic_most_pr))
    true_labels = restricted_labels
    videos = restricted_videos
    # print "\n2"
    return true_labels, pred_labels, videos, relevant_objects

def invesitgate_videos_given_topics(pred_labels, video_list):
    # print "\nvideos assigned to each topic: "
    # print "\n3"

    topics_to_videos = {}
    num_of_topics = max(pred_labels)+1
    for topic_id in xrange(num_of_topics):
        topics_to_videos[topic_id] = []
        for label, video in zip(pred_labels, video_list):
            if topic_id == label:
                topics_to_videos[topic_id].append(video.replace("vid", ""))
        # print "\ntopic: %s:  %s" % (topic_id, topics_to_videos[topic_id])

    return

if __name__ == "__main__":

    # if len(sys.argv) < 2:
    #     print "Usage: please provide a QSR run folder number."
    #     sys.exit(1)
    # else:
    #     run = sys.argv[1]

    # ****************************************************************************************************
    # parameters
    # ****************************************************************************************************
    n_iters = 1000
    n_topics = 10
    create_graphlet_images = False
    dirichlet_params = (0.3, 0.05)
    class_thresh = 0.3
    # _lambda = 0.5
    _lambda = 0

    using_language = False
    # ****************************************************************************************************
    # load word counts
    # ****************************************************************************************************
    uuids, wordids, wordcts = [], [], []
    directory = '/home/'+getpass.getuser()+'/Datasets/CAD120/qsr_data'
    # if not os.path.exists(os.path.join(directory, 'TopicData')):
    #     os.makedirs(os.path.join(directory, 'TopicData'))

    print "directory: ", directory
    with open(directory+"/codebook_data.p", 'r') as f:
        loaded_data = pickle.load(f)

    vocab, graphlets, video_ids, labels = loaded_data
    # vocab = loaded_data[0]
    code_book = vocab
    codebook_lengh = len(vocab)

    # video_ids = loaded_data[2]
    num_of_vids = len(video_ids)

    ordered_list_of_video_names, ordered_list_of_true_labels = [], []
    for task in video_ids:
        d_video = os.path.join(directory, task+".p")
        if not os.path.isfile(d_video): continue
        with open(d_video, 'r') as f:
            try:
                (uuid, label, ids, histogram) = pickle.load(f)
            except:
                "failed to load properly: \n %s" % (d_video)

        wordids.append(ids)
        wordcts.append(histogram)
        ordered_list_of_video_names.append(task)
        ordered_list_of_true_labels.append(label)

    print "#videos: ", len(wordids), len(wordcts)
    # ****************************************************************************************************
    # Term-Freq Matrix
    # ****************************************************************************************************
    term_freq = term_frequency_mat(codebook_lengh, wordids, wordcts)
    language_indices = ""

    low_pass_instance = 5
    term_freq_red, code_book_red, graphlets_red =  fr.high_instance_code_words(term_freq, code_book, graphlets, low_pass_instance)

    # ****************************************************************************************************
    # Supervised SVM
    # ****************************************************************************************************
    labels = ordered_list_of_true_labels
    supervised.run_svm(term_freq, labels)

    # ****************************************************************************************************
    # Unsupervised kmeans
    # ****************************************************************************************************
    supervised.kmeans_clustering(term_freq, labels, threshold=0.01)

    # ****************************************************************************************************
    # call batch LSA
    # ****************************************************************************************************
    print "\nLSA: "
    lsa.svd_clusters(term_freq_red, labels, n_comps=10, threshold=0.0)
    print "LSA done"

    # ****************************************************************************************************
    # random clustering
    # ****************************************************************************************************
    lsa.svd_clusters(term_freq_red, labels, n_comps=10, random=True)
    print "Random done"


    # ****************************************************************************************************
    # call batch LDA
    # ****************************************************************************************************

    relevant_words = {100:{}}

    # term_freq = term_freq_red
    # loaded_data = (code_book_red, graphlets_red, uuids, ordered_list_of_true_labels)

    model =  run_topic_model(term_freq, loaded_data, n_iters, n_topics, create_graphlet_images, dirichlet_params, class_thresh)

    # for i, j in itertools.combinations(xrange(11), 2):
    #     print "Topics: %s and %s" %(i, j)
    #     compare_genome(model.topic_word_[i], model.topic_word_[j])
    # for i in xrange(11):
    #     genome(model.topic_word_[i], i)

    # ****************************************************************************************************
    # find the probability of each word from the term frequency matrix
    # ****************************************************************************************************

    sum_of_graphlet_words = term_freq.sum()
    sum_of_all_words = 0
    probability_of_words = term_freq.sum(axis = 0) / float(sum_of_graphlet_words)



    # ****************************************************************************************************
    # save each document distribution
    # ****************************************************************************************************
    percentage = ""

    name = '/document_topics_%s.p' % percentage
    f1 = open(directory+name, 'w')
    pickle.dump(model.doc_topic_, f1, 2)
    f1.close()

    name = '/topic_words_%s.p' % percentage
    f1 = open(directory+name, 'w')
    pickle.dump(model.topic_word_, f1, 2)
    f1.close()

    # ****************************************************************************************************
    # investigate the relevant words in each topic, and see which documents are classified into each topic
    # ****************************************************************************************************
    labels = ordered_list_of_true_labels
    vidoes = ordered_list_of_video_names
    true_labels, pred_labels, videos, relevant_words[percentage] = investigate_topics(model, loaded_data, labels, vidoes, probability_of_words, language_indices, _lambda, n_top_words=30)

    invesitgate_videos_given_topics(pred_labels, videos)
    print "\nvideos classified:", len(true_labels), len(pred_labels)

    # ****************************************************************************************************
    # Get results
    # ****************************************************************************************************
    n, l, v, h, c, mi, nmi, a, mat, labs = print_results(true_labels, pred_labels, n_topics)

    # ****************************************************************************************************
    # Write out results
    # ****************************************************************************************************
    name = '/results_%s.txt' % percentage
    f1 = open(directory+name, 'w')
    f1.write('n_topics: %s \n' % n_topics)
    f1.write('n_iters: %s \n' % n_iters)
    f1.write('dirichlet_params: (%s, %s) \n' % (dirichlet_params[0], dirichlet_params[1]))
    f1.write('class_thresh: %s \n' % class_thresh)
    f1.write('code book length: %s \n' % codebook_lengh)
    f1.write('sum of all words: %s \n' % sum_of_all_words)
    f1.write('videos classified: %s \n \n' % len(pred_labels))

    f1.write('v-score: %s \n' % v)
    f1.write('homo-score: %s \n' % h)
    f1.write('comp-score: %s \n' % c)
    f1.write('mi: %s \n' % mi)
    f1.write('nmi: %s \n \n' % nmi)
    f1.write('mat: \n')

    headings = ['{:3d}'.format(int(r)) for r in xrange(n_topics)]
    f1.write('T = %s \n \n' % headings)
    for row, lab in zip(mat, labs):
        text_row = ['{:3d}'.format(int(r)) for r in row]
        f1.write('    %s : %s \n' % (text_row, lab))
    f1.write('\n')
    f1.write('relevant_words: \n')

    for i, words in relevant_words[percentage].items():
        f1.write('Topic %s : %s \n' % (i, words[:10]))
    f1.close()


    # ****************************************************************************************************
    # Write out confusion matrix as array and dictionary
    # ****************************************************************************************************
    confusion_dic    = {}
    labels = []
    for row, lab in zip(mat, labs):
        labels.append(lab)
        confusion_dic[lab] = row

    name = '/confusion_mat%s.p' % percentage
    f1 = open(directory+name, 'w')
    pickle.dump(mat, f1, 2)
    f1.close()

    name = '/confusion_dic%s.p' % percentage
    f1 = open(directory+name, 'w')
    pickle.dump(confusion_dic, f1, 2)
    f1.close()
