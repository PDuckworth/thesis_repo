#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import cPickle as pickle
import copy
import math
import getpass
import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import threshold
import pdb
import lda
import pyLDAvis
import ch6_2.onlineldavb as OLDA
import ch6_2.visualisations as vis
import ch6_2.create_term_freq as fr
import ch6_2.LSA as lsa
import ch6_2.svm as supervised
from sklearn.metrics.pairwise import cosine_similarity
import hungarian

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

def just_topic_model(term_freq,   n_iters, n_topics, (alpha, eta)):
    X = term_freq
    print "sum of all data: X.shape: %s and X.sum: %s" % (X.shape, X.sum())

    model = lda.LDA(n_topics=n_topics, n_iter=n_iters, random_state=1, alpha=alpha, eta=eta)
    model.fit(X)

    feature_freq = (X != 0).sum(axis=0)
    doc_lengths = (X != 0).sum(axis=1)
    return model.topic_word_


def run_topic_model(term_freq, code_book, graphlets, online_data, directory, n_iters, n_topics, create_graphlet_images, (alpha, eta), class_thresh=0, _lambda=0.5):
    X = term_freq
    wordids, wordcts, true_labels, true_videos = online_data
    graphlets = get_dic_codebook(code_book, graphlets, create_graphlet_images)
    print "sum of all data: X.shape: %s and X.sum: %s" % (X.shape, X.sum())

    model = lda.LDA(n_topics=n_topics, n_iter=n_iters, random_state=1, alpha=alpha, eta=eta)
    model.fit(X)

    feature_freq = (X != 0).sum(axis=0)
    doc_lengths = (X != 0).sum(axis=1)

    print "phi: %s. theta: %s. nd: %s. vocab: %s. Mw: %s" \
        %( model.topic_word_.shape, model.doc_topic_.shape, doc_lengths.shape, len(graphlets.keys()), len(feature_freq))
    # vis_data = pyLDAvis.prepare(model.topic_word_, model.doc_topic_, doc_lengths, graphlets.keys(), feature_freq)
    # html_file = "/home/"+getpass.getuser()+"/Dropbox/Programming/topics_to_language/topic_model_ecai.html"
    # pyLDAvis.save_html(vis_data, html_file)
    # print "PyLDAVis ran. output: %s" % html_file


    # ****************************************************************************************************
    # find the probability of each word from the term frequency matrix
    # ****************************************************************************************************
    sum_of_graphlet_words = term_freq.sum()
    sum_of_all_words = 0
    probability_of_words = term_freq.sum(axis = 0) / float(sum_of_graphlet_words)

    # ****************************************************************************************************
    # save each document distribution
    # ****************************************************************************************************
    name = '/TopicData/document_topics.p'
    f1 = open(directory+name, 'w')
    pickle.dump(model.doc_topic_, f1, 2)
    f1.close()

    name = '/TopicData/topic_words.p'
    f1 = open(directory+name, 'w')
    pickle.dump(model.topic_word_, f1, 2)
    f1.close()

    name = '/TopicData/code_book.p'
    f1 = open(directory+name, 'w')
    pickle.dump(code_book, f1, 2)
    f1.close()

    name = '/TopicData/graphlets.p'
    f1 = open(directory+name, 'w')
    pickle.dump(graphlets, f1, 2)
    f1.close()

    vis.make_radar_plot(model.topic_word_)

    # ****************************************************************************************************
    # investigate the relevant words in each topic, and see which documents are classified into each topic
    # ****************************************************************************************************
    true_labels, pred_labels, videos, relevant_words = investigate_topics(model, code_book, true_labels, true_videos, probability_of_words, _lambda, n_top_words=30)

    print "\nvideos classified:", len(true_labels), len(pred_labels)

    # ****************************************************************************************************
    # Get results
    # ****************************************************************************************************
    n, l, v, h, c, mi, nmi, a, mat, labs = vis.print_results(true_labels, pred_labels, n_topics)

    # ****************************************************************************************************
    # Write out results
    # ****************************************************************************************************
    name = '/TopicData/results.txt'
    f1 = open(directory+name, 'w')
    f1.write('n_topics: %s \n' % n_topics)
    f1.write('n_iters: %s \n' % n_iters)
    f1.write('dirichlet_params: (%s, %s) \n' % (alpha, eta))
    f1.write('class_thresh: %s \n' % class_thresh)
    f1.write('code book length: %s \n' % len(code_book))
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
    for i, words in relevant_words.items():
        f1.write('Topic %s : %s \n' % (i, words[:10]))
    f1.close()

    # ****************************************************************************************************
    # Write out confusion matrix as array and dictionary
    # ****************************************************************************************************
    confusion_dic    = {}
    for row, lab in zip(mat, labs):
        confusion_dic[lab] = row

    name = '/TopicData/confusion_mat.p'
    f1 = open(directory+name, 'w')
    pickle.dump(mat, f1, 2)
    f1.close()

    name = '/TopicData/confusion_dic.p'
    f1 = open(directory+name, 'w')
    pickle.dump(confusion_dic, f1, 2)
    f1.close()


    # Draw pictures
    # for i, j in itertools.combinations(xrange(11), 2):
    #     print "Topics: %s and %s" %(i, j)
    #     compare_genome(model.topic_word_[i], model.topic_word_[j])
    # for i in xrange(11):
    #     vis.genome(model.topic_word_[i], i)

    return model.topic_word_

def investigate_topics(model, code_book, labels, videos, prob_of_words, _lambda, n_top_words = 30):
    """investigate the learned topics
    Relevance defined: http://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf
    """

    topic_word = model.topic_word_
    doc_topic = model.doc_topic_
    # code_book, graphlets, uuids, miss_labels = loaded_data
    # print "1"
    # import pdb; pdb.set_trace()

    true_labels = labels
    vocab = [hash for hash in list(code_book)]

    # ****************************************************************************************************
    # Relevance
    # ****************************************************************************************************
    # names_list = [i.lower() for i in ['Alan','Alex','Andy','Amy','Michael','Ben','Bruno','Chris','Colin','Collin','Ellie','Daniel','Dave','Eris','Emma','Helen','Holly','Jay','the_cleaner',
            # 'Jo','Luke','Mark','Louis','Laura', 'Kat','Matt','Nick','Lucy','Rebecca','Jennifer','Ollie','Rob','Ryan','Rachel','Sarah','Stefan','Susan']]

    relevant_words = {}
    for i, phi_kw in enumerate(topic_word):

        phi_kw = threshold(np.asarray(phi_kw), 0.00001)
        log_ttd = [_lambda*math.log(y) if y!=0 else 0  for y in phi_kw]
        log_lift = [(1-_lambda)*math.log(y) if y!=0 else 0 for y in phi_kw / prob_of_words]
        relevance = np.add(log_ttd, log_lift)

        # cnt = 0
        # import pdb; pdb.set_trace()
        # for h, g in zip(np.asarray(vocab)[relevance >2.1], graphs[relevance >2.1]):
        #     o, s, t = object_nodes(g)
        #     if "hand" in o and "object_14" in o and len(s) == 2:
        #         print h, s, t
        #         cnt+=1
        # print cnt
        # vis.genome_rel(relevance, i)

        inds = np.argsort(relevance)[::-1]
        # top_relevant_words_in_topic = np.array(vocab)[inds] #[:-(n_top_words+1):-1]
        # pdb.set_trace()
        relevant_language_words_in_topic = []

        for ind in inds:
            word = vocab[ind]

            #todo: somehting is wrong here.
            if relevance[ind] <= 1.0 and word.isalpha() and word not in names_list:
                relevant_language_words_in_topic.append(word)
                # pdb.set_trace()
        relevant_words[i] = relevant_language_words_in_topic[:10]

    # print("\ntype(topic_word): {}".format(type(topic_word)))
    # print("shape: {}".format(topic_word.shape))
    # print "objects in each topic: "
    topics = {}
    for i, topic_dist in enumerate(topic_word):
        objs = []
        top_words_in_topic = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]

        #print('Topic {}: {}'.format(i, ' '.join( [repr(i) for i in top_words_in_topic] )))
        # for j in [graphlets[k] for k in top_words_in_topic]:
        #     objs.extend(object_nodes(j)[0])
        topics[i] = objs
        # print('Topic {}: {}'.format(i, list(set(objs))))
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
    # print "2"
    # import pdb; pdb.set_trace()

    return true_labels, pred_labels, videos, relevant_words


def compare_two_topic_models((topic_word1, code_book1), \
                            (topic_word2, code_book2), title=""):

    #Adjust the columns on two topic_word matrices, and use hungarian alg to find best matches.

    print "shape of both codebooks:", len(code_book1), len(code_book2)
    f1, f2, global_codebook = lsa.create_feature_spaces_over_codebooks(topic_word1, code_book1, topic_word2, code_book2)

    cosine_matrix = cosine_similarity(f1, f2)
    print "cosine matrix shape:", cosine_matrix.shape
    # returns: rows: n_samples_X by columns: n_samples_Y.
    cosine_matrix[cosine_matrix < 0.001] = 0  # All low values set to 0
    for i in xrange(cosine_matrix.shape[0]):
        print list(cosine_matrix[i])
        # for j in list(cosine_matrix[i]):
        #     print type(j), j, "%0.2f" % j

    vis.plot_topic_similarity_matrix(cosine_matrix, title)

    reordering = np.argmax(cosine_matrix, axis=1)
    reordered = cosine_matrix[:,reordering]
    hungarian_out = hungarian.lap( (np.ones(reordered.shape) - reordered))

    matches = []
    print "hungarian alg say: "
    for i, j in zip(hungarian_out[0], hungarian_out[1]):
        print "row %s, column %s match with score: %s" % (i, j, reordered[i][j])
        matches.append(reordered[i][j])
    average_sim = sum(matches) / float(len(matches))
    print "avg simi: %s" % average_sim

    if raw_input("press y for genome graph comparisons of the learned topics...")== "y":
        for i in xrange(11):
                vis.genome(f1[i], f2[reordering[i]])



if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "Usage: please provide a QSR run folder number."
        sys.exit(1)
    else:
        run = sys.argv[1]

    # ****************************************************************************************************
    # script parameters
    # ****************************************************************************************************
    directory = "/home/"+getpass.getuser()+"/Datasets/ECAI_Data/dataset_segmented_15_12_16"
    create_graphlet_images = False

    # ****************************************************************************************************
    # Load Term Frequency
    # ****************************************************************************************************
    term_freq, online_data, code_book, graphlets = fr.load_term_frequency(directory, run)
    wordids, wordcts, true_labels, video_uuids = online_data

    # # ****************************************************************************************************
    # # Segmented Term Freq
    # # ****************************************************************************************************
    low_pass_instance = 5
    term_freq_seg_red, code_book_seg_red, graphlets_red =  fr.high_instance_code_words(term_freq, code_book, graphlets, low_pass_instance)


    # ****************************************************************************************************
    # Concatenated Term-Freq Matrix
    # ****************************************************************************************************
    term_freq_concat_dic = {}
    for i, uuid in enumerate(video_uuids):
        if uuid not in term_freq_concat_dic:
            term_freq_concat_dic[uuid] = term_freq[i]
        else:
            term_freq_concat_dic[uuid] += term_freq[i]

    concat_uuids = [uuid for uuid in term_freq_concat_dic.keys()]
    term_freq_concat = np.vstack([hist for hist in term_freq_concat_dic.values()])

    term_freq_concat_red, code_book_concat_red, graphlets_red =  fr.high_instance_code_words(term_freq_concat, code_book, graphlets, low_pass_instance)


    # # ****************************************************************************************************
    # # Supervised SVM
    # # ****************************************************************************************************
    labels = online_data[2]
    # supervised.run_svm(term_freq_red, labels)
    # supervised.kmeans_clustering(term_freq_red, labels, threshold=0.01)
    #
    # # ****************************************************************************************************
    # # call batch LSA
    # # ****************************************************************************************************
    print "\nLSA: "

    lsa_Vt1 = lsa.svd_clusters(term_freq_seg_red, labels)
    lsa_Vt2 = lsa.svd_clusters(term_freq_concat_red, labels)

    title = "Cosine Similarity Between Two LSA Models"
    lsa.compare_two_LSA_models(lsa_Vt1, code_book_seg_red, lsa_Vt2, code_book_concat_red, title)

    # print "\Random clustering: "
    # lsa.svd_clusters(term_freq_red, labels, n_comps=10, random=True)
    # print "LSA done"
    #
    # import pdb; pdb.set_trace()

    # ****************************************************************************************************
    # call batch LDA
    # ****************************************************************************************************
    print "\nLDA: "
    n_iters = 1000
    n_topics = 11
    alpha, eta = 0.5, 0.03
    class_thresh = 0.3
    _lambda = 0.5 # relevance scale

    directory = os.path.join(directory, "QSR_path/run_%s" % run)

    phi_2 = just_topic_model(term_freq_concat_red, n_iters, n_topics, (alpha, eta))
    print "\nConcatenated LDA done"

    phi_1 = run_topic_model(term_freq_seg_red, code_book_seg_red, graphlets_red, online_data, directory, n_iters, n_topics, create_graphlet_images, (alpha, eta), class_thresh, _lambda)
    print "\nSegmented LDA done"

    title = "Cosine Similarity Between Two Topic Models"
    compare_two_topic_models((phi_1, code_book_seg_red), (phi_2, code_book_concat_red), title)
