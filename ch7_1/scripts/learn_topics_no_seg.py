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
from scipy.stats import threshold
import pdb
import lda
import pyLDAvis
import ch6_2.onlineldavb as OLDA
import ch6_2.visualisations as vis
import ch6_2.create_term_freq as fr
import ch6_2.LSA as lsa
import ch6_2.svm as supervised

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

    invesitgate_videos_given_topics(pred_labels, true_videos)
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

    return

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

def invesitgate_videos_given_topics(pred_labels, video_list):
    # print "\nvideos assigned to each topic: "

    # print "3"
    # import pdb; pdb.set_trace()

    topics_to_videos = {}
    num_of_topics = max(pred_labels)+1
    for topic_id in xrange(num_of_topics):
        topics_to_videos[topic_id] = []
        for label, video in zip(pred_labels, video_list):
            if topic_id == label:
                topics_to_videos[topic_id].append(video.replace("vid", ""))
        # print "\ntopic: %s:  %s" % (topic_id, topics_to_videos[topic_id])

    # print "4"
    # import pdb; pdb.set_trace()
    return


def online_lda_model(code_book, graphlets, online_data, directory, K, D, (alpha,eta), tau0, kappa, batchsize, class_thresh):

    wordids, wordcts, sorted_labels, true_videos = online_data

    num_iters = int(len(wordids)/float(batchsize))

    true_labels, pred_labels = [], []
    olda = OLDA.OnlineLDA(code_book, K, D, alpha, eta, tau0, kappa, 0)

    # import random
    # a = ordered_labels
    # b = ordered_vid_names
    # c = list(zip(a, b))
    # random.shuffle(c)
    # ordered_labels, ordered_vid_names = zip(*c)

    for iteration in range(0, num_iters):
        # iteration = num_iters - iteration

        # print "it: %s. " %iteration #start: %s. end: %s" % (iteration, iteration*batchsize, (iteration+1)*batchsize)
        ids = wordids[iteration*batchsize:(iteration+1)*batchsize]
        cts = wordcts[iteration*batchsize:(iteration+1)*batchsize]

        labels = sorted_labels[iteration*batchsize:(iteration+1)*batchsize]
        vids = true_videos[iteration*batchsize:(iteration+1)*batchsize]

        (gamma, bound) = olda.update_lambda(ids, cts, dbg=False)
        print ">>", bound, len(wordids), D, sum(map(sum, cts)), olda._rhot

        thresholded_true, thresholded_pred = [], []
        for n, gam in enumerate(gamma):
            gam = gam / float(sum(gam))

            if max(gam) > class_thresh:
                # thresholded_true.append(ordered_labels[n])
                thresholded_true.append(labels[n])
                thresholded_pred.append(np.argmax(gam))

        # true_labels.extend(labels)
        # pred_labels.extend([np.argmax(i) for i in gamma])
        true_labels.extend(thresholded_true)
        pred_labels.extend(thresholded_pred)

        word_counter = 0
        for i in cts:
            word_counter+=sum(i)

        perwordbound = bound * len(wordids) / (D * word_counter)

        print 'iter: %s:  rho_t = %f,  held-out per-word perplexity estimate = %f. LDA - Done\n' % \
            (iteration, olda._rhot, np.exp(-perwordbound))

        if (iteration % 1 == 0):
            # import pdb; pdb.set_trace()
            np.savetxt(directory + '/oldaData/lambda-%d.dat' % iteration, olda._lambda)
            np.savetxt(directory + '/oldaData/gamma-%d.dat' % iteration, gamma)

            np.savetxt(directory + '/oldaData/vids-%d.dat' % iteration, vids, delimiter=" ", fmt="%s")
            np.savetxt(directory + '/oldaData/labels-%d.dat' % iteration, labels, delimiter=" ", fmt="%s")

    n, l, v, h, c, mi, nmi, a, mat, labs = vis.print_results(true_labels, pred_labels, K)
    # ****************************************************************************************************
    # do a final E-step on all the data :)
    # ****************************************************************************************************
    rhot = pow(olda._tau0 + olda._updatect, -olda._kappa)
    olda._rhot = rhot
    (gamma, sstats) = olda.do_e_step(wordids, wordcts)

    # import pdb; pdb.set_trace()
    y_true, y_pred = [], []
    for n, gam in enumerate(gamma):
        gam = gam / float(sum(gam))

        if max(gam) > class_thresh:
            # thresholded_true.append(ordered_labels[n])
            y_true.append(sorted_labels[n])
            y_pred.append(np.argmax(gam))

    n, l, v, h, c, mi, nmi, a, mat, labs = vis.print_results(y_true, y_pred, K)

    # ****************************************************************************************************
    # Write out a file of results
    # ****************************************************************************************************
    name = '/oldaData/results_vb.txt'
    f1 = open(directory+name, 'w')
    f1.write('n_topics: %s \n' % K)
    f1.write('dirichlet_params: (%s, %s) \n' % (alpha, eta))
    f1.write('class_thresh: %s \n' % class_thresh)
    f1.write('code book length: %s \n' % len(code_book))
    # f1.write('sum of all words: %s \n' % sum_of_all_words)
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
    # f1.write('relevant_words: \n')
    # for i, words in relevant_words[percentage].items():
    #     f1.write('Topic %s : %s \n' % (i, words[:10]))
    f1.close()



def load_term_frequency(directory, run):
    """Slightly different to the one in create_term_freq.py
    This loops over files, not vid IDs.
    """

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
    video_uuids, true_labels = [], []
    num_of_vids = len(os.listdir(directory))

    for task in sorted(os.listdir(directory)):
        if task[0].isalpha(): continue
        d_video = os.path.join(directory, task)
        if not os.path.isfile(d_video): continue
        with open(d_video, 'r') as f:
            try:
                (uuid, label, ids, histogram) = pickle.load(f)
            except:
                "failed to load properly: \n %s" % (task)

        wordids.append(ids)
        wordcts.append(histogram)
        # true_videos.append(video)
        video_uuids.append(uuid)
        true_labels.append(label)

    print "#videos: ", len(wordids), len(wordcts)
    online_data = wordids, wordcts, true_labels, video_uuids

    # ****************************************************************************************************
    # Term-Freq Matrix
    # ****************************************************************************************************
    term_freq = fr.term_frequency_mat(codebook_lengh, wordids, wordcts)

    return term_freq, online_data, code_book, graphlets


def nonsegmented_evaluation(doc_topic, true_labels, threshold):

    label_dict = {}
    all_labels = []
    for cnt, lab in enumerate(true_labels):
        #if "print" in lab:
        #    lab = "printing"
        temp = lab.split(":")
        label_dict[cnt] = temp
        all_labels.extend(temp)

    unique_labels = sorted(list(set(all_labels)))
    y_axis_labels = []
    label_counts = {}

    unique_labels = ['N/A', 'take_paper_towel', 'use_kettle', 'washing_up', 'printing_interface',  'printing_take_printout', 'take_tea_bag',  'throw_trash',   'take_from_fridge', 'use_water_cooler', 'microwaving_food']
    #unique_labels = ['take_paper_towel', 'use_kettle', 'washing_up', 'printing', 'take_tea_bag',  'throw_trash',   'take_from_fridge', 'use_water_cooler', 'microwaving_food']
    for label in unique_labels:
        for cnt, multi_lab in label_dict.items():
            if label in multi_lab:

                try:
                    label_counts[label] +=1
                except:
                    label_counts[label] =1

    # for i, j in label_counts.items():
    #     print i, j
    print "\n", label_counts

    print sum(label_counts.values()), doc_topic[0].shape

    eval_mat1 = np.zeros((len(unique_labels), doc_topic[0].shape[0]) )
    eval_mat2 = np.zeros((len(unique_labels), doc_topic[0].shape[0]) )

    np.set_printoptions(precision=2)
    counter = 0
    for row_n, label in enumerate(unique_labels):
        print "\n>>", label

        y_axis_labels.append(label)
        for ind in xrange(len(label_dict.keys())):
            # print ind, label_dict[ind]

            # if len(label_dict[ind]) > 1: continue  # THIS REDUCES THE ANALYSIS TO ONLY ONE SEGMENT VIDEOS

            if label in label_dict[ind]:

                # print "<", ind, gt_labs
                doc = doc_topic[ind]
                temp1 = [i if i > 0.0 else 0 for i in doc]
                temp2 = [i if i > threshold else 0 for i in doc]
                # print "?", doc, len(doc), doc.shape
                print ">", ["%0.3f" % i if i > 0.4 else "-.---" for i in doc]
                # print "row:", eval_mat[row_n], eval_mat[row_n].shape
                eval_mat1[row_n] += temp1
                eval_mat2[row_n] += temp2
                counter+=1

    print "==", counter
    norm_row1, norm_row2 = [], []

    for cnt, row in enumerate(eval_mat1):
        norm_row1.append([ float(j)/float(sum(row)) if sum(row) !=0 else 0 for j in row])
        print ">", sum(row), y_axis_labels[cnt], label_counts[y_axis_labels[cnt]], sum([float(j)/float( sum(row)) if sum(row) !=0 else 0 for j in row]), row

    for cnt, row in enumerate(eval_mat2):
        norm_row2.append([ float(j)/float(sum(row, 0)) if sum(row, 0) !=0 else 0 for j in row ])



    for cnt, mat in enumerate([eval_mat1, norm_row1, eval_mat2, norm_row2]):

        f, ax = plt.subplots(nrows=1, ncols=1)
        plt.title("Correlation between Ground Truth Labels \nand Emergent Topics C 1 C 2", fontsize=16)
        plt.xlabel('Learned Topics', fontsize=16)
        plt.ylabel('Ground Truth Labels', fontsize=16)

        alphas = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
        plt.xticks(range(len(alphas)), alphas, fontsize=16)
        plt.yticks(range(len(y_axis_labels)), y_axis_labels, fontsize=14, color='red')

        cmap=plt.cm.Blues
        plt.imshow(mat, interpolation='nearest', cmap=cmap)

        plt.tight_layout()
        plt.colorbar()
    plt.show()


    fig = plt.figure()
    cmap=plt.cm.Blues
    # plt.tight_layout()

    ax1 = fig.add_subplot(221)
    plt.yticks(range(len(y_axis_labels)), y_axis_labels, fontsize=14)
    plt.imshow(eval_mat1, interpolation='none', cmap=cmap)

    ax2 = fig.add_subplot(222)
    plt.imshow(norm_row1, interpolation='none', cmap=cmap)

    ax3 = fig.add_subplot(223)
    plt.yticks(range(len(y_axis_labels)), y_axis_labels, fontsize=14)
    plt.imshow(eval_mat2,  interpolation='none', cmap=cmap)

    ax4 = fig.add_subplot(224)
    alphas = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    plt.xticks(range(len(eval_mat2)), alphas, fontsize=14)

    plt.imshow(norm_row2,  interpolation='none', cmap=cmap)

    plt.colorbar()
    plt.show()
    return


def just_topic_model(term_freq,   n_iters, n_topics, (alpha, eta)):
    X = term_freq
    print "sum of all data: X.shape: %s and X.sum: %s" % (X.shape, X.sum())

    model = lda.LDA(n_topics=n_topics, n_iter=n_iters, random_state=1, alpha=alpha, eta=eta)
    model.fit(X)

    feature_freq = (X != 0).sum(axis=0)
    doc_lengths = (X != 0).sum(axis=1)

    # return (doc_topic, topic_word, code_book, pred_labels, true_labels)
    return model.doc_topic_, model.topic_word_



if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "Usage: please provide a QSR run folder number."
        sys.exit(1)
    else:
        run = sys.argv[1]

    # ****************************************************************************************************
    # script parameters
    # ****************************************************************************************************
    directory = "/home/"+getpass.getuser()+"/Datasets/ECAI_Data/dataset_not_segmented"

    # ****************************************************************************************************
    # Term Frequency
    # ****************************************************************************************************
    low_pass_instance = 15
    term_freq, online_data, code_book, graphlets = load_term_frequency(directory, run)
    term_freq_red, code_book_red, graphlets_red =  fr.high_instance_code_words(term_freq, code_book, graphlets, low_pass_instance)

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
    # print "\nLSA: "
    # lsa.svd_clusters(term_freq_red, labels)

    # print "\Random clustering: "
    # lsa.svd_clusters(term_freq_red, labels, n_comps=10, random=True)
    # print "LSA done"
    #

    # ****************************************************************************************************
    # call batch LDA
    # ****************************************************************************************************
    print "\nLDA: "
    n_iters = 500
    n_topics = 11
    alpha, eta = 0.8, 0.03
    class_thresh = 0.5
    _lambda = 0.5 # relevance scale

    directory = os.path.join(directory, "QSR_path/run_%s" % run)

    doc_topic, topic_word = just_topic_model(term_freq_red, n_iters, n_topics, (alpha, eta))

    nonsegmented_evaluation(doc_topic, labels, class_thresh)

    print "LDA done"
