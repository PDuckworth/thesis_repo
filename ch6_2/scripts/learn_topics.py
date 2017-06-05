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
from sklearn import metrics
import pdb
import lda
import pyLDAvis
import ch6_2.onlineldavb as OLDA
import ch6_2.visualisations as vis

def recreate_data_with_high_instance_graphlets(term_frequency, code_book, graphlets, low_instance):
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
    set_of_true_labs = list(set(true_labels))
    for cnt, i in enumerate(set_of_true_labs):
        row_inds[i] = cnt
    print "row inds",  row_inds

    res = {}
    mat = np.zeros( (len(set_of_true_labs), num_clusters))

    for i, j in zip(true_labels, pred_labels):
        row_ind = row_inds[i]
        mat[row_ind][j] +=1
        try:
            res[i].append(j)
        except:
            res[i] = [j]

    for cnt, i in enumerate(mat):
        print "label: %s: %s" % (set_of_true_labs[cnt], i)

    # pdb.set_trace()

    # Find assignments:
    for true, preds in res.items():
        print true, max(set(preds), key=preds.count)

    norm_mat = []
    for i in mat:
        norm_mat.append([float(j)/float( sum(i, 0) ) for j in i ])

    f, ax = plt.subplots(nrows=1, ncols=1)
    f.suptitle("Topic Confusion Marix: using segmented clips")
    plt.xlabel('Learnt Topics')
    plt.ylabel('Ground Truth Labels')
    plt.setp((ax), xticks=range(len(mat[0])), yticks=xrange(len(set_of_true_labs)))
    plt.yticks(xrange(len(set_of_true_labs)), set_of_true_labs, color='red')
    ax.imshow(mat, interpolation='nearest')

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

def run_topic_model(term_freq, code_book, graphlets, online_data, directory, n_iters, n_topics, create_graphlet_image, (alpha, eta), class_thresh=0, _lambda=0.5):

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

    # ****************************************************************************************************
    # investigate the relevant words in each topic, and see which documents are classified into each topic
    # ****************************************************************************************************
    true_labels, pred_labels, videos, relevant_words = investigate_topics(model, loaded_data, true_labels, true_videos, probability_of_words, _lambda, n_top_words=30)

    invesitgate_videos_given_topics(pred_labels, true_videos)
    print "\nvideos classified:", len(true_labels), len(pred_labels)

    # ****************************************************************************************************
    # Get results
    # ****************************************************************************************************
    n, l, v, h, c, mi, nmi, a, mat, labs = print_results(true_labels, pred_labels, n_topics)

    # ****************************************************************************************************
    # Write out results
    # ****************************************************************************************************
    name = '/TopicData/results.txt'
    f1 = open(directory+name, 'w')
    f1.write('n_topics: %s \n' % n_topics)
    f1.write('n_iters: %s \n' % n_iters)
    f1.write('dirichlet_params: (%s, %s) \n' % (alpha, eta))
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

def investigate_topics(model, loaded_data, labels, videos, prob_of_words, _lambda, n_top_words = 30):
    """investigate the learned topics
    Relevance defined: http://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf
    """

    topic_word = model.topic_word_
    doc_topic = model.doc_topic_
    code_book, graphlets, uuids, miss_labels = loaded_data
    print "1"
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
    print "objects in each topic: "
    topics = {}
    for i, topic_dist in enumerate(topic_word):
        objs = []
        top_words_in_topic = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]

        #print('Topic {}: {}'.format(i, ' '.join( [repr(i) for i in top_words_in_topic] )))
        # for j in [graphlets[k] for k in top_words_in_topic]:
        #     objs.extend(object_nodes(j)[0])
        topics[i] = objs
        print('Topic {}: {}'.format(i, list(set(objs))))
        print top_words_in_topic

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
    print "2"
    # import pdb; pdb.set_trace()

    return true_labels, pred_labels, videos, relevant_words

def invesitgate_videos_given_topics(pred_labels, video_list):
    print "\nvideos assigned to each topic: "

    print "3"
    # import pdb; pdb.set_trace()

    topics_to_videos = {}
    num_of_topics = max(pred_labels)+1
    for topic_id in xrange(num_of_topics):
        topics_to_videos[topic_id] = []
        for label, video in zip(pred_labels, video_list):
            if topic_id == label:
                topics_to_videos[topic_id].append(video.replace("vid", ""))
        print "\ntopic: %s:  %s" % (topic_id, topics_to_videos[topic_id])

    print "4"
    # import pdb; pdb.set_trace()
    return


def online_lda_model(loaded_data, online_data, directory, K, D, (alpha,eta), tau0, kappa, batchsize, class_thresh):

    code_book, graphlets, uuids, true_labels = loaded_data
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

    n, l, v, h, c, mi, nmi, a, mat, labs = print_results(true_labels, pred_labels, K)
    # ****************************************************************************************************
    # do a final E-step on all the data :)
    # ****************************************************************************************************
    rhot = pow(olda._tau0 + olda._updatect, -olda._kappa)
    olda._rhot = rhot
    (gamma, sstats) = olda.do_e_step(wordids, wordcts)

    import pdb; pdb.set_trace()

    y_true, y_pred = [], []
    for n, gam in enumerate(gamma):
        gam = gam / float(sum(gam))

        if max(gam) > class_thresh:
            # thresholded_true.append(ordered_labels[n])
            y_true.append(sorted_labels[n])
            y_pred.append(np.argmax(gam))

    n, l, v, h, c, mi, nmi, a, mat, labs = print_results(y_true, y_pred, K)

    # ****************************************************************************************************
    # Write out a file of results
    # ****************************************************************************************************
    name = '/oldaData/results_vb.txt'
    f1 = open(directory+name, 'w')
    f1.write('n_topics: %s \n' % K)
    f1.write('dirichlet_params: (%s, %s) \n' % (alpha, eta))
    f1.write('class_thresh: %s \n' % class_thresh)
    f1.write('code book length: %s \n' % codebook_lengh)
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


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "Usage: please provide a QSR run folder number."
        sys.exit(1)
    else:
        run = sys.argv[1]

    # ****************************************************************************************************
    # script parameters
    # ****************************************************************************************************
    create_graphlet_images = False

    # ****************************************************************************************************
    # load word counts
    # ****************************************************************************************************
    uuids, wordids, wordcts = [], [], []
    directory = '/home/'+getpass.getuser()+'/Datasets/ECAI_Data/dataset_segmented_15_12_16/QSR_path/run_%s' % run
    if not os.path.exists(os.path.join(directory, 'TopicData')): os.makedirs(os.path.join(directory, 'TopicData'))
    if not os.path.exists(os.path.join(directory, 'oldaData')): os.makedirs(os.path.join(directory, 'oldaData'))

    print "directory: ", directory
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

    # ****************************************************************************************************
    # call batch LDA
    # ****************************************************************************************************
    n_iters = 1000
    n_topics = 11
    alpha, eta = 0.5, 0.03
    class_thresh = 0.5
    low_pass_instance = 5
    _lambda = 0.5 # relevance scale

    term_freq, code_book, graphlets =  recreate_data_with_high_instance_graphlets(term_freq, code_book, graphlets, low_pass_instance)

    run_topic_model(term_freq, code_book, graphlets, online_data, directory, n_iters, n_topics, create_graphlet_images, (alpha, eta), class_thresh, _lambda)


    # ****************************************************************************************************
    # run online LDA
    # ****************************************************************************************************

    K = 11
    D = 500
    alpha, eta = 0.1, 0.03
    tau0 = 10
    kappa = 0.7
    batchsize = 10

    # online_lda_model(loaded_data, online_data, directory, K, D, (alpha,eta), tau0,kappa,batchsize, class_thresh)
