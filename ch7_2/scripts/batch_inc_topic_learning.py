#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import math
import cPickle as pickle
import numpy as np
import getpass
import itertools
import copy
import lda
from scipy.stats import threshold
from sklearn import metrics
import matplotlib.pyplot as plt
import ch6_2.onlineldavb as OLDA
import ch6_2.visualisations as vis
import ch6_2.ECAI_videos_segmented_by_day as vids #  segmented_videos, random_segmented_videos

def term_frequency_mat(codebook_length, wordids, wordcnts):
    term_freq = []
    for counter, (ids, cnts) in enumerate(zip(wordids, wordcnts)):
        # print ids, cnts
        vec = np.array([0] * codebook_length)
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

    set_of_true_labs = list(set(true_labels))

    (h, c, v) =  metrics.homogeneity_completeness_v_measure(true_labels, pred_labels)
    print "#Topics=%s (%s). v-measure: %0.3f. homo: %0.3f. comp: %0.3f. MI: %0.3f. NMI: %0.3f." \
      % (num_clusters, len(pred_labels), v, h, c,
        metrics.mutual_info_score(true_labels, pred_labels),
        metrics.normalized_mutual_info_score(true_labels, pred_labels) )

    return    (num_clusters, len(pred_labels),
           metrics.v_measure_score(true_labels, pred_labels),
           metrics.homogeneity_score(true_labels, pred_labels),
           metrics.completeness_score(true_labels, pred_labels),
           metrics.mutual_info_score(true_labels, pred_labels),
           metrics.normalized_mutual_info_score(true_labels, pred_labels),
           set_of_true_labs)

def confusion_mat(true_labels, pred_labels, num_clusters):

    row_inds = {}
    set_of_true_labs = list(set(true_labels))
    for cnt, i in enumerate(set_of_true_labs):
        row_inds[i] = cnt
    # print "row inds",  row_indsnew code book length

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

    # Find assignments:
    # for true, preds in res.items():
    #     print true, max(set(preds), key=preds.count)

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

    return mat

def run_topic_model(X, n_iters, n_topics, (alpha, eta)):

    print "sum of all data: X.shape: %s and X.sum: %s" % (X.shape, X.sum())
    feature_freq = (X != 0).sum(axis=0)
    doc_lengths = (X != 0).sum(axis=1)

    model = lda.LDA(n_topics=n_topics, n_iter=n_iters, random_state=1, alpha=alpha, eta=eta)
    model.fit(X)

    print "phi: %s. theta: %s. nd: %s. Mw: %s" \
        %( model.topic_word_.shape, model.doc_topic_.shape, doc_lengths.shape, len(feature_freq))

    # vis_data = pyLDAvis.prepare(model.topic_word_, model.doc_topic_, doc_lengths, graphlets.keys(), feature_freq)
    # html_file = "/home/"+getpass.getuser()+"/Dropbox/Programming/topics_to_language/topic_model_ecai.html"
    # pyLDAvis.save_html(vis_data, html_file)
    # print "PyLDAVis ran. output: %s" % html_file
    return model

def investigate_topics(model, loaded_data, labels, videos, prob_of_words, language_indices, _lambda, n_top_words = 30):
    """investigate the learned topics
    Relevance defined: http://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf
    """

    topic_word = model.topic_word_
    doc_topic = model.doc_topic_
    code_book, graphlets_, uuids, miss_labels = loaded_data
    print "1"

    true_labels = labels
    vocab = [hash for hash in list(code_book)]
    graphs = loaded_data[1]
    # ****************************************************************************************************
    # Relevance
    # ****************************************************************************************************
    names_list = [i.lower() for i in ['Alan','Alex','Andy','Amy','Michael','Ben','Bruno','Chris','Colin','Collin','Ellie','Daniel','Dave','Eris','Emma','Helen','Holly','Jay','the_cleaner','Jo','Luke','Mark','Louis','Laura', 'Kat','Matt','Nick','Lucy','Rebecca','Jennifer','Ollie','Rob','Ryan','Rachel','Sarah','Stefan','Susan']]

    relevant_words = {}
    for i, phi_kw in enumerate(topic_word):

        phi_kw = threshold(np.asarray(phi_kw), 0.00001)
        log_ttd = [_lambda*math.log(y) if y!=0 else 0  for y in phi_kw]
        log_lift = [(1-_lambda)*math.log(y) if y!=0 else 0 for y in phi_kw / probability_of_words]
        relevance = np.add(log_ttd, log_lift)

        # cnt = 0
        # for h, g in zip(np.asarray(vocab)[relevance >2.1], graphs[relevance >2.1]):
        #     o, s, t = object_nodes(g)
        #     if "hand" in o and "object_14" in o and len(s) == 2:
        #         print h, s, t
        #         cnt+=1
        # print cnt
        # genome_rel(relevance, i)

        inds = np.argsort(relevance)[::-1]
        # top_relevant_words_in_topic = np.array(vocab)[inds] #[:-(n_top_words+1):-1]
        relevant_language_words_in_topic = []

        for ind in inds:
            word = vocab[ind]

            #todo: somehting is wrong here.
            if relevance[ind] <= 1.0 and word.isalpha() and word not in names_list:
                relevant_language_words_in_topic.append(word)

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
    import pdb; pdb.set_trace()

    return true_labels, pred_labels, videos, relevant_words

def invesitgate_videos_given_topics(pred_labels, video_list):
    print "\nvideos assigned to each topic: "

    print "3"
    import pdb; pdb.set_trace()

    topics_to_videos = {}
    num_of_topics = max(pred_labels)+1
    for topic_id in xrange(num_of_topics):
        topics_to_videos[topic_id] = []
        for label, video in zip(pred_labels, video_list):
            if topic_id == label:
                topics_to_videos[topic_id].append(video.replace("vid", ""))
        print "\ntopic: %s:  %s" % (topic_id, topics_to_videos[topic_id])

    print "4"
    import pdb; pdb.set_trace()
    return

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "Usage: please provide a QSR run folder number."
        sys.exit(1)
    else:
        run = sys.argv[1]

    # ****************************************************************************************************
    # parameters
    # ****************************************************************************************************
    n_iters = 200
    n_topics = 11
    # create_graphlet_images = False
    dirichlet_params = (0.5, 0.01)
    # class_thresh = 0.3
    # _lambda = 0.5
    # _lambda = 0
    # using_language = False
    assign_class_thresh = 0.0

    ### load file of dates vs clips
    videos_by_day = vids.segmented_videos()            # actually per day
    # videos_by_day = vids.random_segmented_videos()   # random

    # ****************************************************************************************************
    # load qsr videos
    # ****************************************************************************************************
    directory = '/home/'+getpass.getuser()+'/Datasets/ECAI_Data/dataset_segmented_15_12_16/QSR_path/run_%s' % run
    # if not os.path.exists(os.path.join(directory, 'inc_results')): os.makedirs(os.path.join(directory, 'inc_results'))
    print "directory: ", directory

    num_of_batch_vids = sum([len(j) if i in ['2016-04-05'] else 0 for i, j in videos_by_day.items()])
    print "#Videos = ", num_of_batch_vids

    codebook_length = 0
    wordids_b, wordcts_b, video_names_b, true_labels_b  = [], [], [], []
    for task in videos_by_day['2016-04-05']:
        video = "%s.p" % task
        d_video = os.path.join(directory, video)
        if not os.path.isfile(d_video): continue
        with open(d_video, 'r') as f:
            try:
                (uuid, label, ids, histogram) = pickle.load(f)
            except:
                "failed to load properly: \n %s" % (video)


        wordids_b.append(ids)
        wordcts_b.append(histogram)
        video_names_b.append(video)
        true_labels_b.append(label)

        if max(ids) > codebook_length:
            codebook_length = max(ids)

    codebook_length+=1
    print "codebook length: ", codebook_length

    # ****************************************************************************************************
    # Batch LDA
    # ****************************************************************************************************
    term_freq = term_frequency_mat(codebook_length, wordids_b, wordcts_b)
    model =  run_topic_model(term_freq, n_iters, n_topics, dirichlet_params)

    ### Results on batch day one :)
    videos, thresholded_true, pred_labels = [], [], []
    for n, gam in enumerate(model.doc_topic_):
        if max(gam) > assign_class_thresh:
            thresholded_true.append(true_labels_b[n])
            pred_labels.append(np.argmax(gam))
            videos.append(video_names_b[n])

    n, l, v, h, c, mi, nmi, labs = print_results(thresholded_true, pred_labels, n_topics)

    # ****************************************************************************************************
    # Get Incremental videos
    # ****************************************************************************************************
    new_codebook_length = codebook_length
    wordids, wordcts, video_names, true_labels = [], [], [], []
    per_video_codebook_length = []

    counter = 0
    minibatch_by_day = []
    for date in sorted(videos_by_day.keys()):
        if date == '2016-04-05': continue   # exclude the first day
        minibatch_by_day.append(counter)

        for task in videos_by_day[date]:
            # if '196' in task or '211' in task: continue

            video = "%s.p" % task
            d_video = os.path.join(directory, video)
            if not os.path.isfile(d_video): continue
            with open(d_video, 'r') as f:
                try:
                    (uuid, label, ids, histogram) = pickle.load(f)
                except:
                    "failed to load properly: \n %s" % (video)

            if sum(ids) != 0:
                new_codebook_length = max( max(ids), new_codebook_length)
                per_video_codebook_length.append(new_codebook_length)

            else:
                print "no QSR code words in %s" % video
                per_video_codebook_length.append(new_codebook_length)

            wordids.append(ids)
            wordcts.append(histogram)
            video_names.append(video)
            true_labels.append(label)
            counter+=1

    minibatch_by_day.append(counter)
    new_codebook_length+=1
    print "codebook length: ", new_codebook_length

    # ****************************************************************************************************
    # Initiate a VB LDA model
    # ****************************************************************************************************
    K = n_topics
    D = 400
    alpha, eta = 0.1, 0.01
    tau0 = 1
    kappa = 0.7
    updatect = 0

    #batch codebook size
    vocab = xrange(codebook_length)  # trick the oLDA to confuse codewords with just IDs.
    olda = OLDA.OnlineLDA(vocab, K, D, alpha, eta, tau0, kappa, updatect)
    olda._lambda = model.topic_word_  # check betty :(
    olda.add_new_topics(1)
    # minibatch_by_day = [len(videos_by_day[i]) for i in sorted(videos_by_day.keys())]

    # num_days = 4
    # for iteration in range(0, num_days):
    #     st = minibatch_by_day[iteration]
    #     en = minibatch_by_day[iteration+1]
    #
    #     ids = wordids[st:en]
    #     cts = wordcts[st:en]
    #     vids = video_names[st:en]
    #     labels = true_labels[st:en]
    #     new_codebook_lengths = per_video_codebook_length[st:en]

    #minibatches within the day
    test = []
    batchsize = 5
    num_iters = int(len(wordids)/float(batchsize))

    subjects_left_out = len(wordids) - (num_iters*5)

    for i in range(0, num_iters):
        ids = wordids[i*batchsize:(i+1)*batchsize]
        cts = wordcts[i*batchsize:(i+1)*batchsize]
        vids = video_names[i*batchsize:(i+1)*batchsize]

        labels = true_labels[i*batchsize:(i+1)*batchsize]
        new_codebook_lengths = per_video_codebook_length[i*batchsize:(i+1)*batchsize]

        new_codebook_length = max(new_codebook_lengths)+1
        if new_codebook_length != codebook_length:
            olda.add_new_features(new_codebook_length)
            # print "new code book length:", new_codebook_length

        (gamma, bound) = olda.update_lambda(ids, cts, False)
        # print ">>", bound, len(wordids), D, sum(map(sum, cts)), olda._rhot
        for n, gam in enumerate(gamma):
            gam = gam / float(sum(gam))

            if max(gam) > assign_class_thresh:
                thresholded_true.append(labels[n])
                pred_labels.append(np.argmax(gam))

        #add a new topic each day

        if i*5 in [0, 175, 210, 240, 340]:
            olda.add_new_topics(1)

        #results:
        n, l, v, h, c, mi, nmi, labs = print_results(thresholded_true, pred_labels, olda._K)
        continue

    if subjects_left_out>0:
        ids = wordids[-subjects_left_out:]
        cts = wordcts[-subjects_left_out:]
        vids = video_names[-subjects_left_out:]
        labels = true_labels[-subjects_left_out:]
        new_codebook_lengths = per_video_codebook_length[-subjects_left_out:]

        new_codebook_length = max(new_codebook_lengths)+1
        if new_codebook_length != codebook_length:
            olda.add_new_features(new_codebook_length)
            # print "new code book length:", new_codebook_length

        (gamma, bound) = olda.update_lambda(ids, cts, False)
        # print ">>", bound, len(wordids), D, sum(map(sum, cts)), olda._rhot
        for n, gam in enumerate(gamma):
            gam = gam / float(sum(gam))

            if max(gam) > assign_class_thresh:
                thresholded_true.append(labels[n])
                pred_labels.append(np.argmax(gam))

            #results:
            n, l, v, h, c, mi, nmi, labs = print_results(thresholded_true, pred_labels, olda._K)


    mat = confusion_mat(thresholded_true, pred_labels, olda._K)
    all_video_names_in_order = video_names_b + video_names

    import pdb; pdb.set_trace()

    f1 = open(directory+'/document_assignments.p', 'w')
    pickle.dump(pred_labels, f1, 2)
    f1.close()

    f1 = open(directory+'/topic_words.p' , 'w')
    pickle.dump(olda._lambda, f1, 2)
    f1.close()

    # ****************************************************************************************************
    # Write out results
    # ****************************************************************************************************
    name = '/results.txt'
    f1 = open(directory+name, 'w')
    f1.write('n_topics: %s \n' % olda._K)
    f1.write('n_iters: %s \n' % n_iters)
    f1.write('dirichlet_params: (%s, %s) \n' % (alpha, eta))
    f1.write('class_thresh: %s \n' % assign_class_thresh)
    f1.write('code book length: %s \n' % new_codebook_length)
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
    f1.close()


    # ****************************************************************************************************
    # Write out confusion matrix and results
    # ****************************************************************************************************
    confusion_dic    = {}
    for row, lab in zip(mat, labs):
        confusion_dic[lab] = row

    f1 = open(directory+'/confusion_mat.p', 'w')
    pickle.dump(mat, f1, 2)
    f1.close()

    f1 = open(directory+'/confusion_dic.p', 'w')
    pickle.dump(confusion_dic, f1, 2)
    f1.close()

sys.exit(1)


    # word_counter = 0
    # for i in cts:
    #     word_counter+=sum(i)
    # perwordbound = bound * len(wordids) / (D * word_counter)
    # print 'iter: %s:  rho_t = %f,  held-out per-word perplexity estimate = %f. LDA - Done\n' % \
    #     (iteration, olda._rhot, np.exp(-perwordbound))

    # if (iteration % 1 == 0):
    #     if not os.path.exists(os.path.join(directory, 'TopicData')): os.makedirs(os.path.join(directory, 'TopicData'))
    #     np.savetxt(directory + '/TopicData/lambda-%d.dat' % iteration, olda._lambda)
    #     np.savetxt(directory + '/TopicData/gamma-%d.dat' % iteration, gamma)
    #
    #     np.savetxt(directory + '/TopicData/vids-%d.dat' % iteration, vids, delimiter=" ", fmt="%s")
    #     np.savetxt(directory + '/TopicData/labels-%d.dat' % iteration, labels, delimiter=" ", fmt="%s")
    #
    #     sys.exit(1)
    #     # ****************************************************************************************************
    #     # save each document distribution
    #     # ****************************************************************************************************
    #     if not using_language:
    #         percentage = ""
