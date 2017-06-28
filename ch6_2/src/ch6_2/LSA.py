#!/usr/bin/env python
__author__ = 'p_duckworth'
import os, sys, csv
import getpass
import time
import cPickle as pickle
import numpy as np
import math
import multiprocessing as mp
import itertools
import matplotlib.pyplot as plt
import hungarian
# from read_data import *
# from manage_histograms import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.fixes import bincount
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, cdist, euclidean
import ch6_2.visualisations as vis
import scipy

def get_tf_idf_scores(data, tfidf=True, vis=False):

    if tfidf:

        ##BINARY COUNTING OF FEATURES:
        # #Feature space info:
        feature_freq = (data != 0).sum(axis=0)  # TF: document_frequencies
        (N, f) = data.shape                     # Number of documents, and number of features
        print "nuber of documents = %s, number of features = %s " %(N, f)

        ## Inverse document frequency scores
        ## LONG HAND
        # idf_scores=[]
        # for i in feature_freq:
        #     try:
        #         idf_scores.append(math.log((N /float(i))))
        #     except:
        #         idf_scores.append(0)

        idf_scores = [(math.log((N / float(i)))) if i > 0 else 0 for i in feature_freq]

        tf_idf_scores = np.array([[]])

        for histogram in data:

            #freq = 1+math.log(freq) if freq>0 else 0  #log normalisation of Term Frequency
            foo = [idf_scores[cnt]*(math.log(1+freq)) for cnt, freq in enumerate(histogram)]
            try:
                tf_idf_scores = np.append(tf_idf_scores, np.array([foo]), axis=0)
            except ValueError:
                tf_idf_scores = np.array([foo])
        print "new shape:", tf_idf_scores.shape

    else:
        tf_idf_scores = data
    return np.array(tf_idf_scores)


def svd_clusters(term_freq, all_labels, n_comps=11, threshold=0.01, random=False):

    data = get_tf_idf_scores(term_freq)

    U, Sigma, VT = randomized_svd(data, n_components=1000, n_iter=5, random_state=None)

    # for i, j in itertools.combinations(VT[2:7], 2):
    # vis.comare_genomes(VT[2], VT[3])
    # vis.comare_genomes(VT[2], VT[4])
    # vis.comare_genomes(VT[2], VT[5])

    # X_transformed = np.dot(U, np.diag(Sigma))
    #Investigate data

    # vis.screeplot(Sigma, vis=False)

    #print "all labels:", len(set(all_labels)), set(all_labels)
    #print "\nLSA Results"
    max_v = 0

    if random: print "compare to RANDOM CHANCE"

    repeat = {"v": [], "ho": [], "co": [], "mi": [], "nmi": [], "a":[] }
    for i in xrange(10):   #repeat 10 times
        U, Sigma, VT = randomized_svd(data, n_components=n_comps, n_iter=5, random_state=None)

        random_labels = []
        true_labels, pred_labels = [], []
        U_assignments = []
        for label, doc in zip(all_labels, U):
            if np.max(abs(doc))>threshold:
                pred_labels.append(np.argmax(doc))
                true_labels.append(label)
                random_labels.append(label)
                temp = [0]*n_comps
                temp[np.argmax(doc)] = 1
                U_assignments.append(temp)

        # new_labels = [np.argmax(doc) > 0.1 for doc in U]
        num_clusters = (Sigma > 1.0).sum(axis=0)

        if random == True:
            #calculate the random chance scores:
            from random import shuffle
            shuffle(random_labels)
            pred_labels = random_labels

        # print "k=%s (%s). v-measure: %0.3f. homo: %0.3f. comp: %0.3f. MI: %0.3f. NMI: %0.3f. Acc: %0.3f. LL: %0.3f." \
        #   % (num_clusters, len(pred_labels), metrics.v_measure_score(true_labels, pred_labels),
        #      metrics.homogeneity_score(true_labels, pred_labels),
        #      metrics.completeness_score(true_labels, pred_labels), metrics.mutual_info_score(true_labels, pred_labels),
        #      metrics.normalized_mutual_info_score(true_labels, pred_labels),
        #      metrics.accuracy_score(true_labels, pred_labels),
        #      metrics.log_loss(true_labels, U_assignments))

        v = ("V-measure: %0.3f" % metrics.v_measure_score(true_labels, pred_labels))
        if v > max_v:
            max_v = v
            num_c = n_comps

        repeat["v"].append(metrics.v_measure_score(true_labels, pred_labels))
        repeat["ho"].append(metrics.homogeneity_score(true_labels, pred_labels))
        repeat["co"].append(metrics.completeness_score(true_labels, pred_labels))
        repeat["mi"].append(metrics.mutual_info_score(true_labels, pred_labels))
        repeat["nmi"].append(metrics.normalized_mutual_info_score(true_labels, pred_labels))
        repeat["a"].append(metrics.accuracy_score(true_labels, pred_labels))
    print "Overall: "
    for key, val in repeat.items():
        print key, sum(val)/float(len(val))
    print "\n"



    U, Sigma, VT = randomized_svd(data, n_components=num_c, n_iter=5, random_state=None)
    true_labels, pred_labels = [], []
    for label, doc in zip(all_labels, U):
        if np.max(abs(doc)) > threshold:
            pred_labels.append(np.argmax(doc))
            true_labels.append(label)

    return VT

    # print "k=%s (%s). v-measure: %0.3f. homo: %0.3f. comp: %0.3f. MI: %0.3f. NMI: %0.3f." \
    #       % (num_c, len(pred_labels),  metrics.v_measure_score(true_labels, pred_labels),
    #          metrics.homogeneity_score(true_labels, pred_labels),
    #          metrics.completeness_score(true_labels, pred_labels), metrics.mutual_info_score(true_labels, pred_labels),
    #          metrics.normalized_mutual_info_score(true_labels, pred_labels))



def compare_two_LSA_models(topic_word1, code_book1, topic_word2, code_book2, title=""):

    #Adjust the columns on two topic_word matrices, and use hungarian alg to find best matches.

    print "shape of both codebooks:", len(code_book1), len(code_book2)
    f1, f2, global_codebook = create_feature_spaces_over_codebooks(topic_word1, code_book1, topic_word2, code_book2)

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
            genome(f1[i], f2[reordering[i]])





def create_feature_spaces_over_codebooks(topic_word1, code_book1, topic_word2, code_book2):

    list_of_cross_over_hashes = []
    global_codebook = code_book1
    for ind2, hash in enumerate(code_book2):
        if hash in code_book1:
            ind1 = np.where(code_book1 == hash)[0][0]  # Or use something quicker like: code_book1.find(hash) (which doesnt exist in np < 2.0.0)
            # print "in:", ind1, ind2
            list_of_cross_over_hashes.append( (ind1, ind2))
        else:
            global_codebook = np.append(global_codebook, hash)

    print "\nlen global codebook:", len(global_codebook)
    (n_topics1, n_features1) = topic_word1.shape
    (n_topics2, n_features2) = topic_word2.shape

    # Create f1 feature space over the new global code book.
    f1 = np.zeros( (n_topics1, len(global_codebook)))

    for cnt, row in enumerate(topic_word1):
        f1[cnt][:n_features1] = row

    # print ">", f1[0], f1.shape, len(f1[0])
    # print ">", topic_word1[0]
    print "> crossovers", len(list_of_cross_over_hashes)

    f2 = np.zeros( (n_topics2, len(global_codebook)))

    # CROSS OVER CODE WORDS
    for cnt, (ind1, ind2) in enumerate(list_of_cross_over_hashes):
        keep_col = topic_word2.T[ind2]
        # print "a:", cnt, ind1, ind2, len(f2.T[ind1][:n_topics2]), len(f2.T[ind1]), len(keep_col)
        # f2.T[ind1][:n_topics2] = keep_col
        f2.T[ind1] = keep_col

    print ">", f2.shape

    # Then remove the cross over code words.
    only_ind2s = [j for i, j in list_of_cross_over_hashes]
    for ind2 in sorted(only_ind2s, reverse=True):
        topic_word2 = scipy.delete(topic_word2, ind2, 1)

    # Finally stick the vectors onto the end of each row in f2 feature space
    for cnt, row in enumerate(topic_word2):
        # if cnt in [ i for i, j in list_of_cross_over_hashes]:
        #     continue
        remaining_inds = len(global_codebook) - n_features2 + len(list_of_cross_over_hashes)
        f2[cnt][remaining_inds:] = row

    print "new feature space shapes:", f1.shape, f2.shape
    return f1, f2, global_codebook







def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def non_svd_clustering(path, tf_data=None, threshold=0.01):

    (data, true_labels) = tf_data

    n_samples, n_features = data.shape

    # determine your range of K
    k_range = range(10, 11)
    repeat = {"v": [], "ho": [], "co": [], "mi": [], "nmi": []}

    print "Kmeans Results"
    for rerun in xrange(10):
        # Fit the kmeans model for each n_clusters = k
        k_means_var = [KMeans(n_clusters=k, init='k-means++').fit(data) for k in k_range]

        # Pull out the cluster centeres for each model
        centroids = [X.cluster_centers_ for X in k_means_var]

        # Total within-cluster sum of squares (variance or inertia)
        inertia_ = [X.inertia_ for X in k_means_var]
        # print inertia_

        # silhouette scores of each model
        sil_scores = [metrics.silhouette_score(data, X.labels_, metric='sqeuclidean') for X in k_means_var]

        # Calculate the Euclidean distance from each point to each cluster center
        k_euclid = [cdist(data, cent, 'euclidean') for cent in centroids]
        dist = [np.min(ke, axis=1) for ke in k_euclid]

        # Keep the index of which cluster is clostest to each example
        cluster_comp = [np.argmin(ke, axis=1) for ke in k_euclid]

        for i, k in enumerate(k_range):
            pred_labels = cluster_comp[i]
            print "k: %s. v-measure: %0.3f. homo: %0.3f. comp: %0.3f. MI: %0.3f. NMI: %0.3f. "  \
              %(k, metrics.v_measure_score(true_labels, pred_labels), metrics.homogeneity_score(true_labels, pred_labels),
                metrics.completeness_score(true_labels, pred_labels), metrics.mutual_info_score(true_labels, pred_labels),
                metrics.normalized_mutual_info_score(true_labels, pred_labels))

        repeat["v"].append(metrics.v_measure_score(true_labels, pred_labels))
        repeat["ho"].append(metrics.homogeneity_score(true_labels, pred_labels))
        repeat["co"].append(metrics.completeness_score(true_labels, pred_labels))
        repeat["mi"].append(metrics.mutual_info_score(true_labels, pred_labels))
        repeat["nmi"].append(metrics.normalized_mutual_info_score(true_labels, pred_labels))

    for key, val in repeat.items():
        print key, sum(val)/float(len(val))
    print "\n"

    # The total sum of squares
    #tss = sum(pdist(data) ** 2 / n_samples)
    # print "The total sum of squares: %0.3f" % tss
    # Total within-cluster sum of squares (variance)
    #wcss = [sum(d ** 2) for d in dist]
    # The between-cluster sum of squares
    #print "Between-cluster sum of squares: %s" % (tss - wcss)
    #print "Percentage of Variance explained by clustering: \n%s" % (((tss - wcss) / tss) * 100)
    #for cnt, i in enumerate(inertia_):
    #    print "k: %s. inertia: %0.2f. Penalty: %0.2f. silhouette: %0.3f." \
    #          % (k_range[cnt], i, i * k_range[cnt], sil_scores[cnt])



def get_supervised_svm(path, tf_data=None, which_features="features"):
    from sklearn import svm

    try:
        (data, all_labels) = tf_data
        print all_labels[:5]
    except:
        print "except..."
        # date = '13_04_2016'
        # date = time.strftime("%d_%m_%Y")
        date = "Seg_ecai"
        with open(path + "accumulate_data/Seg_ecai/feature_space.p", 'r') as f:
            data = pickle.load(f)
        with open(path + "accumulate_data/Seg_ecai/labels.p", 'r') as f:
            labels = pickle.load(f)

        all_labels = []
        for (dir, file_name) in labels:
            # print ">>", dir, file_name
            temp = "_".join(file_name.split("_")[1:-3])           ## CHECK THIS STILL HOLDS:
            all_labels.append(temp)

    pred_labels, true_labels = [], []

    print data[0]
    for fold in [0,1,2,3,4]:  #folds
        X_test, X_train, y_train = [], [], []
        for cnt, (i, j) in enumerate(zip(data, all_labels)):

            if (cnt + fold) % 5 ==0:
                # if cnt % 4 == 0:
                X_test.append(i)
                true_labels.append(j)
            else:
                X_train.append(i)
                y_train.append(j)

        C =.1
        # print len(X_train), len(y_train)
        svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
        # svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)

        for cnt, i in enumerate(X_test):
            # print y[cnt], svc.predict(i.reshape(1,11))
            lab = svc.predict(i.reshape(1,len(i)))[0]
            pred_labels.append(lab)
        # print "number of tests= ", cnt

    print "\nSVM results"
    # for i, j in zip(true_labels, pred_labels):
    #     print i, j

    print "k=%s (%s). v-measure: %0.3f. homo: %0.3f. comp: %0.3f. MI: %0.3f. NMI: %0.3f. Acc: %0.3f" \
          % (len(set(true_labels)), len(pred_labels),  metrics.v_measure_score(true_labels, pred_labels),
             metrics.homogeneity_score(true_labels, pred_labels),
             metrics.completeness_score(true_labels, pred_labels), metrics.mutual_info_score(true_labels, pred_labels),
             metrics.normalized_mutual_info_score(true_labels, pred_labels),
             metrics.accuracy_score(true_labels, pred_labels))


def compare_LDA_over_different_data(path, segmented_data):

    threshold=0.01

    """SEGMENTED"""
    id = "Seg_%s" % which_setting
    in_path = os.path.join(path, "accumulate_data", id)

    print in_path
    (data, labels), code_book1 = get_tf_idf_scores(in_path, tfidf=True, vis=False)
    print ">>", data.shape

    U, Sigma, VT1 = randomized_svd(data, n_components=11, n_iter=5, random_state=None)
    print "vt>", VT1

    print "\n"


    """CONCATENTED """
    id = id + "_concated"
    in_path = os.path.join(path, "accumulate_data", id)
    print in_path

    (data, labels), code_book2 = get_tf_idf_scores(in_path, tfidf=True, vis=False)
    print ">>", data.shape

    U, Sigma, VT2 = randomized_svd(data, n_components=11, n_iter=5, random_state=None)
    print "vt>", VT2


    compare_two_LSA_models(VT1, code_book1, VT2, code_book2, title="LSA comparison")


if __name__ == "__main__":
    """	Load the feature space,
    (i.e. one histogram over the global codebook, per event)
    Maintain the labels, and graphlet iGraphs (for the learning/validation)
    """

    ##DEFAULTS:
    path = '/home/' + getpass.getuser() + '/Datasets/Lucie_skeletons/'
    which_setting = select_options()

    """UNSUPERVISED LSA - comparison of concepts between segmented data and concatenated"""
    compare_LDA_over_different_data(path, which_setting)
    sys.exit(1)

    id1 = "Seg_%s" % which_setting
    in_path = os.path.join(path, "accumulate_data", id1)
    tf_idf_scores = get_tf_idf_scores(path, tfidf=True, vis=False)
    """UNSUPERVISED KMEANS IMPLEMENTATION #comparison to (AAMAS2016)"""
    # non_svd_clustering(path, tf_idf_scores, threshold=0.01)

    """UNSUPERVISED LSA IMPLEMENTATION"""
    get_svd_learn_clusters(path, tf_idf_scores, threshold=0.01, random=False)

    """UNSUPERVISED RANDOM CHANCE"""
    # get_svd_learn_clusters(path, tf_idf_scores, threshold=0.01, random=True)

    """SUPERVISED SVM IMPLEMENTATION (with cv)"""
    # get_supervised_svm(path, tf_idf_scores)   #With TFIDF scores
    get_supervised_svm(path)                  #Load data without TFIDF scores
