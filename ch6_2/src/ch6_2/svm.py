#!/usr/bin/env python
__author__ = 'p_duckworth'
import os, sys, csv
import time
import scipy
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, cdist, euclidean
import matplotlib.pyplot as plt



def run_svm(data, labels):


    pred_labels, true_labels = [], []

    for fold in [0,1,2,3,4]:  #folds
        X_test, X_train, y_train = [], [], []
        for cnt, (i, j) in enumerate(zip(data, labels)):

            if (cnt + fold) % 5 ==0:
                # if cnt % 4 == 0:
                X_test.append(i)
                true_labels.append(j)
            else:
                X_train.append(i)
                y_train.append(j)

        C = 0.1
        # print len(X_train), len(y_train)
        svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
        # svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)

        for cnt, i in enumerate(X_test):
            lab = svc.predict(i.reshape(1,len(i)))[0]
            pred_labels.append(lab)
            # print y[cnt], lab
        # print "number of tests= ", cnt

    # for i, j in zip(true_labels, pred_labels):
    #     print i, j

    print "SVM: \nk=%s (%s). v-measure: %0.3f. homo: %0.3f. comp: %0.3f. MI: %0.3f. NMI: %0.3f. Acc: %0.3f \n \n" \
          % (len(set(true_labels)), len(pred_labels),  metrics.v_measure_score(true_labels, pred_labels),
             metrics.homogeneity_score(true_labels, pred_labels),
             metrics.completeness_score(true_labels, pred_labels), metrics.mutual_info_score(true_labels, pred_labels),
             metrics.normalized_mutual_info_score(true_labels, pred_labels),
             metrics.accuracy_score(true_labels, pred_labels))


def kmeans_clustering(data, true_labels, threshold=0.01):

    n_samples, n_features = data.shape
    print "data: ", data.shape
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

    print "Overall: "
    for key, val in repeat.items():
        print key, sum(val)/float(len(val))

##############################################
##    SEGMENTATION ANALYSIS:
##############################################


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

    plot_topic_similarity_matrix(cosine_matrix, title)

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
