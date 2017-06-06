#!/usr/bin/env python
__author__ = 'p_duckworth'
import sys, os
import numpy as np
import matplotlib.pylab as plt
import cPickle as pickle
from sklearn import metrics


def plot_confusion_matrix(cm, classes, loc = "", title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # %todo: save it somewhere

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


def screeplot(filepath, sigma, comps, div=2, vis=False):
    y = sigma
    x = np.arange(len(y)) + 1

    plt.subplot(2, 1, 1)
    plt.plot(x, y, "o-", ms=2)

    xticks = ["Comp." + str(i) if i%2 == 0 else "" for i in x]

    plt.xticks(x, xticks, rotation=45, fontsize=20)

    # plt.yticks([0, .25, .5, .75, 1], fontsize=20)
    plt.yticks(fontsize=15)
    plt.ylabel("Variance", fontsize=20)
    plt.xlim([0, len(y)])
    plt.title("Plot of the variance of each Singular component", fontsize=25)
    plt.axvspan(10, 11, facecolor='g', alpha=0.5)

    filepath_g = os.path.join(filepath, "graphs")
    if not os.path.exists(filepath_g):
        os.makedirs(filepath_g)

    plt.savefig(filepath_g + "/scree_plot.png", bbox_inches='tight')
    if vis:
        plt.show()
