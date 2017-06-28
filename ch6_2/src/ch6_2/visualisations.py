#!/usr/bin/env python
__author__ = 'p_duckworth'
import sys, os
import numpy as np
import matplotlib.pylab as plt
import cPickle as pickle
import getpass, datetime


from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import hungarian

def plot_topic_similarity_matrix(cosine_matrix, title=None):

    reordering = np.argmax(cosine_matrix, axis=1)
    reorder_values = np.max(cosine_matrix, axis=1)
    reordered = cosine_matrix[:,reordering]
    print "reordering columns:", reordering
    print "values:", reorder_values

    # Normalise Similarity Matrix - ? No?
    norm_mat = []
    for i in reordered:
        norm_mat.append([float(j)/float( sum(i, 0) ) for j in i ])

    print "\nnormed: ",norm_mat
    print "\n, reordered: ", reordered

    classes = ["a","b","c","d","e","f","g","h","i","j","k"]
    xlabel = 'Concatenated Clips Topics'
    ylabel = 'Segmented Clips Topics'
    plot_confusion_matrix(reordered, classes, "", title, plt.cm.Blues, xlabel, ylabel)

    return


def plot_confusion_matrix(cm, classes, loc = "", title='Confusion matrix', cmap=plt.cm.Blues, xlabel = 'Predicted label', ylabel= 'True label'):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    topics = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, color='red', fontsize=16)
    plt.yticks(tick_marks, topics, color='red', fontsize=16)
    plt.tight_layout()
    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)

    width, height = cm.shape
    for x in xrange(width):
        for y in xrange(height):
            plt.annotate("%0.2f" % cm[x][y], xy=(y, x), horizontalalignment='center', verticalalignment='center')

    plt.show()
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

    norm_gt = []
    for i in mat:
        norm_gt.append([float(j)/float( sum(i) ) for j in i ])


    f, ax = plt.subplots(nrows=1, ncols=1)
    plt.title("Topic Confusion Marix: Leeds Activity Dataset", fontsize=14)
    plt.xlabel('Learned Topics')
    plt.ylabel('Ground Truth Labels')

    classes = ["a","b","c","d","e","f","g","h","i","j","k"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=14) #, rotation=90)

    # plt.setp((ax), xticks=range(len(mat[0])), yticks=xrange(len(set_of_true_labs)))
    plt.yticks(xrange(len(set_of_true_labs)), set_of_true_labs, color='red', fontsize=14)

    cmap=plt.cm.Blues
    plt.imshow(norm_gt, interpolation='nearest', cmap=cmap)

    plt.tight_layout()
    plt.colorbar()

    # reorder = np.argmax(mat, axis=1)
    # reorder = [3, 8, 1 , 5, 0, 2, 4, 7, 6, 9]
    # plt.yticks(tick_marks, [yaxis[i] for i in reorder])
    # ordering = raw_input("reordering = ",)

    ordering = np.array([4,3,8,7,6,5,2,4,0,9,10]) #raw_input("enter reordering")
    # reordering = np.argmax(norm_gt, axis=1)
    reordered_mat = np.vstack(norm_gt)[:, np.array(ordering)]

    f, ax = plt.subplots(nrows=1, ncols=1)
    f.suptitle("Topic Confusion Marix: Leeds Activity Dataset\n(normalised by GT)", fontsize=20)
    plt.xlabel('Learned Topics', fontsize=18)
    plt.ylabel('Ground Truth Labels', fontsize=18)


    plt.xticks(tick_marks, classes, fontsize=14) #, rotation=90)
    # plt.setp((ax), xticks=range(len(mat[0])), yticks=xrange(len(set_of_true_labs)))

    plt.yticks(xrange(len(set_of_true_labs)), set_of_true_labs, color='red', fontsize=14)
    plt.imshow(reordered_mat, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.show()

    # width, height = reordered.shape
    # for x in xrange(width):
    #     for y in xrange(height):
    #         ax2.annotate("%0.2f" % reordered[x][y], xy=(y, x), horizontalalignment='center', verticalalignment='center')

    norm_topics = mat / mat.max(axis=0)

    f, ax = plt.subplots(nrows=1, ncols=1)
    f.suptitle("Topic Confusion Marix: Leeds Activity Dataset\n(normalised by topic)", fontsize=20)
    plt.xlabel('Learned Topics', fontsize=18)
    plt.ylabel('Ground Truth Labels', fontsize=18)

    plt.xticks(tick_marks, classes, fontsize=14) #, rotation=90)
    # plt.setp((ax), xticks=range(len(mat[0])), yticks=xrange(len(set_of_true_labs)))

    plt.yticks(xrange(len(set_of_true_labs)), set_of_true_labs, color='red', fontsize=14)
    plt.imshow(norm_topics[:, ordering], interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.show()


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
    ax1.set_xlabel('code word', fontsize=15)
    ax1.set_ylabel('Weight', fontsize=15)
    ax1.set_title('Latent Concept 1', fontsize=20)
    # ax1.set_ylim([0,0.1])
    # ax1.set_title('Topic 1', fontsize=25)
    ax1.grid(True)

    ax2 = fig.add_subplot(212)
    # ax1.plot(t, data2, 'b^')
    ax2.vlines(t, [0], data2)
    ax2.set_xlabel('code word', fontsize=15)
    ax2.set_ylabel('Weight', fontsize=15)
    ax2.set_title('Latent Concept 2', fontsize=20)
    # ax2.set_ylim([0, 0.1])
    # ax2.set_title('Topic 2', fontsize=25)
    ax2.grid(True)
    plt.show()


def genome_rel(data1, i, ymax=0):

    t = np.arange(0.0, len(data1), 1)

    print "\ndata1", min(data1), max(data1), sum(data1)/float(len(data1))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.vlines(t, [0], data1)
    ax1.set_xlabel('code word', fontsize=20)
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
    ax1.set_xlabel('code word', fontsize=20)
    ax1.set_ylabel('Weight', fontsize=20)
    ax1.set_ylim([0,ymax])
    ax1.set_title('Topic %s' %i, fontsize=25)
    ax1.grid(True)
    plt.show()


def screeplot(sigma, comps=100, filepath = "/tmp", div=2, vis=False):
    y = sigma
    x = np.arange(len(y)) + 1

    plt.subplot(2, 1, 1)
    plt.plot(x, y, "o-", ms=2)
    xticks = ["Comp." + str(i) if i%25 == 0 else "" for i in x]

    plt.xticks(x, xticks, rotation=45, fontsize=20)

    # plt.yticks([0, .25, .5, .75, 1], fontsize=20)
    plt.yticks(fontsize=15)
    plt.ylabel("Variance", fontsize=20)
    plt.xlim([0, len(y)])
    plt.title("Plot of the variance of each Singular component", fontsize=25)
    plt.axvspan(11, 12, facecolor='g', alpha=0.7)

    # filepath_g = os.path.join(filepath, "graphs")
    # if not os.path.exists(filepath_g): os.makedirs(filepath_g)

    plt.savefig(filepath + "/scree_plot.png", bbox_inches='tight')
    if vis:
        plt.show()




def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts



def make_radar_plot(topic_words, path = ""):


    # with open(accu_path + "/topic_word.p", 'r') as f:
    #     topic_words = pickle.load(f)

    topic_limit = 5
    topic_names = ["a"]#, "b", "c", "d", "e", "f"]

    keep = []
    for cnt, row in enumerate(topic_words):
        if cnt >= topic_limit: continue
        keep.extend(row.argsort()[-10:][::-1])
    keep = list(set(keep))

    selected_cols = topic_words.T[keep]
    topic_words = selected_cols.T
    print "new feature space shape: ", topic_words.shape
    wordids = keep

    N = topic_words.shape[1]
    num_topics = topic_words.shape[0]
    data1 = []
    for topic_id, topic in enumerate(topic_names[:num_topics])  :
        tup = (topic_names[topic_id], [list(topic_words[topic_id])])
        data1.append(tup)

    d = []
    for topic_id in xrange(num_topics):
        if topic_id > topic_limit: continue
        d.append(list(topic_words[topic_id]))
    data = ("", d)

    theta = radar_factory(N, frame='polygon')

    spoke_labels = sorted(wordids)

    # fig, axes = plt.subplots(figsize=(9, 9), nrows=2, ncols=2, subplot_kw=dict(projection='radar'))
    # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)


    colors = ['b', 'r', 'g', 'm', 'y']
    # Plot the four cases from the example data on separate axes
    (title, case_data) = data
    # for ax, (title, case_data) in zip(axes.flatten(), data):

    fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])

    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                 horizontalalignment='center', verticalalignment='center')
    for d, color in zip(case_data, colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25)
    ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    # ax = axes[0, 0]
    labels = ('Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5')
    legend = ax.legend(labels, loc=(0.9, .95), labelspacing=0.1)

    fig.text(0.5, 0.965, 'Leeds Activity Dataset LDA Model Radar Plot',
             horizontalalignment='center', color='black', weight='bold',
             size='large')

    plt.show()
    # plt.savefig('/tmp/topics.png', bbox_inches='tight')
