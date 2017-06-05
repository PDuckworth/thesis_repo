#!/usr/bin/env python
__author__ = 'p_duckworth'
import sys, os
import numpy as np
import matplotlib.pylab as plt
import cPickle as pickle

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
