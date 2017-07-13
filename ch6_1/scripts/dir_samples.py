#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec

corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

refiner = tri.UniformTriRefiner(triangle)
trimesh = refiner.refine_triangulation(subdiv=4)

# plt.figure(figsize=(8, 4))
# for (i, mesh) in enumerate((triangle, trimesh)):
#     plt.subplot(1, 2, i+ 1)
#     plt.triplot(mesh)
#     plt.axis('off')
#     plt.axis('equal')

# Mid-points of triangle sides opposite of each corner
midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0 \
             for i in range(3)]
def xy2bc(xy, tol=1.e-3):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    s = [(corners[i] - midpoints[i]).dot(xy - midpoints[i]) / 0.75 \
         for i in range(3)]
    return np.clip(s, tol, 1.0 - tol)


class Dirichlet(object):
    def __init__(self, alpha):
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                     reduce(mul, [gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa)in zip(x, self._alpha)])

def draw_pdf_contours(dist, nlevels=200, subdiv=8, **kwargs):
    import math

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    plt.show()




# Working with multiple figure windows and subplots
def draw_bars(data):

    # yticks = np.arange(0.0, 1.0, 0.2)
    xticks = np.arange(0.0, 12, 2)
    x = range(10)



    plt.figure(figsize = (3,6))
    gs1 = gridspec.GridSpec(3, 6)
    gs1.update(wspace=0.25, hspace=0.15) # set the spacing between axes.

    for i, hist in enumerate(data):

        ax = plt.subplot(gs1[i])
        ax.set_xlim([-0.5, 9.5])
        ax.set_ylim([0, 1.1])
        ax.set_xticks( range(0,10,2) )


        # only set the labels on the left and bottom column/row figs
        if i not in [0,6,12]:
            ax.set_yticklabels([])

        if i < 12:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels( range(0,10,2) )
            # ax.set_xticklabels([0,2,4,6,8])

        ax.grid(True)
        ticklines = ax.get_xticklines() + ax.get_yticklines()
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

        # for line in ticklines:
        #     line.set_linewidth(3)

        for line in gridlines:
            line.set_linestyle('-.')

        # for label in ticklabels:
        #     label.set_color('r')
        #     label.set_fontsize('medium')


        # draw the verticle line
        for xc, yc in enumerate(hist):
            ax.plot([xc, xc], [0, yc], 'b-', )
        # draw the dot
        ax.plot(x, hist, 'bo')

    # plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    """	Dirichlet draws  """
    #
    # draw_pdf_contours(Dirichlet([2, 2, 2]))
    # draw_pdf_contours(Dirichlet([0.6, 0.6, 0.6]))
    # draw_pdf_contours(Dirichlet([4, 4, 4]))

    dir_data = []
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    for i in xrange(3):
        for alpha in alphas:
            dir_data.append(np.random.dirichlet([alpha]*10))

    draw_bars(dir_data)
