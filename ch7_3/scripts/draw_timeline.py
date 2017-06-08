#! /usr/bin/env python
__author__ = 'p_duckworth'
import sys, os
import getpass, datetime

import math
import numpy as np
import cPickle as pickle
import time
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import math

cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',

'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',

'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',

'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',

'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'gold':                 '#FFD700',
'indigo':               '#4B0082',

'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',

'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',

'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',

'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}


colours = {
'peru':                 '#CD853F',
'dodgerblue':           '#1E90FF',
'brown':                '#A52A2A',
'orange':               '#FFA500',
'lightsalmon':          '#FFA07A',
'palegreen':            '#98FB98',
'pink':                 '#FFC0CB',
'indianred':            '#CD5C5C',
'gold':                 '#FFD700',
'cyan':                 '#00FFFF',
'purple':               '#800080',
'teal':                 '#008080',
'wheat':                '#F5DEB3',
'gray':                 '#808080',
'darkgreen':            '#006400',
}


path = "/home/scpd/Datasets/ECAI_Data/dataset_segmented_15_12_16/QSR_path/run_38/online_rec/"
name1 = "online_gt.p"
name2 = "online_predictions.p"

act_colours = {"na" :  'gray', 100  : 'gray',
            "microwaving_food": 'purple',        4 :'purple',
            "use_kettle": 'orange',              3 :'orange',
            "take_paper_towel" :'gold',
            "openning_double_doors" :'wheat',    7 :'wheat',
            "take_tea_bag" :'darkgreen',         6 :'darkgreen',
            "take_from_fridge":'pink',           5: 'pink',
            "printing_interface":'teal',         2 :'teal',
            "use_water_cooler":'palegreen',
            "printing_take_printout":'cyan',
            "throw_trash":'brown',               8 :'brown',
            "washing_up":'dodgerblue',           10 :'dodgerblue', 1 : 'dodgerblue', 9 :'dodgerblue', 0 :'dodgerblue'
        }


act_colours = {"na" :  'gray', 100  : 'gray',
            "microwaving_food": 'purple',        4 :'purple',
            "use_kettle": 'orange',              3 :'orange',
            "take_paper_towel" :'gold',
            "openning_double_doors" :'wheat',    7 :'wheat',
            "take_tea_bag" :'darkgreen',         6 :'darkgreen',
            "take_from_fridge":'pink',           5: 'pink',
            "printing_interface":'teal',         2 :'teal',
            "use_water_cooler":'palegreen',
            "printing_take_printout":'cyan',
            "throw_trash":'brown',               8 :'brown',
            "washing_up":'dodgerblue',           10 :'dodgerblue', 1 : 'dodgerblue', 9 :'dodgerblue', 0 :'dodgerblue'
        }



with open(path+name1, 'r') as f:
    true_labels = pickle.load(f)
true_labels = [t if t != "making_tea" else "na" for t in true_labels]

with open(path+name2, 'r') as f:
    pred_labels = pickle.load(f)

fig = plt.figure()
ax = fig.add_subplot(111)

start_break = 30000
end_break = 40000
ax.set_ylim(7,9)
ax.set_xlim(start_break, end_break)
range_ = range(start_break,end_break)

x1, x2 = 0, 0
y1, y2 = 7.5, 7.5
prev_lab = pred_labels[0]
for frame in range_:
    lab = pred_labels[frame]
    if lab != prev_lab:
        x2 = frame-1
        # print (x1, x2), act_colours[lab]
        plt.plot([x1, x2], [y1,y2], color=act_colours[prev_lab], linestyle='-', linewidth=300)
        x1 = frame
    prev_lab=lab

x1, x2 = 0, 0
y1, y2 = 8.5, 8.5
prev_lab = true_labels[0]

for frame in range_:
    lab = true_labels[frame]
    # if frame == 42000:
    #     pdb.set_trace()
    if lab != prev_lab:
        x2 = frame-1
        # print (x1, x2), act_colours[lab]
        plt.plot([x1, x2], [y1,y2], color=act_colours[prev_lab], linestyle='-', linewidth=300)
        x1 = frame
    prev_lab=lab

plt.show()

# x_count = count * ratio
# y_count = count / ratio
# x = 0
# y = 0
# w = 1 / x_count
# h = 1 / y_count
#
#
#     pos = (x / x_count, y / y_count)
#     ax.add_patch(patches.Rectangle(pos, w, h, color=c))
#     ax.annotate(c, xy=pos)
#     if y >= y_count-1:
#         x += 1
#         y = 0
#     else:
#         y += 1
#
# plt.show()
