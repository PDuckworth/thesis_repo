#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import argparse
import cPickle as pickle
import numpy as np
import scipy as sp
import scipy.signal
import getpass
import math
import matplotlib.pyplot as plt
from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State, World_Trace
from tf.transformations import euler_from_quaternion
import pdb
from random import randint
#from ch6_2 import ECAI_videos_segmented_by_day #import segmented_videos
import ch6_2.ECAI_videos_segmented_by_day as seg


def pretty_print_world_qsr_trace(which_qsr, qsrlib_response_message):
    print(which_qsr, "request was made at ", str(qsrlib_response_message.req_made_at)
          + " and received at " + str(qsrlib_response_message.req_received_at)
          + " and finished at " + str(qsrlib_response_message.req_finished_at))
    print("---")
    print("Response is:")
    for t in qsrlib_response_message.qsrs.get_sorted_timestamps():
        foo = str(t) + ": "
        for k, v in zip(qsrlib_response_message.qsrs.trace[t].qsrs.keys(),
                        qsrlib_response_message.qsrs.trace[t].qsrs.values()):
            foo += str(k) + ":" + str(v.qsr) + "; "
        print(foo)

def get_soma_objects():
    # kitchen
    objects = {
    'Printer_console_11': (-8.957, -17.511, 1.1),
    'Printer_paper_tray_110': (-9.420, -18.413, 1.132),
    'Microwave_3': (-4.835, -15.812, 1.0),
    'Kettle_32': (-2.511, -15.724, 1.41),
    'Tea_Pot_47': (-3.855, -15.957, 1.0),
    'Water_Cooler_33': (-4.703, -15.558, 1.132),
    'Waste_Bin_24': (-1.982, -16.681, 0.91),
    'Waste_Bin_27': (-1.7636072635650635, -17.074087142944336, 0.5),
    'Sink_28': (-2.754, -15.645, 1.046),
    'Fridge_7': (-2.425, -16.304, 0.885),
    'Paper_towel_111': (-1.845, -16.346, 1.213),
    'Double_doors_112': (-8.365, -18.440, 1.021) }
    return objects

def get_point_cloud_objects(path):
    objects = {}

    # allowed_objects = ['22','13','21','23','4','17','11','15','19','30','28','3','44','0','12']
    # Objects clusters extracted from point clouds by Nils, then filtered based on a where the human hands interact with (applying a 3% margin around the BBs).
    allowed_objects = ['21','13','17','19','4','22','18','15','23','3','11','24','16','12']
    for file in sorted(os.listdir(path)):
        if file.endswith(".txt"):
            file_num = file.replace("cluster_","").replace(".txt","")
            if file_num not in allowed_objects: continue
            with open(os.path.join(path,file),'r') as f:
                f.readline()
                line=f.readline()
                (labs,xyz) = line.replace("\n","").split(":")
                x,y,z = xyz.split(",")
                objects["object_%s_%s" % (file_num,file_num)] = (float(x),float(y),float(z)) # hack to keep file_num when object type is passed to QSRLib
    return objects

def get_sk_info(f1):
    joints = {}
    for count, line in enumerate(f1):
        if count == 0:
            t = np.float64(line.split(':')[1].split('\n')[0])
        # read the joint name
        elif (count-1)%10 == 0:
            j = line.split('\n')[0]
            joints[j] = []
        # read the x value
        elif (count-1)%10 == 2:
            a = float(line.split('\n')[0].split(':')[1])
            joints[j].append(a)
        # read the y value
        elif (count-1)%10 == 3:
            a = float(line.split('\n')[0].split(':')[1])
            joints[j].append(a)
        # read the z value
        elif (count-1)%10 == 4:
            a = float(line.split('\n')[0].split(':')[1])
            joints[j].append(a)
    return joints

def get_rob_info(f1):
    rob_data = [[], []] # format: [(xyz),(r,p,y)]
    for count, line in enumerate(f1):
        # read the x value
        if count == 1:
            a = float(line.split('\n')[0].split(':')[1])
            rob_data[0].append(a)
        # read the y value
        elif count == 2:
            a = float(line.split('\n')[0].split(':')[1])
            rob_data[0].append(a)
        # read the z value
        elif count == 3:
            a = float(line.split('\n')[0].split(':')[1])
            rob_data[0].append(a)
        # read roll pitch yaw
        elif count == 5:
            ax = float(line.split('\n')[0].split(':')[1])
        elif count == 6:
            ay = float(line.split('\n')[0].split(':')[1])
        elif count == 7:
            az = float(line.split('\n')[0].split(':')[1])
        elif count == 8:
            aw = float(line.split('\n')[0].split(':')[1])
            # ax,ay,az,aw
            roll, pitch, yaw = euler_from_quaternion([ax, ay, az, aw])    #odom
            pitch = 10*math.pi / 180.   #we pointed the pan tilt 10 degrees
            rob_data[1] = [roll, pitch, yaw]
    return rob_data

def plot_mean_filter(values, filtered_values, text1, text2):
    t = range(len(values))
    plt.subplot(2,1,1)
    plt.plot(t,values,'yo-', markersize=2)
    plt.title(text1)
    # plt.title('input %s position: %s' % (joint_id, dim))
    plt.xlabel('time')
    plt.subplot(2,1,2)
    plt.plot(t,filtered_values,'yo-', markersize=2)
    plt.title(text2)
    # plt.title('filtered %s position: %s' % (joint_id, dim))
    plt.xlabel('frames')
    plt.show()


def apply_median_filter(skeleton_data, window_size=11, vis=False):
    """Once obtained the joint x,y,z coords.
    Apply a median filter over a temporal window to smooth the joint positions.
    Whilst doing this, create a world Trace object for the camera frame coordinates"""

    fx = 525.0
    fy = 525.0
    cx = 319.5
    cy = 239.5

    data, f_data, ob_states = {}, {}, {}
    camera_world = World_Trace()
    f_skeleton_data = {}

    for cnt, t in sorted(enumerate(skeleton_data.keys())):
        f_skeleton_data[t] = {}
        for joint_id, (x,y,z) in skeleton_data[t].items():
            # if joint_id == "torso":
            #     print "\njointID=", joint_id, (x,y,z)
            try:
                data[joint_id]["x"].append(x)
                data[joint_id]["y"].append(y)
                data[joint_id]["z"].append(z)
            except:
                data[joint_id] = {"x":[x], "y":[y], "z":[z]}

    for joint_id, joint_dic in data.items():
        f_data[joint_id] = {}
        for dim, values in joint_dic.items():
            filtered_values = sp.signal.medfilt(values, window_size) # filter

            if dim is "z" and 0 in filtered_values:
                print "Z should never be 0. (distance to camera):", joint_id, filtered_values
                filtered_values = [0.5 if i==0 else i for i in filtered_values]

            f_data[joint_id][dim] = filtered_values

            if vis and "hand" in joint_id:
                print "plotting ", joint_id, "  " , dim
                title1 = 'input %s position: %s' % (joint_id, dim)
                title2 = 'filtered %s position: %s' % (joint_id, dim)
                plot_mean_filter(values, filtered_values, title1, title2)

        # Create a QSRLib format list of Object States (for each joint id)
        for cnt, t in enumerate(f_skeleton_data.keys()):
            x = f_data[joint_id]["x"][cnt]
            y = f_data[joint_id]["y"][cnt]
            z = f_data[joint_id]["z"][cnt]

            # add the x2d and y2d (using filtered x,y,z data)
            # print self.uuid, t, joint_id, (x,y,z)
            x2d = int(x*fx/z*1 +cx);
            y2d = int(y*fy/z*-1+cy);
            f_skeleton_data[t][joint_id] = (x, y, z, x2d, y2d)    # Kept for Legacy
            try:
                ob_states[joint_id].append(Object_State(name=joint_id, timestamp=cnt, x=x, y=y, z=z))
            except:
                ob_states[joint_id] = [Object_State(name=joint_id, timestamp=cnt, x=x, y=y, z=z)]
    # #Add all the joint IDs into the World Trace
    for joint_id, obj in ob_states.items():
        camera_world.add_object_state_series(obj)

    return f_skeleton_data, camera_world

def object_nodes(graph):
    object_node_names = []
    num_of_eps = 0
    for node in graph.vs():
        if node['node_type'] == 'object':
            if node['name'] not in ["hand", "torso"]:
                object_node_names.append(node['name'])
        if node['node_type'] == 'spatial_relation':
            num_of_eps+=1

    return object_node_names

def worker_qsrs(chunk):
    (file, directory, which_setting) = chunk
    e = load_e(directory, file)
    print file

class sentance_featurespace:
    """
    class to store the histogram and language words
    """
    def __init__(self, histogram, words):
        self.histogram = histogram
        self.code_book = words
        self.graphlets = dict(zip(words, words))

def get_sentance_annotation(vid):
    """
    3 sentences describe the activity taking place, and 3 to describe the physical appearance of the person.
    (X) is a rejected sentence.
    """
    annotate_dir = "/home/"+getpass.getuser()+"/Datasets/ECAI_Data/dataset_activity_and_person_annotations/" + vid

    #read from files
    file = open(annotate_dir+"/activity.txt", 'r')
    wordstring = ""
    for line in file.readlines():
        if "#" in line: continue
        if "(X)" in line: continue
        line = line.replace(".", "").replace(",", "").replace("\\", " ").replace("/", " ").replace("!", "").replace("  ", " ")
        wordstring += line.lower()

    wordlist = wordstring.split()
    wordfreq = [wordlist.count(w) for w in wordlist]
    # print("Dictionary:\n" + str(dict(zip(wordlist,wordfreq))))
    return dict(zip(wordlist, wordfreq)).keys(), dict(zip(wordlist, wordfreq)).values()

if __name__ == "__main__":

    # create a QSRlib object if there isn't one already
    qsrlib = QSRlib()
    uuids, labels, wordids, wordcts = [], [], [], []
    global_codebook=np.empty([0,0])
    all_graphlets=[]
    path = directory = '/home/'+getpass.getuser()+'/Datasets/ECAI_Data/'

    # ****************************************************************************************************
    # static landmarks
    # ****************************************************************************************************
    static_objects = get_soma_objects()
    # static_objects = get_point_cloud_objects(os.path.join(path, "point_cloud_object_clusters"))
    # import pdb; pdb.set_trace()


    # ****************************************************************************************************
    # data directories
    # ****************************************************************************************************

    directory = os.path.join(path, 'dataset_segmented_15_12_16')
    qsr_dir = os.path.join(directory, "QSR_path")
    if not os.path.exists(qsr_dir): os.makedirs(qsr_dir)

    run = 0
    for file_ in sorted(os.listdir(qsr_dir)):
        it = file_.replace("run_", "")
        if int(it) >= run:
            run = int(it)
    qsr_path = os.path.join(qsr_dir, "run_%s" % str(run+1))
    if not os.path.exists(qsr_path): os.makedirs(qsr_path)
    print "qsr path >", qsr_path

    # ****************************************************************************************************
    # Dynamic Args list
    # ****************************************************************************************************

    data_subset = False
    frame_rate_reduce = 1     # drop every other frame - before median filter applies
    mean_window = 11          # use scipy medfilt with the window_size - after down sampling frame rate
    qsr_median_window = 5

    tpcc = True # run 42-48,49,51, 53
    # tpcc = False #run 50, 54

    objects_inc_type = True
    which_qsr=["argd", "qtcbs"] # run 42-48, 50
    # which_qsr=["qtcbs"]  # run49, run54
    # which_qsr=["argd"]  # run51, 52
    # which_qsr = None  # run53

    objects_used = ['left_hand', 'right_hand'] #, 'torso']

    qsrs_for = []
    for selected_joint in objects_used:
        qsrs_for.extend([(str(ob), selected_joint) for ob in static_objects])

    object_types = {'head': 'head', 'torso': 'torso', 'left_knee': 'knee', 'right_knee': 'knee',
                    'left_shoulder': 'shoulder', 'right_shoulder': 'shoulder',
                    # 'left_hand': 'left_hand', 'right_hand': 'right_hand'
                    'left_hand': 'hand', 'right_hand': 'hand'
                    }

    for ob in static_objects:
        if objects_inc_type:
            object_types[ob] = "_".join(ob.split("_")[:-1])
        else:
            object_types[ob] = "object"

    dynamic_args = { "qtcbs": {"qsrs_for" : qsrs_for, "no_collapse": True, "quantisation_factor":0.1, "validate":False },
                    "argd" : {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 0.25, 'Near': 0.5, 'Away': 1.0, 'Ignore': 10}},  #Run 38 (and 42)

                     # 2 Tests varying each of these values:
                    # "argd" : {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 0.15, 'Near': 0.5, 'Away': 1.0, 'Ignore': 10}},  # run 43
                    # "argd" : {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 0.4, 'Near': 0.5, 'Away': 1.0, 'Ignore': 10}},   # run 44
                    # "argd" : {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 0.25, 'Near': 0.35, 'Away': 1.0, 'Ignore': 10}}, # run 45
                    # "argd" : {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 0.25, 'Near': 0.75, 'Away': 1.0, 'Ignore': 10}},   # run 46
                    # "argd" : {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 0.25, 'Near': 0.5, 'Away': 0.75, 'Ignore': 10}},   # run47
                    # "argd" : {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 0.25, 'Near': 0.5, 'Away': 1.25, 'Ignore': 10}},     # run48

                    #  "argd" : {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 0.15, 'Near': 0.3, 'Ignore': 10}},
                    # max_rows is considered \rho  and max_eps is considered \eta
                     # "qstag" : {"params" : {"min_rows": 1, "max_rows": 2, "max_eps": 4, "frames_per_ep": 0, "split_qsrs": False}, "object_types": object_types},  # run 38 and 42
                     # "qstag" : {"params" : {"min_rows": 1, "max_rows": 1, "max_eps": 3, "frames_per_ep": 0, "split_qsrs": False}, "object_types": object_types},  #run55
                     # "qstag" : {"params" : {"min_rows": 1, "max_rows": 1, "max_eps": 5, "frames_per_ep": 0, "split_qsrs": False}, "object_types": object_types},  #run56
                     # "qstag" : {"params" : {"min_rows": 1, "max_rows": 2, "max_eps": 3, "frames_per_ep": 0, "split_qsrs": False}, "object_types": object_types},  #run57
                    "qstag" : {"params" : {"min_rows": 1, "max_rows": 3, "max_eps": 4, "frames_per_ep": 0, "split_qsrs": False}, "object_types": object_types},  #run58

                     "filters" : {"median_filter": {"window": qsr_median_window}}}

    # print "static objects used: ", static_objects.keys()


    # ****************************************************************************************************
    # Write out a file of arguments
    # ****************************************************************************************************
    with open(os.path.join(qsr_path, 'dynamic_args.txt'),'w') as f1:
        f1.write('dataset: %s \n \n' % directory)
        f1.write('subset of data: %s \n \n' % data_subset)
        f1.write('which_qsr: %s \n \n' % which_qsr)
        f1.write('objects_used: %s \n \n' % objects_used)
        f1.write('objects include types: %s \n \n' % objects_inc_type)
        f1.write('dynamic_args: \n \n' )
        for k, v in dynamic_args.items():
            f1.write('%s: %s \n \n' % (k, v))
        f1.write('frame_rate_reduce: %s \n \n' % frame_rate_reduce)
        f1.write('mean_window: %s \n \n' % mean_window)
        f1.write('qsr median window: %s \n \n' % qsr_median_window)

    if tpcc:
        objects_used_tpcc = ['left_hand', 'right_hand', 'left_shoulder', 'right_shoulder', 'left_knee', 'right_knee']
        # qsrs_for = [('head', 'torso', ob) if ob not in ['head', 'torso'] and ob != 'head-torso' else () for ob in object_types.keys()]
        qsrs_for_tpcc = [('head', 'torso', ob) for ob in objects_used_tpcc]
        object_types['head-torso'] = 'tpcc-plane'

        # dynamic_args["qstag"] = {"params" : {"min_rows": 1, "max_rows": 2, "max_eps": 4}, "object_types": object_types}
        dynamic_args['tpcc'] = {"qsrs_for": qsrs_for_tpcc}

        with open(os.path.join(qsr_path, 'dynamic_args.txt'), 'a') as f1:
            f1.write('which_qsr: "tpcc" \n \n')
            f1.write('objects_used: %s \n \n' % objects_used_tpcc)
            f1.write('dynamic_args: \n \n' )
            f1.write('%s: %s \n \n' % ('tpcc', dynamic_args['tpcc']))
            # for k, v in dynamic_args.items():
            #     f1.write('%s: %s \n \n' % (k, v))

    ### load file of dates vs clips
    videos_by_day = seg.segmented_videos()

    counter = 1
    # for counter, task in enumerate(sorted(os.listdir(directory))):
    for date in sorted(videos_by_day.keys()):
        print date

        for task in videos_by_day[date]:
            # if '196' in task or '211' in task: continue
            video = "%s.p" % task
            d_video = os.path.join(directory, task)

            # ****************************************************************************************************
            # open file
            # ****************************************************************************************************
            d_sk = os.path.join(d_video, 'skeleton')
            d_robot = os.path.join(d_video, 'robot')
            try:
                with open(os.path.join(d_video, 'label.txt')) as f:
                    for i, row in enumerate(f):
                        # print row
                        if i == 0:
                            (date, _id) = row.replace("\n", "").split("/")
                        if i == 1:
                            label = row
                    time, uuid = _id.replace("\r", "").split("_")
                print "> %s, %s, %s, %s, %s, %s" % (counter, task, date, time, uuid, label)
                counter+=1

            except IOError:
                print "no labels here: %s" % d_video
                continue

            # ****************************************************************************************************
            # read human pose and robot files
            # ****************************************************************************************************
            sk_files = [f for f in sorted(os.listdir(d_sk)) if os.path.isfile(os.path.join(d_sk, f))]
            r_files = [f for f in sorted(os.listdir(d_robot)) if os.path.isfile(os.path.join(d_robot,f))]

            # ****************************************************************************************************
            # skeleton and robot data
            # ****************************************************************************************************
            skeleton_data, robot_data = {}, {}
            for _file in sorted(sk_files):
                frame = int(_file.replace(".txt", ""))
                if frame % frame_rate_reduce != 0: continue

                skeleton_data[frame] = get_sk_info(open(os.path.join(d_sk, _file),'r'))   # old ECAI data format.
                robot_data[frame] = get_rob_info(open(os.path.join(d_robot,_file),'r'))

            # ****************************************************************************************************
            # filter skeleton data
            # ****************************************************************************************************
            f_skeleton_data, camera_world = apply_median_filter(skeleton_data, mean_window, vis=False)

            # ****************************************************************************************************
            # transform to map coordinate frame
            # ****************************************************************************************************
            ob_states={}
            map_world = World_Trace()
            map_skeleton_data = {}

            for cnt, (frame, sk_data) in enumerate(sorted(skeleton_data.items())):
                xr, yr, zr = robot_data[frame][0]
                yawr = robot_data[frame][1][2]
                pr = robot_data[frame][1][1]

                map_skeleton_data[frame] = {}

                for joint, (y,z,x,x2d,y2d) in sorted(f_skeleton_data[frame].items()):
                    # transformation from camera to map
                    rot_y = np.matrix([[np.cos(pr), 0, np.sin(pr)], [0, 1, 0], [-np.sin(pr), 0, np.cos(pr)]])
                    rot_z = np.matrix([[np.cos(yawr), -np.sin(yawr), 0], [np.sin(yawr), np.cos(yawr), 0], [0, 0, 1]])
                    rot = rot_z*rot_y

                    # robot's position in map frame
                    pos_r = np.matrix([[xr], [yr], [zr+1.66]])

                    # person's position in camera frame
                    pos_p = np.matrix([[x], [-y], [z]])

                    # person's position in map frame
                    map_pos = rot*pos_p+pos_r
                    x_mf, y_mf, z_mf = map_pos.flat
                    map_skeleton_data[frame][joint] = (x_mf, y_mf, z_mf)

                    # ****************************************************************************************
                    # create World Trace
                    # ****************************************************************************************
                    if joint not in ob_states.keys():
                        ob_states[joint] = [Object_State(name=joint, timestamp=cnt, x=x_mf, y=y_mf, z=z_mf)]
                    else:
                        ob_states[joint].append(Object_State(name=joint, timestamp=cnt, x=x_mf, y=y_mf, z=z_mf))

                # add objects to object states
                for ob, (x,y,z) in static_objects.items():
                    # print object, x,y,z
                    if ob not in ob_states.keys():
                        ob_states[ob] = [Object_State(name=str(ob), timestamp=cnt, x=x, y=y, z=z)]
                    else:
                        ob_states[ob].append(Object_State(name=str(ob), timestamp=cnt, x=x, y=y, z=z))

                # add the robot position
                if 'robot' not in ob_states.keys():
                    ob_states['robot'] = [Object_State(name='robot', timestamp=cnt, x=xr, y=yr, z=zr)]
                else:
                    ob_states['robot'].append(Object_State(name='robot', timestamp=cnt, x=xr, y=yr, z=zr))

            # add all the object states to the World Trace
            for k, object_state in ob_states.items():
                # print "added: %s object state" % k
                map_world.add_object_state_series(object_state)

            # ****************************************************************************************************
            # Call QSRLib
            # ****************************************************************************************************
            # f = open(qsr_path + "/WorldTraces/%s.p" % task, "w")
            # pickle.dump((map_world, camera_world), f, 2)

            if which_qsr != None:
                # print("\n>>",dynamic_args["qstag"])
                qsrlib_request_message = QSRlib_Request_Message(which_qsr, map_world, dynamic_args)
                map_response_message = qsrlib.request_qsrs(req_msg=qsrlib_request_message)


            # objects_in_graphs = set([])
            # for igraph in map_response_message.qstag.graphlets.graphlets.values():
            # #    print(igraph)
            #    for i in object_nodes(igraph):
            #         objects_in_graphs.add(i)

            # pretty_print_world_qsr_trace(which_qsr, map_response_message)
            # print(map_response_message.qstag.graphlets.code_book)
            # print(map_response_message.qstag.graphlets.histogram)
            # print len(map_response_message.qstag.graphlets.histogram)
            # # print objects_in_graphs
            # print map_response_message.qstag.graphlets.params
            # ****************************************************************************************************
            # add TPCC relations - from camera_world
            # ****************************************************************************************************
            if tpcc:
                req = QSRlib_Request_Message(which_qsr="tpcc", input_data=camera_world, dynamic_args=dynamic_args)
                camera_response_message = qsrlib.request_qsrs(req_msg=req)
                # pretty_print_world_qsr_trace("tpcc", camera_response_message)

                if which_qsr != None:
                    feature_spaces = [map_response_message.qstag.graphlets, camera_response_message.qstag.graphlets]
                else:
                    feature_spaces = [camera_response_message.qstag.graphlets]
            else:
                feature_spaces =[map_response_message.qstag.graphlets]

            # ****************************************************************************************************
            # Create sparse histogram and code words
            # ****************************************************************************************************
            histogram = np.array([0] * (global_codebook.shape[0]))
            for cnt, f in enumerate(feature_spaces):
                for freq, hash in zip(f.histogram, f.code_book):
                    hash_s = "{:20d}".format(hash).lstrip() # string

                    try:
                        ind = np.where(global_codebook == hash_s)[0][0]
                        histogram[ind] += freq
                    except IndexError:
                        global_codebook = np.append(global_codebook, hash_s)
                        histogram = np.append(histogram, freq)
                        all_graphlets = np.append(all_graphlets, f.graphlets[hash])

            uuids.append(uuid)
            labels.append(label)
            ids = np.nonzero(histogram)[0]
            # wordids.append(ids)
            # wordcts.append(histogram[ids])

            data_to_store = (uuid, label, ids, histogram[ids])
            f = open(qsr_path + "/%s.p" % task, "w")
            pickle.dump(data_to_store, f, 2)
            f.close()

    data_to_store = (global_codebook, all_graphlets, uuids, labels)
    f =open(qsr_path + "/codebook_data.p", "w")
    pickle.dump(data_to_store, f, 2)
    f.close()
