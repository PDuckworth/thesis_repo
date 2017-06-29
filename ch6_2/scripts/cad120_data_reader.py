# -*- coding: utf-8 -*-
"""
CAD120 data reader that is compatible with QSRlib.

:Author: Yiannis Gatsoulis <y.gatsoulis@leeds.ac.uk>
:Organization: University of Leeds
"""

from __future__ import print_function, division
import timeit
import ConfigParser
try:
    import cPickle as pickle
except ImportError:
    import pickle
import csv
import argparse
import sys
import os
import numpy as np
import scipy.signal
import getpass
# from utilities import *
from qsrlib_io.world_trace import *
from colorama import Fore, Back, Style
from qsrlib_io.world_qsr_trace import World_QSR_Trace
from qsrlib_io.world_trace import World_Trace, Object_State
from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
# from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
# from qsrlib_io.world_trace import Object_State, World_Trace

def print_success():
    print("success")
    #print("\t\t" + Fore.GREEN + "done" + Fore.RESET)

def print_fail():
    print("fail")
    #print("\t\t" + Fore.RED + "fail" + Fore.RESET)

def cprint(s, color, reset=Fore.RESET):
    print(s + color + reset)

def colorify(color, s, reset=Fore.RESET):
    return color + s + reset

def merge_world_qsr_traces(traces, qsr_type=None):
    """
    Merge a list of traces into one world_qsr_trace. It offers no protection versus overwriting previously
    existing relation.
    :param traces: list of World_QSR_Trace objects
    :param qsr_type: the qsr_type of the returned merged World_QSR_Trace object; if nothing given it is retrieved
    from the qsr_type of the first object in the traces list
    :return: a World_QSR_Trace that is the merge of all World_QSR_Trace objects in traces
    """
    if len(traces) == 0:
        raise ValueError("'traces' can't be of 0 length")
    if qsr_type is None:
        qsr_type = traces[0].qsr_type
    world_qsr_trace = World_QSR_Trace(qsr_type=qsr_type)
    for trace in traces:
        for t, s in zip(trace.trace.keys(), trace.trace.values()):
            for k, qsr in zip(s.qsrs.keys(), s.qsrs.values()):
                world_qsr_trace.add_qsr(qsr, t)
    return world_qsr_trace



### TESTS
def test_merge_world_qsr_traces():
    which_qsr = "rcc3_rectangle_bounding_boxes_2d"
    traj = [Object_State(name="traj", timestamp=0, x=1., y=1., width=5., length=8.),
            Object_State(name="traj", timestamp=1, x=1., y=2., width=5., length=8.)]
    o1 = [Object_State(name="o1", timestamp=0, x=11., y=1., width=5., length=8.),
          Object_State(name="o1", timestamp=1, x=11., y=2., width=5., length=8.)]
    o2 = [Object_State(name="o2", timestamp=0, x=11., y=1., width=5., length=8.),
          Object_State(name="o2", timestamp=1, x=11., y=2., width=5., length=8.)]
    world_trace1 = World_Trace()
    world_trace1.add_object_state_series(traj)
    world_trace1.add_object_state_series(o1)
    world_trace2 = World_Trace()
    world_trace2.add_object_state_series(traj)
    world_trace2.add_object_state_series(o2)
    world_traces = [world_trace1, world_trace2]

    qsrlib = QSRlib()

    world_qsr_traces = []
    for world_trace in world_traces:
        request_message = QSRlib_Request_Message(which_qsr=which_qsr, input_data=world_trace, include_missing_data=True)
        qsrlib_res = qsrlib.request_qsrs(request_message=request_message)
        world_qsr_traces.append(qsrlib_res.qsrs)

    world_qsr_trace = merge_world_qsr_traces(world_qsr_traces, which_qsr)

    print("Response is:")
    for t in world_qsr_trace.get_sorted_timestamps():
        foo = str(t) + ": "
        for k, v in zip(world_qsr_trace.trace[t].qsrs.keys(), world_qsr_trace.trace[t].qsrs.values()):
            foo += str(k) + ":" + str(v.qsr) + "; "
        print(foo)


class CAD120_Data_Reader(object):
    def __init__(self, config_filename="config.ini", skeleton_pass_filter=("H", "LH", "RH"),
                 load_from_files=False, read_tracks=True, which_qsrs = [], dynamic_args = {}, qsr_path=""):
        start = timeit.default_timer()
        print("\n--", self.__class__.__name__)
        print("Initializing...", end="")
        self.load_from_files = load_from_files
        self.read_tracks = read_tracks

        config_parser = ConfigParser.SafeConfigParser()
        if len(config_parser.read(config_filename)) == 0:
            raise ValueError("Config file not found, please provide a config.ini file as described in the documentation")
        config_section = "cad120_data_reader"
        try:
            self.corrected_labeling_path = config_parser.get(config_section, "corrected_labeling_path")
            self.tracks_path = config_parser.get(config_section, "path")
            self.sub_sequences_filename = config_parser.get(config_section, "sub_sequences_filename")
            self.sub_time_segmentation_filename = config_parser.get(config_section, "sub_time_segmentation_filename")
            self.ground_truth_tracks_filename = config_parser.get(config_section, "ground_truth_tracks_filename")
        except ConfigParser.NoOptionError:
            raise

        if skeleton_pass_filter == "all":
            self.skeleton_pass_filter = ['H', 'N', 'T', 'LS', 'LE', 'RS', 'RE', 'LHIP', 'LK', 'RHIP', 'RK', 'LH', 'RH',
                                         'LF', 'RF']
        else:
            self.skeleton_pass_filter = skeleton_pass_filter

        self.subjects_names_all = ["Subject1", "Subject3", "Subject4", "Subject5"]
        # self.subjects_names_all = ["Subject1"]

        self.super_names = ["arranging_objects", "cleaning_objects", "having_meal", "making_cereal",
                            "microwaving_food", "picking_objects", "stacking_objects", "taking_food",
                            "taking_medicine", "unstacking_objects"]
        # self.super_names = ["arranging_objects"]
        self.super_names = sorted(self.super_names)
        self.super_names_indexes = {}
        for i in range(len(self.super_names)):
            self.super_names_indexes[self.super_names[i]] = i

        self.sub_names = ["eating", "pouring", "opening", "placing", "reaching",
                          "moving", "cleaning", "closing", "null", "drinking"]
        self.sub_names = sorted(self.sub_names)
        self.sub_names_indexes = {}
        for i in range(len(self.sub_names)):
            self.sub_names_indexes[self.sub_names[i]] = i
        print_success()

        if not self.load_from_files:
            print(colorify(Fore.YELLOW, "Check that labeling.txt is read from the corrected version directory: '%s'" % self.corrected_labeling_path))

        # get the sequences of subactivities in a superactivity video; these are condensed (i.e. no repetitions,
        # hence, no self-loops
        if self.load_from_files and self.sub_sequences_filename != "":
            print("Loading sub-activities sequences from file...", end='')
            with open(self.sub_sequences_filename, "rb") as f:
                self.sub_sequences = pickle.load(f)
            print_success()
        else:
            print("Making sub-activities sequences from raw...", end='')
            self.sub_sequences = []
            self.__read_sub_seqs_csv()
            print_success()

        # get ground truth time segmentations of sub activities
        if self.load_from_files and self.sub_sequences_filename != "":
            print("Loading sub-activities time segmentations from file...", end='')
            with open(self.sub_time_segmentation_filename, "rb") as f:
                self.sub_time_segmentation = pickle.load(f)
            print_success()
        else:
            print("Making sub-activities time segmentations from raw...", end='')
            self.sub_time_segmentation = []
            self.__read_sub_times()
            print_success()

        # TODO should provide functions that search-return from self.sub_sequences, self.sub_time_segmentation, etc.

        # TODO should really be a different class
        # read QSRs
        # print("Reading training qsrs from files in path", root_path_qsrs, end="")
        # self.subject_super_vid_qsrs_seqs = {}
        # self.__read_subject_super_vid_qsrs_seqs()
        # print("\t\t\t" + Fore.GREEN + "success" + Fore.RESET)

        # TODO should actually be reading the raw trajectories also (which would then be passed to QSRlib)
        # TODO once raw trajectories are read I should provide an interface to QSRlib (or keep them in QSRlib format)

        # Updates for QSR generateion also:
        self.dynamic_args = dynamic_args
        self.which_qsrs = which_qsrs

        self.global_codebook = np.empty([0,0])
        self.all_graphlets=[]
        self.uuids, self.labels = [], [] #, self.wordids, self.wordcts = [], [], [], []
        self.qsr_path = qsr_path

        # read track traces
        if self.read_tracks:
            if self.load_from_files and self.ground_truth_tracks_filename != "":
                print("Loading tracks from file...", end='')
                with open(self.ground_truth_tracks_filename, "rb") as f:
                    self.world_traces = pickle.load(f)
                print_success()
            else:
                print("Making tracks from raw...", end='')
                self.world_traces2D = {}
                self.world_traces3D = {}
                self.read_ground_truth_trajectories()
                print_success()
        else:
            print(colorify(Fore.YELLOW, "Warning:"), "Was requested to skip tracks reading")

        stop = timeit.default_timer()
        print("Data loaded in: %.2f secs" % (stop - start))


    def __init_subject_super_vid_qsrs_seqs(self):
        for subject_name in self.subjects_names_all:
            self.subject_super_vid_qsrs_seqs[subject_name] = {}
            for super_name in self.super_names:
                self.subject_super_vid_qsrs_seqs[subject_name][super_name] = {}
                # for sub_activity_name in self.sub_names:
                #     self.subject_super_sub_vid_qsrs_raw[subject_name][super_activity_name][sub_activity_name] = {}

    # def __read_subject_super_vid_qsrs_seqs(self):
    #     self.__init_subject_super_vid_qsrs_seqs()
    #     filenames = os.listdir(self.root_path_qsrs)
    #     filenames.sort()
    #     for filename in filenames:
    #         data_file = {}
    #         # data_file = []
    #         fname = self.root_path_qsrs + filename
    #         filename_split = filename.split("-")
    #         filename_split[-1] = filename_split[-1].replace(".csv", "")
    #         subject_name = filename_split[0]
    #         super_activity_name = filename_split[1]
    #         video_id = filename_split[2]
    #         data_type = filename_split[3]
    #         with open(fname, "r") as f:
    #             for line in f:
    #                 line = line.rstrip("\n")
    #                 line = line.replace(" ", "")
    #                 line = line.split(",")
    #                 # data_file.append(line)
    #                 data_file[line[0]] = line[1:]
    #         try:
    #             self.subject_super_vid_qsrs_seqs[subject_name][super_activity_name][video_id][data_type] = data_file
    #         except KeyError:
    #             self.subject_super_vid_qsrs_seqs[subject_name][super_activity_name][video_id] = {}
    #             self.subject_super_vid_qsrs_seqs[subject_name][super_activity_name][video_id][data_type] = data_file

    def __read_sub_seqs_csv(self, filename="temp_superactivities_subactivities_data.csv"):
        self.__rewrite_sub_seqs_csv(filename)
        with open(filename, "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                # self.sub_sequences.append(row)
                v = {'subject_name': str(row[0]),
                     'super_name': str(row[1]),
                     'video_name': str(row[2]),
                     'sub_seq': row[3:]}
                # k = str(row[0]) + '_' + str(row[1]) + '_' + str(row[2])
                self.sub_sequences.append(v)
        try:
            os.remove(filename)
        except OSError:
            pass

    def __rewrite_sub_seqs_csv(self, write_filename):
        text = ""
        for subject_name in self.subjects_names_all:
            subject_dir = subject_name + "_annotations"
            for super_activity_name in self.super_names:
                # filename = self.corrected_labeling_path + "annotations/" + subject_dir + super_activity_name + "/" + "labeling.txt"
                filename = os.path.join(self.corrected_labeling_path, "annotations", subject_dir, super_activity_name, "labeling.txt")
                try:
                    with open(filename, "r") as f:
                        previous_video_id = ""
                        for line in f:
                            line = line.rstrip("\n")
                            line = line.replace(" ", "")
                            line = line.split(",")
                            video_id = line[0]
                            sub_activity_name = line[3]

                            if video_id != previous_video_id:
                                text += "\n" + subject_name + "," + super_activity_name + "," + video_id + "," + sub_activity_name
                            else:
                                text += "," + sub_activity_name
                            previous_video_id = video_id
                except IOError:
                    print("file not found:", filename)
        text = text[1:]
        with open(write_filename, "w") as f:
            # print("rewriting...")
            f.write(text)


    def __read_sub_times(self):
        # # if needed to be read from file for faster loading, but does not populate self.sub_times_raw
        # with open('sub_activities_time_segmentation.pickle', 'rb') as handle:
        #     sub_time_segmentation = pickle.load(handle)

        # reconstructed from cad120 datafiles
        sub_time_segmentation = {}
        for subject_name in self.subjects_names_all:
            subject_dir = subject_name + "_annotations/"
            sub_time_segmentation[subject_name] = {}
            for super_activity_name in self.super_names:
                # filename = self.corrected_labeling_path + "annotations/" + subject_dir + super_activity_name + "/" + "labeling.txt"
                filename = os.path.join(self.corrected_labeling_path, "annotations", subject_dir, super_activity_name, "labeling.txt")
                sub_time_segmentation[subject_name][super_activity_name] = {}
                try:
                    with open(filename, "r") as f:
                        for line in f:
                            line = line.rstrip("\n")
                            line = line.replace(" ", "")
                            line = line.split(",")

                            video_id = line[0]
                            start_frame = int(line[1])
                            end_frame = int(line[2])
                            duration_frames = end_frame - start_frame + 1
                            sub_activity_name = line[3]

                            d2 = {"sub_activity": sub_activity_name, "start_frame": start_frame,
                                  "end_frame": end_frame, "duration_frames": duration_frames}
                            try:
                                sub_time_segmentation[subject_name][super_activity_name][video_id].append(d2)
                            except KeyError:
                                sub_time_segmentation[subject_name][super_activity_name][video_id] = []
                                sub_time_segmentation[subject_name][super_activity_name][video_id].append(d2)
                except IOError:
                    print("file not found:", filename)
        # end of needed to be reconstructed from cad120 datafiles

        self.sub_time_segmentation[:] = []
        for subject_name, v1 in zip(sub_time_segmentation.keys(), sub_time_segmentation.values()):
            for super_name, v2 in zip(v1.keys(), v1.values()):
                for video_name, durations in zip(v2.keys(), v2.values()):
                    self.sub_time_segmentation.append({'subject_name': subject_name, 'super_name': super_name,
                                                       'video_name': video_name, 'durations': durations})

    def __make_fname(self, filename, path, ext, prefix, postfix):
        fname = path
        if fname[-1] != "/":
            fname += "/"
        if prefix != "":
            fname += prefix + "_"
        fname += filename
        if postfix != "":
            fname += "_" + postfix
        if ext[0] != ".":
            fname += "."
        fname += ext
        return fname

    def read_ground_truth_trajectories(self):

        qsrlib = QSRlib()

        world_traces2D = {}
        world_traces3D = {}

        labels_file = "activityLabel.txt"
        for subject_name in self.subjects_names_all:
            subject_dir = os.path.join(self.tracks_path, "annotations", str(subject_name + "_annotations"))
            for super_name in self.super_names:
                act_dir = os.path.join(subject_dir, super_name)
                video_ids = []
                with open(os.path.join(act_dir, labels_file)) as f:
                    foo = f.readlines()
                    for line in foo:
                        line = line.strip()
                        fields = line.split(',')
                        video_ids.append(fields[0])

                # Get object data
                for video_id in video_ids:
                    world_trace_description = subject_name + "_" + super_name + "_" + str(video_id)
                    world_trace = World_Trace(description=world_trace_description)

                    frame_data = self.get_objects_annotation_data(act_dir, video_id)

                    world_trace2D = self.object_frame_data_to_qsrlib_world_trace(world_trace, frame_data)
                    world_trace3D = self.object_frame_data_to_qsrlib_world_trace3D(world_trace, frame_data)

                    # Get skeleton data
                    skeleton_file = os.path.join(act_dir, video_id + '.txt')
                    joints2D, joints3D = self.parse_skeleton_data(skeleton_file)

                    world_trace2D = self.skeleton_frame_data_to_qsrlib_world_trace(world_trace2D, joints2D)
                    world_trace3D = self.skeleton_frame_data_to_qsrlib_world_trace_3D(world_trace3D, joints3D)

                    world_traces2D[world_trace_description] = world_trace2D
                    world_traces3D[world_trace_description] = world_trace3D


                    qsrlib_request_message = QSRlib_Request_Message(self.which_qsrs, world_trace2D, self.dynamic_args)
                    response_message2D = qsrlib.request_qsrs(req_msg=qsrlib_request_message)

                    qsrlib_request_message = QSRlib_Request_Message("tpcc", world_trace2D, self.dynamic_args)
                    response_messagetpcc = qsrlib.request_qsrs(req_msg=qsrlib_request_message)

                    # Create sparse histogram and code words
                    # ****************************************************************************************************
                    feature_spaces = [response_message2D.qstag.graphlets, response_messagetpcc.qstag.graphlets] #, camera_response_message.qstag.graphlets]
                    histogram = np.array([0] * (self.global_codebook.shape[0]))

                    for cnt, f in enumerate(feature_spaces):
                        # import pdb; pdb.set_trace()
                        for freq, hash in zip(f.histogram, f.code_book):
                            hash_s = "{:20d}".format(hash).lstrip() # string
                            try:
                                ind = np.where(self.global_codebook == hash_s)[0][0]
                                histogram[ind] += freq
                            except IndexError:
                                self.global_codebook = np.append(self.global_codebook, hash_s)
                                histogram = np.append(histogram, freq)
                                self.all_graphlets = np.append(self.all_graphlets, f.graphlets[hash])

                    self.uuids.append(video_id)
                    self.labels.append(super_name)
                    ids = np.nonzero(histogram)[0]
                    # wordids.append(ids)
                    # wordcts.append(histogram[ids])
                    f = open(self.qsr_path + "/%s.p" % video_id, "w")
                    pickle.dump((video_id, super_name, ids, histogram[ids]), f, 2)
                    f.close()

        # print(joints2D[1])
        # print(type(joints2D))
        self.world_traces2D = world_traces2D
        self.world_traces3D = world_traces3D

        f =open(self.qsr_path + "/codebook_data.p", "w")
        pickle.dump((self.global_codebook, self.all_graphlets, self.uuids, self.labels), f, 2)
        f.close()
        print("QSRs all stored: %s" % self.qsr_path)


    def object_frame_data_to_qsrlib_world_trace(self, world_trace, frame_data):
        ts = sorted(frame_data.keys())
        for t in ts:
            for object_name, bbox in zip(frame_data[t].keys(), frame_data[t].values()):
                xc, yc, w, l = self.bbox_to_center_lw(bbox=bbox)
                abstract_name = "_".join(object_name.split("_")[:-1])
                if abstract_name == "unknown": abstract_name+= "_obj"
                object_state = Object_State(name=abstract_name, timestamp=str(t), x=xc, y=yc, length=l, width=w, category="object")
                world_trace.add_object_state(object_state=object_state)
        return world_trace

    def object_frame_data_to_qsrlib_world_trace3D(self, world_trace, frame_data):
        ts = sorted(frame_data.keys())
        for t in ts:
            for object_name, bbox in zip(frame_data[t].keys(), frame_data[t].values()):
                xc, yc, w, l = self.bbox_to_center_lw(bbox=bbox)
                abstract_name = "_".join(object_name.split("_")[:-1])
                if abstract_name == "unknown": abstract_name+= "_obj"
                object_state = Object_State(name=abstract_name, timestamp=str(t), x=xc, y=yc, length=l, width=w, category="object")
                world_trace.add_object_state(object_state=object_state)
        return world_trace

    def skeleton_frame_data_to_qsrlib_world_trace(self, world_trace, frame_data):
        ts = sorted(frame_data.keys())
        for t in ts:
            for joint_name, coords in zip(frame_data[t].keys(), frame_data[t].values()):
                if joint_name in self.skeleton_pass_filter:
                    xc = coords[0]
                    yc = coords[1]
                    object_state = Object_State(name=joint_name, timestamp=str(t), x=xc, y=yc, category="joint")
                    world_trace.add_object_state(object_state=object_state)
        return world_trace

    def skeleton_frame_data_to_qsrlib_world_trace_3D(self, world_trace, frame_data):
        ts = sorted(frame_data.keys())
        for t in ts:
            for joint_name, coords in zip(frame_data[t].keys(), frame_data[t].values()):
                if joint_name in self.skeleton_pass_filter:
                    xc = coords[0]
                    yc = coords[1]
                    zc = coords[2]
                    object_state = Object_State(name=joint_name, timestamp=str(t), x=xc, y=yc, z=zc, category="joint")
                    world_trace.add_object_state(object_state=object_state)
        return world_trace

    def bbox_to_center_lw(self, bbox):
        w = float(bbox[2] - bbox[0])
        l = float(bbox[3] - bbox[1])
        xc = float(bbox[0] + w/2.0)
        yc = float(bbox[3] + l/2.0)
        return xc, yc, w, l

    def get_objects_annotation_data(self, obj_annotation_dir, activity_id, start_frame=None, end_frame=None):
        frame_data = {}
        obj_types = {}
        obj_info_index = 3

        # Get the dict of types of objects, so we can use the type of the obj in this file
        # TODO Need to investigate what the following actually means? How can there be unknown in ground truth data?
        # Use 'unknown' if object type is not known
        activitylabel_file = open(os.path.join(obj_annotation_dir, 'activityLabel.txt'))
        for line in activitylabel_file:
            line = line.strip()
            fields = line.split(',')
            if activity_id == fields[0]:
                # Skip the last element as it is just empty string ''
                for obj_info in fields[obj_info_index:-1]:
                    obj_types[int(obj_info.split(':')[0])] = obj_info.split(':')[1]

        for ifile in os.listdir(obj_annotation_dir):
            if activity_id + '_obj' not in ifile:
                continue
            obj_annotation_file = open(os.path.join(obj_annotation_dir, ifile))
            for line in obj_annotation_file:
                if line == '\n':
                    break
                line = line.strip(',\n')
                fields = line.split(',')
                if len(fields) != 12:
                    continue
                int_fields = map(int, fields[:6])
                (frame, obj_id, minx, miny, maxx, maxy) = int_fields

                if start_frame != None and frame < start_frame:
                    continue
                if end_frame != None   and frame > end_frame:
                    break

                STIP_diff_fields = map(float, fields[6:])
                obj_id_str = obj_types.get(obj_id,'unknown') + '_' + repr(obj_id)

                if frame not in frame_data:
                    frame_data[frame] = {}

                first_frame = (frame-start_frame)+1 if start_frame != None else 1

                #If first frame is missing - don't add anything to the obj or frame vectors
                if first_frame ==1 and (minx, miny, maxx, maxy) == (0, 0, 0, 0):
                    #frame_data[frame][obj_id_str] = (minx, miny, maxx, maxy) # this was commented out from the original
                    last_pos_frame = 1

                #Add object detections which are currently in the scene
                if (minx, miny, maxx, maxy) != (0, 0, 0, 0):
                    frame_data[frame][obj_id_str] = (minx, miny, maxx, maxy)
                    last_pos_frame = frame

                #Keep the previous position of detected objects if they become occluded
                #May have a bug if frame 1 has obj data, but frame 2 does not
                elif (minx, miny, maxx, maxy) == (0, 0, 0, 0) and last_pos_frame != 1:
                    frame_data[frame][obj_id_str] = frame_data[last_pos_frame][obj_id_str]

                #If first frame, or all frames, have no object data
                elif (minx, miny, maxx, maxy) == (0, 0, 0, 0) and last_pos_frame == 1:
                    pass

            obj_annotation_file.close()
        return frame_data

    # Joint number -> Joint name
    #  1 -> HEAD
    #  2 -> NECK
    #  3 -> TORSO
    #  4 -> LEFT_SHOULDER
    #  5 -> LEFT_ELBOW
    #  6 -> RIGHT_SHOULDER
    #  7 -> RIGHT_ELBOW
    #  8 -> LEFT_HIP
    #  9 -> LEFT_KNEE
    # 10 -> RIGHT_HIP
    # 11 -> RIGHT_KNEE
    # 12 -> LEFT_HAND
    # 13 -> RIGHT_HAND
    # 14 -> LEFT_FOOT
    # 15 -> RIGHT_FOOT
    def parse_skeleton_data(self, skeleton_file, start_frame=None, end_frame=None):
        # Parse skeleton data and return 2D and 3D joints dictionary
        TOTAL_JOINTS = 15
        joints_enum = {1:'H',
                       2:'N',
                       3:'T',
                       4:'LS',
                       5:'LE',
                       6:'RS',
                       7:'RE',
                       8:'LHIP',
                       9:'LK',
                      10:'RHIP',
                      11:'RK',
                      12:'LH',
                      13:'RH',
                      14:'LF',
                      15:'RF',
                     }

        joints_dict = {'H':1,
                       'N':2,
                       'T':3,
                       'LS':4,
                       'LE':5,
                       'RS':6,
                       'RE':7,
                       'LHIP':8,
                       'LK':9,
                       'RHIP':10,
                       'RK':11,
                       'LH':12,
                       'RH':13,
                       'LF':14,
                       'RF':15,
                     }
        joints_dict = { 'H':1,
                        'T':3,
                       'LH':12,
                       'RH':13,
                     }

        joints3D = {}
        joints2D = {}

        # Get skeleton data
        skeleton_file_pointer = open(skeleton_file)
        for line in skeleton_file_pointer:
            if 'END' in line:
                break
            line = line.strip(',\n')
            fields = line.split(',')
            fields = map(float, fields)
            frame = int(fields[0])

            if start_frame != None and frame < start_frame:
                continue
            if end_frame != None   and frame > end_frame:
                break

            joints3D[frame] = attrdict(joints_dict)
            joints2D[frame] = attrdict(joints_dict)
            position = 1
            for i in range(1, TOTAL_JOINTS+1):

                if joints_enum[i] not in joints_dict.keys(): continue

                joints3D[frame][joints_enum[i]] = {}
                joints2D[frame][joints_enum[i]] = {}
                # The last value in the tuple is the confidence
                if i <= 11:
                    # The last four joints have no orientation
                    position += 10
                joints3D[frame][joints_enum[i]] = np.array(fields[position:position+3])
                (x,y,z) = joints3D[frame][joints_enum[i]]
                # Got these
                x_2D = 156.8584456124928 + 0.0976862095248 * x * 2 - 0.0006444357104 * y * 3 + 0.0015715946682 * z
                y_2D = 125.5357201011431 + 0.0002153447766 * x - 0.1184874093530 * y - 0.0022134485957 * z
                joints2D[frame][joints_enum[i]] = np.array((x_2D, y_2D))
                # Ignore the confidence
                position += 4
        skeleton_file_pointer.close()
        return (joints2D, joints3D)

    def save(self, filenames=None):
        if not filenames:
            filenames = {"sub_sequences_filename": self.sub_sequences_filename,
                         "sub_time_segmentation_filename": self.sub_time_segmentation_filename,
                         "ground_truth_tracks_filename": self.ground_truth_tracks_filename}
        print("Saving...")

        filename = filenames["sub_sequences_filename"]
        print("sub-activities sequences to " + filename, end="")
        with open(filename, "wb") as f:
            pickle.dump(self.sub_sequences, f)
        print_success()

        filename = filenames["sub_time_segmentation_filename"]
        print("sub-activities time segmentation to " + filename, end="")
        with open(filename, "wb") as f:
            pickle.dump(self.sub_time_segmentation, f)
        print_success()

        if self.read_tracks:
            filename = filenames["ground_truth_tracks_filename"]
            print("tracks to " + filename, end="")
            with open(filename, "wb") as f:
                pickle.dump(self.world_traces2D, f)
            print_success()
        else:
            print(colorify(Fore.YELLOW, "Warning:", "Not saving tracks as it was requested before not to be read"))

class attrdict(dict):
    """ Dictionary with attribute like access """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

def debugging(reader):
    ## DEBUGGING
    print("sub_seqs:", reader.sub_sequences)
    print("sub_tsegs:", reader.sub_time_segmentation)
    print("gt_tracks:", reader.world_traces2D.keys())
    print(reader.world_traces2D["Subject4_unstacking_objects_1130151154"].trace.keys())

    print(len(reader.world_traces2D))
    print(reader.world_traces2D.keys())
    for v in reader.world_traces2D.values():
        print(v.trace[1].objects.keys())

    world_traces2D = reader.world_traces2D
    ks = world_traces2D.keys()
    print([s for s in ks if "Subject5" in s and "arranging_objects" in s])
    print(len(sorted(world_traces2D['Subject5_arranging_objects_0504235908'].trace.keys())))
    print(world_traces2D['Subject5_arranging_objects_0504235908'].trace[1].objects["LH"].kwargs["category"])

    world_traces3D = reader.world_traces3D
    ks = world_traces2D.keys()
    print([s for s in ks if "Subject5" in s and "arranging_objects" in s])
    print(len(sorted(world_traces3D['Subject5_arranging_objects_0504235908'].trace.keys())))
    print(world_traces3D['Subject5_arranging_objects_0504235908'].trace[1].objects["LH"].kwargs["category"])
    import pdb; pdb.set_trace()

    joint = world_traces3D['Subject5_arranging_objects_0504235908'].trace[1].objects["LH"]
    print(joint.x, joint.y, joint.z)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CAD120 data reader in QSRlib format")
    parser.add_argument("-l", "--load", dest="load", action="store_true", help="load the data from the files in 'config.ini'")
    args = parser.parse_args()

    # which_qsr = ["argd", "qtcbs"]
    which_qsr = ["argd", "qtcbs"]

    qsrs_for_tpcc = [('H', 'T', ob) for ob in ['box', 'bowl', 'milk', 'plate', 'medcinebox', 'cup', 'microwave', 'cloth', 'unknown']]

    dynamic_args = { "qtcbs": {"no_collapse": True, "quantisation_factor":0.1, "validate":False}, #"qsrs_for" : qsrs_for},
                     "argd" : {"qsr_relations_and_values": {'Touch': 150, 'Near':400, 'Away': 800, 'Ignore': 1000}},#, "qsrs_for": qsrs_for },
                    #"qstag" : {"params" : {"min_rows": 1, "max_rows": 2, "max_eps": 4}, "object_types": object_types},
                     "qstag" : {"params" : {"min_rows": 1, "max_rows": 1, "max_eps": 3, "frames_per_ep": 0, "split_qsrs": True}},#, "object_types": object_types},
                     "filters" : {"median_filter": {"window": 5}},
                     'tpcc' : {"qsrs_for": qsrs_for_tpcc}
                     }

    qsr_path = "/home/" + getpass.getuser() + "/Datasets/CAD120/qsr_data"
    if not os.path.isdir(qsr_path): os.system('mkdir -p ' + qsr_path)

    reader = CAD120_Data_Reader(config_filename="/home/scpd/Datasets/CAD120/config.ini", which_qsrs = which_qsr, dynamic_args = dynamic_args, qsr_path = qsr_path)
    #reader.save()
    # debugging(reader)
