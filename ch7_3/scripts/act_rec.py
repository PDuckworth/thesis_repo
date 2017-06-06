#! /usr/bin/env python
__author__ = 'p_duckworth'
import roslib
import sys, os
import rospy
import yaml
import actionlib
import rosbag
import getpass, datetime
import shutil

# from mongodb_store.message_store import MessageStoreProxy

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from tf.transformations import euler_from_quaternion
import math
import numpy as np

from qsrlib_io.world_trace import Object_State, World_Trace
from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_qsr_trace import World_QSR_Trace
# from qsrlib_utils.utils import merge_world_qsr_traces
# from qsrlib_qstag.qstag import Activity_Graph
# from qsrlib_qstag.utils import *
import copy
import cv2
import colorsys
import operator

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Header
# from soma_msgs.msg import SOMAObject, SOMAROIObject
# from soma_manager.srv import *
# from shapely.geometry import Polygon, Point
import cPickle as pickle
import time

import ch7_3.skeleton_manager as sk
import ch6_2.ECAI_videos_segmented_by_day as seg
from online_activity_recognition.msg import recogniseAction, recogniseActionResult, skeleton_message


class act_rec_server(object):
    def __init__(self, run=29):

        """
        Currently uses segmented data - change this when produced QSRs for non segmented vidoes
        """
        dataset_path = "/home/" + getpass.getuser() + "/Datasets/ECAI_Data"

        video_filepath = os.path.join(dataset_path, "dataset_segmented_15_12_16")
        self.sk_publisher = sk.SkeletonManager(video_filepath)

        self.run = "run_%s" % run
        topicfilepath = os.path.join(video_filepath, 'QSR_path', self.run)
        print topicfilepath

        self.load_all_topic_files(topicfilepath)

        self.objects = self.get_soma_objects()
        # self.objects = self.get_point_cloud_objects(path)

        # online window of QSTAGS
        #self.windows_size = 150
        #self.th = 4         # column thickness
        #self.th2 = 4        # frame thickness
        #self.th3 = 4        # thickness between images
        #self.th_100 = 500   # how big is the 100%
        self.qsr_median_window = 1   # smooth the QSR relations

        self.axes_th = 101
        self.space_th = 2
        self.num_topics = len(self.actions_vectors) #stupid name!
        self.time = 500
        self.data = {}
        self.colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [170, 255, 0], [0, 255, 170],
        [255, 170, 0], [0, 170, 255], [255, 0, 170], [170, 0, 255]] # note BGR ...
        self._create_image()

        self.online_window = {}
        self.online_window_img = {}
        self.act_results = {}

        self.image_pub = rospy.Publisher("/activity_recognition_results", Image, queue_size=10)
        self.image_label = cv2.imread(dataset_path+'/image_label.png')

        self.bridge = CvBridge()
        self.qsr_median_window = 3

    def _create_image(self):
        #creating the image
        self.img = np.zeros((self.axes_th*self.num_topics + self.space_th*(self.num_topics-1), self.time, 3),dtype=np.uint8)+30
        for i in range(self.num_topics-1):
            self.img[self.space_th*i+self.axes_th*(i+1):(self.space_th+self.axes_th)*(i+1), :, :] = 255
        # add grid on image
        for i in range(self.num_topics):
            for j in [20,40,60,80]:
                self.img[(self.space_th+self.axes_th)*i+j:(self.space_th+self.axes_th)*i+j+1, :3, :] = 255
                self.img[(self.space_th+self.axes_th)*i+j:(self.space_th+self.axes_th)*i+j+1, self.time-3:, :] = 255

    def main(self):
        self.online_window = {}
        self.online_window_img = {}
        self.act_results = {}

        self.pred_topic = []
        self.true_labels = []

        print "running offline..."
        videos_by_day = seg.segmented_videos()
        self.all_labels = []
        counter = 1
        # for counter, task in enumerate(sorted(os.listdir(directory))):
        for date in sorted(videos_by_day.keys()):
            print date

            for video in videos_by_day[date]:
                print "video:", video

                if '196' in video or '211' in video: continue

                # d_video = os.path.join(directory, task)
                self.sk_publisher.run_offline_instead_of_callback(video)

                self.all_labels.append(self.sk_publisher.label)

                self.convert_to_map()
                self.get_world_frame_trace()
                self.update_online_window()
                self.recognise_activities()
                self.plot_online_window()


            true_labels  = self.true_labels
            pred_labels  = self.pred_labels
            import pdb; pdb.set_trace()

            print "k: %s. v-measure: %0.3f. homo: %0.3f. comp: %0.3f. MI: %0.3f. NMI: %0.3f. "  \
              %(self.num_topics, metrics.v_measure_score(true_labels, pred_labels), metrics.homogeneity_score(true_labels, pred_labels),
                metrics.completeness_score(true_labels, pred_labels), metrics.mutual_info_score(true_labels, pred_labels),
                metrics.normalized_mutual_info_score(true_labels, pred_labels))
            print ">> ended\n"


    #########################################################################
    def update_online_window(self):
        for subj in self.subj_world_trace:
            # initiate the window of QSTAGS for this person
            if subj not in self.online_window:
                self.online_window[subj] = np.zeros((self.time, len(self.code_book)), dtype=np.uint8)
                self.online_window_img[subj] = self.img #np.zeros((len(self.code_book)*self.th,self.windows_size*self.th2,3),dtype=np.uint8)+255
            else:  #shifts by one frame
                self.online_window[subj][:self.time-1] = self.online_window[subj][1:]
                #self.online_window_img[subj][:,self.th2:self.windows_size*self.th2,:] = self.online_window_img[subj][:,0:self.windows_size*self.th2-self.th2,:]
            # find which QSTAGS happened in this frame
            ret = self.subj_world_trace[subj]
            self.online_window[subj][-1,:] = 0
            #self.online_window_img[subj][:, 0:self.th2, :] = 255
            for cnt, h in zip(ret.qstag.graphlets.histogram, ret.qstag.graphlets.code_book):
                oss, ss, ts  = nodes(ret.qstag.graphlets.graphlets[h])
                ssl = [d.values() for d in ss]
                # import pdb; pdb.set_trace()

                #if "Microwave" in oos:
                #print ">>>", cnt, h, type(h) #, #oss, ssl #ret.qstag.graphlets.graphlets[h]
                if isinstance(h, int):
                    h = "{:20d}".format(h).lstrip()
                #print cnt, h, type(h), len(h), len(self.code_book), len(self.code_book[0]) #, self.code_book.shape
                #code_book = [str(i) for i in self.code_book]

                if h in self.code_book:  # Futures WARNING here
                    index = list(self.code_book).index(h)
                    self.online_window[subj][-1, index] = 1

                    #self.online_window_img[subj][index*self.th:index*self.th+self.th, 0:self.th2, :] = 10

    #########################################################################
    def recognise_activities(self):
        self.act_results = {}
        # print "\n>>what happens here?"

        for subj, window in self.online_window.items():
            # compressing the different windows to be processed
            for w in range(2,10,2):
                for i in range(self.time-w):
                    compressed_window = copy.deepcopy(window[i,:])
                    for j in range(1,w+1):
                        compressed_window += window[j+i,:]

                    compressed_window = [1 if o !=0 else 0 for o in compressed_window]
                    # compressed_window /= compressed_window

                    # comparing the processed windows with the different actions
                    if subj not in self.act_results:
                        self.act_results[subj] = {}
                    for act in self.actions_vectors:
                        if act not in self.act_results[subj]:
                            self.act_results[subj][act] = np.zeros((self.time), dtype=np.float32)

                        result = np.sum(compressed_window*self.actions_vectors[act])
                        if result != 0:
                            self.act_results[subj][act][i:i+w] += result
                        # if act==2:
                        #     self.act_results[subj][act][i:i+w] += 20
        # calibration
        for subj in self.act_results:
            for act in self.act_results[subj]:
                self.act_results[subj][act] /= 20

        # create a classification
        for subj in self.act_results:
            max_dist = 0
            max_act = 100
            for frame in xrange(self.time):
                for act, dist in self.act_results[subj].items():
                    if dist[frame] > max_dist:
                        max_dist = dist[frame]
                        max_act = act
                if max_act !=100:
                    self.true_labels.append(self.sk_publisher.label)
                    self.pred_topic.append(max_act)


    #########################################################################
    def plot_online_window(self):
        #if len(self.online_window_img) == 0:
        final_img = self.img #

        for counter, subj in enumerate(self.online_window_img):
            img1 = self._update_image(self.act_results[subj])
            if counter==0:
                final_img = img1
            else:
                final_img = np.concatenate((final_img, img1), axis=1)

        #try:
        #print ""
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(final_img, "bgr8"))

        #except CvBridgeError as e:
        #    print(e)

    #########################################################################
    def _update_image(self,data):
        img = self.img.copy()
        #print data
        for i in range(self.num_topics):
            for t,k in enumerate(data[i][self.time*3/4:]):
                a = (i+1)*(self.space_th+self.axes_th)
                #print a,k,t,i
                #print np.mod(i,len(self.colors))
                img[a-int(k):a, 4*t:4*t+4, :] = self.colors[np.mod(i,len(self.colors))]
        return img


    def load_all_topic_files(self, path):
        print "loading topics..."

        with open(path +"/TopicData/code_book.p", 'r') as f:
            self.code_book = pickle.load(f)

        with open(path +"/TopicData/graphlets.p", 'r') as f:
            self.graphlets = pickle.load(f)

        codebook_lengh = len(self.code_book)
        print "codebook:", len(self.code_book), type(self.code_book[0])

        # for code in self.code_book:
            # print code, self.graphlets[code]

        self.actions_vectors = {}
        with open(path + "/TopicData/topic_words.p", 'r') as f:
            VT = pickle.load(f)
            print "Topics:", VT.shape

        for count, act in enumerate(VT):
            self.actions_vectors[count] = act
            # p_sum = sum(x for x in act if x > 0)    # sum of positive graphlets
            # self.actions_vectors[count] = act/p_sum*100
            # self.actions_vectors[count][self.actions_vectors[count]<0] = 0


    def get_object_frame_qsrs(self, world_trace):
        joint_types = {'left_hand': 'hand', 'right_hand': 'hand',  'head-torso': 'tpcc-plane'}

        joint_types_plus_objects = joint_types.copy()
        for object in self.objects:
            generic_object = "_".join(object.split("_")[:-1])
            joint_types_plus_objects[object] = generic_object
        #print joint_types_plus_objects

        """create QSRs between the person's joints and the soma objects in map frame"""
        qsrs_for=[]
        for ob in self.objects:
            qsrs_for.append((str(ob), 'left_hand'))
            qsrs_for.append((str(ob), 'right_hand'))
            #qsrs_for.append((str(ob), 'torso'))

        dynamic_args = {}
        # dynamic_args['argd'] = {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 0.25, 'Near': 0.5,  'Ignore': 10}}
        # dynamic_args['qtcbs'] = {"qsrs_for": qsrs_for, "quantisation_factor": 0.05, "validate": False, "no_collapse": True} # Quant factor is effected by filters to frame rate
        # dynamic_args["qstag"] = {"object_types": joint_types_plus_objects, "params": {"min_rows": 1, "max_rows": 1, "max_eps": 2}}

        dynamic_args['argd'] = {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 0.25, 'Near': 0.5, 'Away': 1.0, 'Ignore': 10}}
        #dynamic_args['argd'] = {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 1.5, 'Ignore': 10}}
        dynamic_args['qtcbs'] = {"qsrs_for": qsrs_for, "quantisation_factor": 0.01, "validate": False, "no_collapse": True} # Quant factor is effected by filters to frame rate
        dynamic_args["qstag"] = {"object_types": joint_types_plus_objects, "params": {"min_rows": 1, "max_rows": 1, "max_eps": 4}}
        dynamic_args["filters"] = {"median_filter": {"window": self.qsr_median_window}}

        qsrlib = QSRlib()
        req = QSRlib_Request_Message(which_qsr=["argd", "qtcbs"], input_data=world_trace, dynamic_args=dynamic_args)
        #req = QSRlib_Request_Message(which_qsr="argd", input_data=world_trace, dynamic_args=dynamic_args)
        ret = qsrlib.request_qsrs(req_msg=req)

        #print "\n"
        #for ep in ret.qstag.episodes:
        #    print ep
        return ret


    def get_world_frame_trace(self):
        """Accepts a dictionary of world (soma) objects.
        Adds the position of the object at each timepoint into the World Trace"""
        self.subj_world_trace = {}

        for subj in self.skeleton_map:
            #print ">", len(self.skeleton_map[subj]["left_hand"])
            ob_states={}
            world = World_Trace()
            map_frame_data = self.skeleton_map[subj]
            for joint_id in map_frame_data.keys():

                #Joints:
                for t in xrange(self.frames):
                    x = map_frame_data[joint_id][t][0]
                    y = map_frame_data[joint_id][t][1]
                    z = map_frame_data[joint_id][t][2]
                    if joint_id not in ob_states.keys():
                        ob_states[joint_id] = [Object_State(name=joint_id, timestamp=t+1, x=x, y=y, z=z)]
                    else:
                        ob_states[joint_id].append(Object_State(name=joint_id, timestamp=t+1, x=x, y=y, z=z))

            # SOMA objects
            for t in xrange(self.frames):
                for object, (x,y,z) in self.objects.items():
                    if object not in ob_states.keys():
                        ob_states[object] = [Object_State(name=str(object), timestamp=t+1, x=x, y=y, z=z)]
                    else:
                        ob_states[object].append(Object_State(name=str(object), timestamp=t+1, x=x, y=y, z=z))

                # # Robot's position
                # (x,y,z) = self.robot_data[t][0]
                # if 'robot' not in ob_states.keys():
                #     ob_states['robot'] = [Object_State(name='robot', timestamp=t, x=x, y=y, z=z)]
                # else:
                #     ob_states['robot'].append(Object_State(name='robot', timestamp=t, x=x, y=y, z=z))

            for obj, object_state in ob_states.items():
                world.add_object_state_series(object_state)

            # get world trace for each person
            # region = self.soma_roi_config[self.waypoint]
            self.subj_world_trace[subj] = self.get_object_frame_qsrs(world)
            self.skeleton_map[subj]["left_hand"] = self.skeleton_map[subj]["left_hand"][2:]
            self.skeleton_map[subj]["right_hand"] = self.skeleton_map[subj]["right_hand"][2:]

    def convert_to_map(self):
        self.skeleton_map = {}
        frames = 20      # frames to be processed
        self.frames = frames

        for subj in self.sk_publisher.accumulate_data.keys():

            all_data = len(self.sk_publisher.accumulate_data[subj])
            if all_data<frames*2:
                continue
            #print all_data

            new_range = range(0,frames*2, 2)
            # new_range = range(np.max([0, all_data-frames*2]), all_data,2)

            self.skeleton_map[subj] = {}
            self.skeleton_map[subj]['right_hand'] = []
            self.skeleton_map[subj]['left_hand'] = []
            for f in new_range:
                # print '*',f
                robot_pose = self.sk_publisher.accumulate_robot[subj][f]

                for j, name in zip([7, 3],["right_hand", "left_hand"]):
                    hand = self.sk_publisher.accumulate_data[subj][f].joints[j].pose.position
                    map_point = self.sk_publisher.convert_to_world_frame(hand, robot_pose)
                    map_joint = [map_point.x, map_point.y, map_point.z]
                    self.skeleton_map[subj][name].append(map_joint)
                    # import pdb; pdb.set_trace()
            self.sk_publisher.accumulate_data[subj] = self.sk_publisher.accumulate_data[subj][2:]


    def get_soma_objects(self):
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


    def get_point_cloud_objects(self, path):
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
                    string = "object_%s_%s" % (int(file_num),int(file_num))
                    objects[string] = (float(x),float(y),float(z)) # hack to keep file_num when object type is passed to QSRLib

        # objects["object_21_21"] = (-7.7,-35.0,1.0)
        # objects["object_13_13"] = (-7.0,-34.2,1.0)
        # objects["object_17_17"] = (-8.5,-31.0,1.0)
        return objects

def nodes(graph):
    """Getter.
    """
    # Get object nodes from graph
    objects, spa, temp = [], [], []
    for node in graph.vs():
        #print node
        if node['node_type'] == 'object':
            objects.append(node['name'])
        if node['node_type'] == 'spatial_relation':
            spa.append(node['name'])
        if node['node_type'] == 'temporal_relation':
            temp.append(node['name'])
    return objects, spa, temp

if __name__ == "__main__":
    rospy.init_node('activity_recognition')

    if len(sys.argv) < 2:
        print "Usage: please provide a QSR run folder number."
        sys.exit(1)
    else:
        run = sys.argv[1]

    act = act_rec_server(run)
    act.main()
