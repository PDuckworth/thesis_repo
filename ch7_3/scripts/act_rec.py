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
from sklearn import metrics
# import ch7_3.skeleton_manager as sk
import ch6_2.ECAI_videos_segmented_by_day as seg
from online_activity_recognition.msg import recogniseAction, recogniseActionResult #, skeleton_message


class act_rec_server(object):
    def __init__(self, run=29):

        """
        Currently uses segmented data - change this when produced QSRs for non segmented vidoes
        """
        dataset_path = "/home/" + getpass.getuser() + "/Datasets/ECAI_Data"
        self.video_filepath = os.path.join(dataset_path, "dataset_segmented_15_12_16")

        self.run = "run_%s" % run
        topicfilepath = os.path.join(self.video_filepath, 'QSR_path', self.run)
        print topicfilepath
        self.load_all_topic_files(topicfilepath)

        self.recog_path = os.path.join(topicfilepath,"online_rec")
        if not os.path.exists(self.recog_path): os.makedirs(self.recog_path)


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
        self.colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [170, 255, 0], [0, 255, 170], [255, 170, 0], [0, 170, 255], [255, 0, 170], [170, 0, 255]]
        self._create_image()

        self.image_pub = rospy.Publisher("/activity_recognition_results", Image, queue_size=10)
        self.image_label = cv2.imread(dataset_path+'/image_label.png')

        self.bridge = CvBridge()
        self.qsr_median_window = 3
        self.ordered_labels = []

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

    def main_loop(self):

        self.act_results = {}

        self.pred_labels = []
        self.true_labels = []

        print "running offline..."
        videos_by_day = seg.segmented_videos()
        self.all_labels = []
        counter = 1
        # for counter, task in enumerate(sorted(os.listdir(directory))):
        for date in sorted(videos_by_day.keys()):
            print "\nDate: ", date

            self.window_size = 20

            for video in videos_by_day[date]:
                if '196' in video or '211' in video: continue

                # self.online_window = np.zeros((self.time, len(self.code_book)), dtype=np.uint8)
                self.run_offline_instead_of_callback(video)

                for it in self.skeleton_map.keys():
                    map_window = self.skeleton_map[it]
                    cam_window = self.skeleton_cam[it]
                    try:
                        map_next = self.skeleton_map[it+1]
                        cam_next = self.skeleton_cam[it+1]
                    except KeyError:
                        map_next = {}
                        cam_next = {}

                    self.get_world_frame_trace(map_window, map_next, cam_window, cam_next)
                    self.update_online_window()
                print "video: %s: %s" %(video, self.label)
                self.recognise_activities()
                self.plot_online_window()

            true_labels  = self.true_labels
            pred_labels  = self.pred_labels

        print "k: %s. v-measure: %0.3f. homo: %0.3f. comp: %0.3f. MI: %0.3f. NMI: %0.3f. "  \
          %(self.num_topics, metrics.v_measure_score(true_labels, pred_labels), metrics.homogeneity_score(true_labels, pred_labels),
            metrics.completeness_score(true_labels, pred_labels), metrics.mutual_info_score(true_labels, pred_labels),
            metrics.normalized_mutual_info_score(true_labels, pred_labels))

        print ">> ended\n"

        name = '/online_predictions.p'
        f1 = open(self.recog_path+name, 'w')
        pickle.dump(pred_labels, f1, 2)
        f1.close()

        name = '/online_gt.p'
        f1 = open(self.recog_path+name, 'w')
        pickle.dump(true_labels, f1, 2)
        f1.close()

        import pdb; pdb.set_trace()


    def run_offline_instead_of_callback(self, vid):

        d_video = os.path.join(self.video_filepath, vid)
        d_sk = os.path.join(d_video, 'skeleton')
        d_robot = os.path.join(d_video, 'robot')

        with open(os.path.join(d_video, 'label.txt')) as f:
            for i, row in enumerate(f):
                if i == 1:
                    self.label = row
        sk_files = [f for f in sorted(os.listdir(d_sk)) if os.path.isfile(os.path.join(d_sk, f))]
        r_files = [f for f in sorted(os.listdir(d_robot)) if os.path.isfile(os.path.join(d_robot,f))]

        self.len_of_video = len(sk_files)
        self.ordered_labels.append(self.label)

        self.skeleton_map = {}
        self.skeleton_cam = {}
        # self.half_windows = math.ceil(self.len_of_video / self.window_size)*2
        self.half_windows = int(math.ceil(self.len_of_video / float(self.window_size)*2))
        for w in xrange(self.half_windows+1):
            self.skeleton_map[w] = {}
            self.skeleton_cam[w] = {}

        # import pdb; pdb.set_trace()

        for _file in sorted(sk_files):

            frame = int(_file.replace(".txt", ""))
            which_window = (frame-1) / (self.window_size/2)

            sk = get_sk_info(open(os.path.join(d_sk, _file),'r'))   # old ECAI data format.
            r =  get_rob_info(open(os.path.join(d_robot,_file),'r'))

            robot_pose = Pose(Point(r[0][0],r[0][1],r[0][2]), Quaternion(r[1][0], r[1][1], r[1][2], r[1][3]))
            for name, j in sk.items():

                if name not in ["head", "torso", "left_hand", "right_hand"]: continue

                # pose = Pose(Point(j[0],j[1],j[2]), Quaternion(0,0,0,1))
                map_point = self.convert_to_world_frame(Point(j[0],j[1],j[2]), robot_pose)

                try:
                    self.skeleton_map[which_window][name].append([map_point.x, map_point.y, map_point.z])
                    self.skeleton_cam[which_window][name].append([j[0],j[1],j[2]])
                except KeyError:
                    self.skeleton_map[which_window][name] = [[map_point.x, map_point.y, map_point.z]]
                    self.skeleton_cam[which_window][name] = [[j[0], j[1], j[2]]]

    #########################################################################
    def update_online_window(self):

        try:
            self.online_window[:self.time-1] = self.online_window[1:]
        except AttributeError:
            # initiate the window of QSTAGS for this person
            self.online_window = np.zeros((self.time, len(self.code_book)), dtype=np.uint8)
            self.online_window_img = self.img #np.zeros((len(self.code_book)*self.th,self.windows_size*self.th2,3),dtype=np.uint8)+255

        #self.online_window_img[subj][:,self.th2:self.windows_size*self.th2,:] = self.online_window_img[subj][:,0:self.windows_size*self.th2-self.th2,:]
        # find which QSTAGS happened in this frame
        self.online_window[-1,:] = 0
        #self.online_window_img[subj][:, 0:self.th2, :] = 255
        for ret in self.subj_world_trace:

            for cnt, h in zip(ret.histogram, ret.code_book):
                oss, ss, ts  = nodes(ret.graphlets[h])
                ssl = [d.values() for d in ss]

                #if "Microwave" in oos:
                #print ">>>", cnt, h, type(h) #, #oss, ssl #ret.qstag.graphlets.graphlets[h]
                if isinstance(h, int):
                    h = "{:20d}".format(h).lstrip()
                #print cnt, h, type(h), len(h), len(self.code_book), len(self.code_book[0]) #, self.code_book.shape
                #code_book = [str(i) for i in self.code_book]

                if h in self.code_book:  # Futures WARNING here
                    index = list(self.code_book).index(h)
                    self.online_window[-1, index] = 1

                #self.online_window_img[subj][index*self.th:index*self.th+self.th, 0:self.th2, :] = 10

    #########################################################################
    def recognise_activities(self):
        self.act_results = {}

        # for window in self.online_window:
            # if sum(window) == 0: continue

        # compressing the different windows to be processed
        for w in range(2,20,2):
            for i in range(self.time - w):
                compressed_window = copy.deepcopy(self.online_window[i,:])
                for j in range(1, w+1):
                    compressed_window += self.online_window[j+i,:]

                compressed_window = [1 if o !=0 else 0 for o in compressed_window]
                # compressed_window /= compressed_window

                for act, topic in self.actions_vectors.items():
                    if act not in self.act_results:
                        self.act_results[act] = np.zeros((self.time), dtype=np.float32)

                    result = np.sum(compressed_window*topic)
                    if result > 0.3:
                    # if result != 0:
                        self.act_results[act][i:i+w] += result
                        # self.act_results[act] += result
                    # if result != 0:
                    #     self.act_results[act][i:i+w] += 20

        # calibration
        for act in self.act_results:
            self.act_results[act] /= 0.4
            # self.act_results[act] = [o if o >30 else 0 for o in self.act_results[act]]

        import pdb; pdb.set_trace()
        # create a classification
        max_dist = 0
        max_act = 100
        for frame in xrange(self.time):
            for act, dist in self.act_results.items():
                if dist[frame] > max_dist:
                    max_dist = dist[frame]
                    max_act = act
            if max_act !=100:
                self.true_labels.append(self.label)
                self.pred_labels.append(max_act)

    #########################################################################
    def plot_online_window(self):
        #if len(self.online_window_img) == 0:
        final_img = self.img #

        final_img = self._update_image(self.act_results)
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

    def get_world_frame_trace(self, map_window, map_next, cam_window, cam_next):
        """Accepts a dictionary of world (soma) objects.
        Adds the position of the object at each timepoint into the World Trace"""

        ### MAP WORLD TRACE
        ob_states={}
        map_world = World_Trace()
        for window in [map_window, map_next]:
            for name, data in window.items():
                for t, j in enumerate(data):
                    try:
                        ob_states[name].append(Object_State(name=name, timestamp=t+1, x=j[0], y=j[1], z=j[2]))
                    except KeyError:
                        ob_states[name] = [Object_State(name=name, timestamp=t+1, x=j[0], y=j[1], z=j[2])]

            for t in xrange(self.window_size/2):
                for name, (x,y,z) in self.objects.items():
                    try:
                        ob_states[name].append(Object_State(name=str(name), timestamp=t+1, x=x, y=y, z=z))
                    except KeyError:
                        ob_states[name] = [Object_State(name=str(name), timestamp=t+1, x=x, y=y, z=z)]

        for obj, object_state in ob_states.items():
            map_world.add_object_state_series(object_state)

        joint_types = {'left_hand': 'hand', 'right_hand': 'hand',  'head-torso': 'tpcc-plane'}

        object_types = joint_types.copy()
        for name in self.objects:
            generic_object = "_".join(name.split("_")[:-1])
            object_types[name] = generic_object

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

        # dynamic_args['argd'] = {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 0.25, 'Near': 0.5, 'Away': 1.0, 'Ignore': 10}}
        dynamic_args['argd'] = {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 0.15, 'Near': 0.3, 'Away': 0.6, 'Ignore': 10}}
        #dynamic_args['argd'] = {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 1.5, 'Ignore': 10}}
        dynamic_args['qtcbs'] = {"qsrs_for": qsrs_for, "quantisation_factor": 0.01, "validate": False, "no_collapse": True} # Quant factor is effected by filters to frame rate
        dynamic_args["qstag"] = {"object_types": object_types, "params": {"min_rows": 1, "max_rows": 1, "max_eps": 4}}
        dynamic_args["filters"] = {"median_filter": {"window": self.qsr_median_window}}

        qsrlib = QSRlib()
        req = QSRlib_Request_Message(which_qsr=["argd", "qtcbs"], input_data=map_world, dynamic_args=dynamic_args)
        #req = QSRlib_Request_Message(which_qsr="argd", input_data=world_trace, dynamic_args=dynamic_args)
        map_ret = qsrlib.request_qsrs(req_msg=req)


        ### CAMERA WORLD TRACE
        ob_states={}
        cam_world = World_Trace()
        for window in [cam_window, cam_next]:
            for name, data in window.items():
                for t, j in enumerate(data):
                    try:
                        ob_states[name].append(Object_State(name=name, timestamp=t+1, x=j[0], y=j[1], z=j[2]))
                    except KeyError:
                        ob_states[name] = [Object_State(name=name, timestamp=t+1, x=j[0], y=j[1], z=j[2])]

        for obj, object_state in ob_states.items():
            cam_world.add_object_state_series(object_state)

        objects_used_tpcc = ['left_hand', 'right_hand', 'left_shoulder', 'right_shoulder', 'left_knee', 'right_knee']
        # qsrs_for = [('head', 'torso', ob) if ob not in ['head', 'torso'] and ob != 'head-torso' else () for ob in object_types.keys()]
        qsrs_for_tpcc = [('head', 'torso', ob) for ob in objects_used_tpcc]

        object_types['head-torso'] = 'tpcc-plane'
        dynamic_args["qstag"] = {"params" : {"min_rows": 1, "max_rows": 2, "max_eps": 4}, "object_types": object_types}
        dynamic_args['tpcc'] = {"qsrs_for": qsrs_for_tpcc}

        req = QSRlib_Request_Message(which_qsr="tpcc", input_data=cam_world, dynamic_args=dynamic_args)
        cam_ret = qsrlib.request_qsrs(req_msg=req)
        # pretty_print_world_qsr_trace("tpcc", camera_response_message)

        #print "\n"
        #for ep in ret.qstag.episodes:
        #    print ep
        self.subj_world_trace = [map_ret.qstag.graphlets, cam_ret.qstag.graphlets]


    # def convert_to_map(self, window_size):
    #     self.skeleton_map = {}
    #
    #
    #     for subj in self.sk_publisher.accumulate_data.keys():
    #
    #         all_data = len(self.sk_publisher.accumulate_data[subj])
    #         if all_data < window_size*2:
    #             continue
    #         #print all_data
    #
    #         new_range = range(0, window_size*2, 2)
    #         # new_range = range(np.max([0, all_data-frames*2]), all_data,2)
    #
    #         self.skeleton_map[subj] = {}
    #         self.skeleton_map[subj]['right_hand'] = []
    #         self.skeleton_map[subj]['left_hand'] = []
    #         for f in new_range:
    #             # print '*',f
    #             robot_pose = self.sk_publisher.accumulate_robot[subj][f]
    #
    #             for j, name in zip([7, 3],["right_hand", "left_hand"]):
    #                 hand = self.sk_publisher.accumulate_data[subj][f].joints[j].pose.position
    #                 map_point = self.convert_to_world_frame(hand, robot_pose)
    #                 map_joint = [map_point.x, map_point.y, map_point.z]
    #                 self.skeleton_map[subj][name].append(map_joint)

            # self.sk_publisher.accumulate_data[subj] = self.sk_publisher.accumulate_data[subj][2:]


    def convert_to_world_frame(self, point, robot_pose):
        """Convert a single camera frame coordinate into a map frame coordinate"""
        fx = 525.0
        fy = 525.0
        cx = 319.5
        cy = 239.5

        y,z,x = point.x, point.y, point.z

        xr = robot_pose.position.x
        yr = robot_pose.position.y
        zr = robot_pose.position.z

        ax = robot_pose.orientation.x
        ay = robot_pose.orientation.y
        az = robot_pose.orientation.z
        aw = robot_pose.orientation.w

        roll, pr, yawr = euler_from_quaternion([ax, ay, az, aw])

        # Fixed for this dataset
        PTU_pan = 0
        PTU_tilt = 10*math.pi / 180.

        yawr += PTU_pan
        pr += PTU_tilt

        # transformation from camera to map
        rot_y = np.matrix([[np.cos(pr), 0, np.sin(pr)], [0, 1, 0], [-np.sin(pr), 0, np.cos(pr)]])
        rot_z = np.matrix([[np.cos(yawr), -np.sin(yawr), 0], [np.sin(yawr), np.cos(yawr), 0], [0, 0, 1]])
        rot = rot_z*rot_y

        pos_r = np.matrix([[xr], [yr], [zr+1.66]]) # robot's position in map frame
        pos_p = np.matrix([[x], [-y], [-z]]) # person's position in camera frame
        map_pos = rot*pos_p+pos_r # person's position in map frame

        x_mf = map_pos[0,0]
        y_mf = map_pos[1,0]
        z_mf = map_pos[2,0]
        # print ">>" , x_mf, y_mf, z_mf
        return Point(x_mf, y_mf, z_mf)


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

            roll, pitch, yaw = euler_from_quaternion([ax, ay, az, aw])    #odom
            #pitch = 10*math.pi / 180.   #we pointed the pan tilt 10 degrees
            #rob_data[1] = [roll, pitch, yaw]
            rob_data[1] = [ax,ay,az,aw]
    return rob_data

if __name__ == "__main__":
    rospy.init_node('activity_recognition')

    if len(sys.argv) < 2:
        print "Usage: please provide a QSR run folder number."
        sys.exit(1)
    else:
        run = sys.argv[1]

    act = act_rec_server(run)
    act.main_loop()
