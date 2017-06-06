#!/usr/bin/env python
import roslib
import rospy
import os
import sys
import cv2
import numpy as np
import math
from cv_bridge import CvBridge
import getpass, datetime
import argparse
import sensor_msgs.msg
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
import topological_navigation.msg
from strands_navigation_msgs.msg import TopologicalMap
from ch7_3.msg import skeleton_tracker_state, joint_message, skeleton_message, robot_message
from mongodb_store.message_store import MessageStoreProxy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from soma_msgs.msg import SOMAROIObject
# from soma_manager.srv import *
from shapely.geometry import Polygon, Point
from activity_data.msg import HumanActivities


class SkeletonManager(object):
    """To deal with Skeleton messages once they are published as incremental msgs by OpenNI2."""

    def __init__(self, directory):

        self.offline_directory = directory
        self.accumulate_data = {} # accumulates multiple skeleton msg
        self.accumulate_robot = {} # accumulates multiple skeleton msg
        self.sk_mapping = {} # does something in for the image logging

        self.reduce_frame_rate_by = rospy.get_param("~frame_rate_reduce", 1) # roughly: 3-4Hz
        # self.max_num_frames = rospy.get_param("~max_frames", 500)  # roughly 2mins

        # flags to make sure we received every thing
        self._flag_robot = 0
        self._flag_node = 0
        self._flag_rgb = 0
        #self._flag_rgb_sk = 0
        self._flag_depth = 0
        self.action_called = 0

        self.fx = 525.0
        self.fy = 525.0
        self.cx = 319.5
        self.cy = 239.5

        # depth threshold on recordings
        self.dist_thresh = rospy.get_param("~dist_thresh", 1.5)

        # open cv stuff
        self.cv_bridge = CvBridge()


    def convert_to_world_frame(self, pose, robot_msg):
        """Convert a single camera frame coordinate into a map frame coordinate"""
        fx = 525.0
        fy = 525.0
        cx = 319.5
        cy = 239.5

        y,z,x = pose.x, pose.y, pose.z

        xr = robot_msg.robot_pose.position.x
        yr = robot_msg.robot_pose.position.y
        zr = robot_msg.robot_pose.position.z

        ax = robot_msg.robot_pose.orientation.x
        ay = robot_msg.robot_pose.orientation.y
        az = robot_msg.robot_pose.orientation.z
        aw = robot_msg.robot_pose.orientation.w

        roll, pr, yawr = euler_from_quaternion([ax, ay, az, aw])

        yawr += robot_msg.PTU_pan
        pr += robot_msg.PTU_tilt

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

    def _publish_complete_data(self, subj, uuid, vis=False):
        """when user goes "out of scene" publish their accumulated data"""
        # print ">> publishing these: ", uuid, len(self.accumulate_data[uuid]

        # remove the user from the users dictionary and the accumulated data dict.
        del self.accumulate_data[uuid]
        del self.sk_mapping[uuid]


    def incremental_callback(self, msg):
        """accumulate the multiple skeleton messages until user goes out of scene"""
        if self.action_called:
            if self._flag_robot:# and self._flag_rgb and self._flag_depth:
                if msg.uuid in self.sk_mapping:
                    if self.sk_mapping[msg.uuid]["state"] is 'Tracking' and len(self.accumulate_data[msg.uuid]) < self.max_num_frames \
                    and msg.joints[0].pose.position.z > self.dist_thresh:

                        self.sk_mapping[msg.uuid]["msgs_recieved"]+=1
                        if self.sk_mapping[msg.uuid]["msgs_recieved"] % self.reduce_frame_rate_by == 0:
                            self.accumulate_data[msg.uuid].append(msg)
                            robot_msg = robot_message(robot_pose = self.robot_pose, PTU_pan = self.ptu_pan, PTU_tilt = self.ptu_tilt)
                            self.accumulate_robot[msg.uuid].append(robot_msg)
                            # print msg.userID, msg.uuid, len(self.accumulate_data[msg.uuid])


    def run_offline_instead_of_callback(self, vid = ""):

        d_video = os.path.join(self.offline_directory, vid)
        d_sk = os.path.join(d_video, 'skeleton')
        d_robot = os.path.join(d_video, 'robot')

        with open(os.path.join(d_video, 'label.txt')) as f:
            for i, row in enumerate(f):
                if i == 1:
                    self.label = row
        sk_files = [f for f in sorted(os.listdir(d_sk)) if os.path.isfile(os.path.join(d_sk, f))]
        r_files = [f for f in sorted(os.listdir(d_robot)) if os.path.isfile(os.path.join(d_robot,f))]


        self.accumulate_data[vid], self.accumulate_robot[vid] = [], []
        for _file in sorted(sk_files):

            frame = int(_file.replace(".txt", ""))
            sk = get_sk_info(open(os.path.join(d_sk, _file),'r'))   # old ECAI data format.
            r =  get_rob_info(open(os.path.join(d_robot,_file),'r'))

            joints_msgs  = [joint_message(name = n, pose = Pose(Point(j[0],j[1],j[2]), Quaternion(0,0,0,1))) for n,j in sk.items() ]
            robot_pose = Pose(Point(r[0][0],r[0][1],r[0][2]), Quaternion(r[1][0], r[1][1], r[1][2], r[1][3]))

            self.accumulate_data[vid].append(skeleton_message(userID=1, joints= joints_msgs, time = frame))
            self.accumulate_robot[vid].append(robot_message(robot_pose = robot_pose, PTU_pan = 0, PTU_tilt = 10*math.pi / 180. )) # pan tilt set to (0, 10) for ecai dataset


    def new_user_detected(self, msg):
        date = str(datetime.datetime.now().date())
        self.sk_mapping[msg.uuid] = {"state":'Tracking', "frame":1, "msgs_recieved":1, "date":date}
        self.accumulate_data[msg.uuid] = []
        self.accumulate_robot[msg.uuid] = []

    def state_callback(self, msg):
        """Reads the state messages from the openNi tracker"""
        # print msg.uuid, msg.userID, msg.message
        if msg.message == "Tracking":
            self.new_user_detected(msg)
        elif msg.message == "Out of Scene" and msg.uuid in self.sk_mapping:
            self.sk_mapping[msg.uuid]["state"] = "Out of Scene"
        elif msg.message == "Visible" and msg.uuid in self.sk_mapping:
            self.sk_mapping[msg.uuid]["state"] = "Tracking"
        elif msg.message == "Stopped tracking" and msg.uuid in self.accumulate_data:
            if len(self.accumulate_data[msg.uuid]) != 0:
                del self.accumulate_data[msg.uuid]
                del self.sk_mapping[msg.uuid]

                # self._publish_complete_data(msg.userID, msg.uuid)   #only publish if data captured


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

if __name__ == '__main__':
    rospy.init_node('skeleton_publisher', anonymous=True)

    sk_manager = SkeletonManager()
    rospy.spin()
