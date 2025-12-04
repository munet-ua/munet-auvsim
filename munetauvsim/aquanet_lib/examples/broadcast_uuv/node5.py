#!/usr/bin/env python

import rospy
import numpy as np
import time

from std_msgs.msg import Header
from uuv_control_msgs.srv import GoTo
from uuv_control_msgs.msg import Waypoint
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker


### Import AquaNet-Lib module here
import aquanet_lib
###


auv5_heading = None              #global var for robot heading vector

#Go To service call
def call_goto(wp, gotoservice, interpolator):
	#rosservice call to Go_To
	try:
		res = gotoservice(wp,wp.max_forward_speed,str(interpolator))
	except rospy.ServiceException as e:
		print("Service call failed: %s"%e)

#callback
def callback(msg):
	msg.header.stamp = rospy.Time.now()
	msg.header.frame_id = "world"
	msg.point.x = msg.point.x+10
	msg.point.y = msg.point.y-10
	msg.point.z = msg.point.z
	msg.max_forward_speed = 0.75
	msg.heading_offset = 0.0
	msg.use_fixed_heading = False
	call_goto(msg, goto5, interpolator)

def listener():
	#rospy.init_node('listener', anonymous=True)
	# rospy.Subscriber("Waypoint_pub", Waypoint, callback)
	# rospy.Subscriber("aquanet_inbound_waypoint", Waypoint, callback)
	# Receive messages from AquaNet
	print("receiving messages from AquaNet")
	aquaNetManager.recv(callback, deserialize=True)


#callback function for auv pose subscriber
def readauvpose(msg):
	global auv_heading
	auv_5heading = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])

if __name__=='__main__':
    ## Initialize aquanet-stack
	nodeAddr = 5
	baseFolder = "/home/user/ros_catkin_ws/src/multi_auv_sim/scripts/backup/aquanet_lib"
	aquaNetManager = aquanet_lib.AquaNetManager(nodeAddr, baseFolder)
	aquaNetManager.initAquaNet()
    ##


	#rospy.init_node('NODE5')
	rospy.init_node('listener', anonymous=True)

	auvpos_sub = rospy.Subscriber(
		'rov5/pose_gt',
		Odometry,
		readauvpose)

	markerpub = rospy.Publisher('sourcemarker', Marker, queue_size=1)

	interpolator = rospy.get_param('~interpolator', 'lipb')
	
	try:
		rospy.wait_for_service('rov5/go_to', timeout=15)
	except rospy.ROSException:
		raise rospy.ROSException('Service not available!')
	
	try:
		goto5 = rospy.ServiceProxy('rov5/go_to', GoTo)
	except rospy.ROSException as e:
		raise rospy.ROSException('Service proxy failed, error=%s', str(e))
	while True:
		listener()
		time.sleep(10)
