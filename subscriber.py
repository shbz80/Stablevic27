#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Int16
import numpy as np
from gps_agent_pkg.msg import TrialCommand, SampleResult, PositionCommand, \
        RelaxCommand, DataRequest, TfActionCommand, TfObsData
from gps_agent_pkg.msg import Custom
import sys
print(sys.path)

def callback(sample):
    # rospy.loginfo(np.array(sample.sensor_data.data))
    rospy.loginfo(sample)
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", sample.data)
    # rospy.loginfo(rospy.get_caller_id() + "I heard %d", sample.data)


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    # pub_topic = AGENT_ROS['sample_result_topic']
    # pub_type = SampleResult
    pub_topic = '/gps_controller_sent_robot_action_tf'
    pub_type = TfActionCommand
    # pub_type = Custom
    # pub_topic = 'chatter'

    # pub_type = String
    # pub_type = Int16
    rospy.init_node('listener', anonymous=True)

    # rospy.Subscriber("chatter", String, callback)
    rospy.Subscriber(pub_topic, pub_type, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()