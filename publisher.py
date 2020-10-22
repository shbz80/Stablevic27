#!/usr/bin/env python
# license removed for brevity
import rospy
import numpy as np
from std_msgs.msg import String, Int16
from gps_agent_pkg.msg import TrialCommand, SampleResult, PositionCommand, \
        RelaxCommand, DataRequest, TfActionCommand, TfObsData
# from gps_agent_pkg.msg import SampleResult
from gps_agent_pkg.msg import Custom

def talker():
    # pub_type = SampleResult
    # sample_result = SampleResult()
    pub_topic = '/gps_controller_sent_robot_action_tf'
    pub_type = TfActionCommand
    msg = TfActionCommand()
    msg.id=0
    msg.dU = 7
    msg.action = np.ones(7)

    # pub_type = Custom
    # msg = Custom()
    # msg.data = "Shahbaz"

    # pub_topic = 'chatter'
    # pub_type = String
    # msg = String()
    # msg.data = "Shahbaz"

    # pub_type = Int16
    # msg = Int16()
    # msg.data = 1

    pub = rospy.Publisher(pub_topic, pub_type, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        # hello_str = "hello world %s" % rospy.get_time()
        # rospy.loginfo(hello_str)
        # pub.publish(hello_str)
        # rospy.loginfo(sample_result)
        # pub.publish(sample_result)
        # rospy.loginfo(tf_cmd.action)
        # pub.publish(tf_cmd)
        rospy.loginfo(msg)
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass