#!/usr/bin/env python
import numpy as np
import rospy
from sensor_msgs.msg import Imu

pub = rospy.Publisher('/sensors/imu_at_base', Imu, queue_size=400)

# imu_to_velo = np.array([[ 0.7661291,  0.6426867,  0.0002618,  0.140],
#                         [ 0.6426867, -0.7661290, -0.0003121, -0.062],
#                         [-0.0000000,  0.0004073, -0.9999999, -0.106],
#                         [ 0,          0,          0,          1]])

imu_to_base = np.array([[-1.0, 0.0,  0.0, -0.038],
                        [ 0.0, 1.0,  0.0, -0.06245],
                        [ 0.0, 0.0, -1.0, -0.1837],
                        [ 0.0, 0.0,  0.0,  1.0]])

def callback(data):
    
    # Orient to base
    angular_velocity_arr = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])
    angular_velocity_arr_transformed = np.matmul(imu_to_base[:3,:3], angular_velocity_arr.T)
    angular_velocity_arr_transformed = angular_velocity_arr_transformed.T

    linear_acceleration_arr = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z])
    linear_acceleration_arr_transformed = np.matmul(imu_to_base[:3,:3], linear_acceleration_arr.T)
    linear_acceleration_arr_transformed = linear_acceleration_arr_transformed.T


    # create message
    imu_msg = Imu()
    imu_msg.header.seq = data.header.seq
    imu_msg.header.stamp = data.header.stamp
    imu_msg.header.frame_id = data.header.frame_id # might need to change to base after
    imu_msg.orientation.w = data.orientation.w
    imu_msg.orientation.x = data.orientation.x
    imu_msg.orientation.y = data.orientation.y
    imu_msg.orientation.w = data.orientation.w
    imu_msg.orientation_covariance = data.orientation_covariance
    imu_msg.angular_velocity.x = angular_velocity_arr_transformed[0]
    imu_msg.angular_velocity.y = angular_velocity_arr_transformed[1]
    imu_msg.angular_velocity.z = angular_velocity_arr_transformed[2]
    imu_msg.angular_velocity_covariance = data.angular_velocity_covariance
    imu_msg.linear_acceleration.x = linear_acceleration_arr_transformed[0]
    imu_msg.linear_acceleration.y = linear_acceleration_arr_transformed[1]
    imu_msg.linear_acceleration.z = linear_acceleration_arr_transformed[2]
    imu_msg.linear_acceleration_covariance = data.linear_acceleration_covariance

    pub.publish(imu_msg)
    
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('imuToBase', anonymous=True)

    rospy.Subscriber("/sensors/imu", Imu, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
