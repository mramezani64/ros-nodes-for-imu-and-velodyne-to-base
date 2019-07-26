#!/usr/bin/env python

import numpy as np
import rospy
import tf2_ros
import tf2_py 
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField

pub = rospy.Publisher('/velodyne/point_cloud_at_base', PointCloud2, queue_size=10)

# imu_to_velo = np.array([[ 0.7661291,  0.6426867,  0.0002618,  0.140],
#                         [ 0.6426867, -0.7661290, -0.0003121, -0.062],
#                         [-0.0000000,  0.0004073, -0.9999999, -0.106],
#                         [ 0,          0,          0,          1]])

velo_to_base_r = np.array([[-0.7660444,  0.6427876,  0.0,   0.0],
                         [-0.6427876, -0.7660444,  0.0,  0.0],
                         [ 0.0,        0.0,        1.0,  0.0],
                         [ 0.0,        0.0,        0.0,  1.0]])

velo_to_base_t = np.array([[1.0,  0.0,  0.0,   0.078],
                         [0.0, 1.0,  0.0,   -0.066],
                         [ 0.0,        0.0,        1.0,  0.2896],
                         [ 0.0,        0.0,        0.0,  1.0]])


def xyz_array_to_pointcloud2(points, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = len(points)
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1),
        PointField('ring', 16, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 20
    msg.row_step = 20*points.shape[0]
    msg.is_dense = True
    msg.data = np.asarray(points, np.float32).tostring()

    return msg 


def remove_nans_mask(cloud_array):
    mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
    return mask

def callback(message):

    arr_all_points = ros_numpy.point_cloud2.pointcloud2_to_array(message)
    mask_of_nans = remove_nans_mask(arr_all_points)

    arr_all_points_removed_nans = arr_all_points[mask_of_nans]

    pc_array = np.zeros(arr_all_points_removed_nans.shape + (3,), dtype=np.float32)
    pc_array[...,0] = arr_all_points_removed_nans['x']
    pc_array[...,1] = arr_all_points_removed_nans['y']
    pc_array[...,2] = arr_all_points_removed_nans['z']


    # print("shape of intensity arr: {}".format(np.shape(arr_all_points_removed_nans['intensity'])))

    pc_array_homogeneous = np.concatenate((pc_array, np.ones((np.shape(pc_array)[0],1))), axis=1)

    # print("original points: {}".format(pc_array[:5,:]))
    # print("original points homogeneous: {}".format(pc_array_homogeneous[:5,:]))

    points_at_base = np.matmul(velo_to_base_t, pc_array_homogeneous.T) # Nx4, 4x4
    points_at_base = np.matmul(velo_to_base_r, points_at_base) # Nx4, 4x4
    points_at_base = points_at_base.T

    # print("points after matmul: {}".format(points_at_base))

    points_at_base_xyz = points_at_base[:,:3]

    # print("shape of points_at_base_xyz: {}".format(np.shape(points_at_base_xyz)))

    points_at_base_xyz_added_intensity = np.concatenate((points_at_base_xyz, np.reshape(arr_all_points_removed_nans['intensity'], (len(arr_all_points_removed_nans['intensity']),1))), axis=1)

    points_at_base_xyz_added_intensity_ring = np.concatenate((points_at_base_xyz_added_intensity,np.reshape(arr_all_points_removed_nans['ring'], (len(arr_all_points_removed_nans['ring']),1))), axis=1)

    
    # print("transformed points: {}".format(points_at_base_xyz[:5,:]))

    # print("sample transformed cloud: {}".format(points_at_base_xyz[:20,:]))

    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('ring', np.uint16)]
    points_at_base_xyz_with_dtype = np.array(points_at_base_xyz_added_intensity_ring, dtype=dtype)


    # points_at_base_xyz.dtype= [np.float32,np.float32,np.float32]

    # print(points_at_base_xyz.dtype)


    pub.publish(xyz_array_to_pointcloud2(points_at_base_xyz_with_dtype, stamp=message.header.stamp, frame_id="base"))
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('veloToBase', anonymous=True)

    rospy.Subscriber("/velodyne/point_cloud", PointCloud2, callback)
    # rospy.Subscriber("/velodyne_points", PointCloud2, callback)


    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
