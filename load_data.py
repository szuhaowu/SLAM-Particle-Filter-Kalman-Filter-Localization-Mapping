import numpy as np


if __name__ == '__main__':
  dataset = 20
  
  with np.load("Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

  with np.load("Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
    
  with np.load("Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
  
  with np.load("Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images



#%%
#np.save('disp_stamps',disp_stamps)
#np.save('encoder_counts',encoder_counts)
#np.save('encoder_stamps',encoder_stamps)
#np.save('imu_angular_velocity',imu_angular_velocity)
#np.save('imu_linear_acceleration',imu_linear_acceleration)
#np.save('imu_stamps',imu_stamps)
#np.save('lidar_angle_max',lidar_angle_max)
#np.save('lidar_angle_min',lidar_angle_min)
#np.save('lidar_range_max',lidar_range_max)
#np.save('lidar_range_min',lidar_range_min)
#np.save('lidar_ranges',lidar_ranges)
#np.save('lidar_stamps',lidar_stamps)
#np.save('rgb_stamps',rgb_stamps)

#%%
#np.save('encoder_counts_testing',encoder_counts)
#np.save('encoder_stamps_testing',encoder_stamps)
#np.save('imu_angular_velocity_testing',imu_angular_velocity)
#np.save('imu_stamps_testing',imu_stamps)
#np.save('lidar_angle_max_testing',lidar_angle_max)
#np.save('lidar_angle_min_testing',lidar_angle_min)
#np.save('lidar_range_max_testing',lidar_range_max)
#np.save('lidar_range_min_testing',lidar_range_min)
#np.save('lidar_ranges_testing',lidar_ranges)
#np.save('lidar_stamps_testing',lidar_stamps)

