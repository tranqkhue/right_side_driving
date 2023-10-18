import rospy
from nav_msgs.msg import Path
from nav_msgs.msg import OccupancyGrid

import cv2
import numpy as np

import time

#============================================================

def path_callback(data):
	print('Path callback check!')
	global poses_array
	poses_array = process_msg_into_array(data)

	global map_size, map_resolution

	global pixel_position
	pixel_position = np.int32(np.round(poses_array[:, 0:2]/map_resolution))
	pixel_position[:,1] = map_size[1] - pixel_position[:,1]

	# Generate discrete viapoints ap
	discrete_viapoints_map = np.zeros(map_size)
	discrete_viapoints_map[pixel_position[:,1], pixel_position[:,0]] = 255

	#==================================================================================
	# Generate boundary

	# Generate continous path map with cv2.polyLines linear interpolation
	PATH_DEFAULT_WIDTH = 5 # unit meter
	path_default_width_pixel = int(PATH_DEFAULT_WIDTH/map_resolution)
	global continous_path_map
	continous_path_map = np.zeros(map_size)
	continous_path_map = cv2.polylines(continous_path_map, np.int32([pixel_position]), 
						  	   		   isClosed=False, color=255, thickness=1)

	inner_filled_map = np.zeros(map_size)
	inner_filled_map = cv2.polylines(inner_filled_map, np.int32([pixel_position]), 
						  	   		 isClosed=False, color=255, thickness=path_default_width_pixel-4)
	outer_filled_map = np.zeros(map_size)
	outer_filled_map = cv2.polylines(outer_filled_map, np.int32([pixel_position]), 
							   		 isClosed=False, color=255, thickness=path_default_width_pixel)
	global boundary_map
	boundary_map = outer_filled_map-inner_filled_map
	nonzero_boundary_map = np.nonzero(boundary_map)
	nonzero_boundary_map = np.array(nonzero_boundary_map)

	#==================================================================================
	# Find right-end points
	
	# Find start_point's right-end point of boundary:
	starting_right_end_point = pixel_position[0]
	starting_right_end_orientation = poses_array[0,2]

	starting_right_end = np.array([starting_right_end_point[1]-path_default_width_pixel/2*np.cos(starting_right_end_orientation-np.pi), \
							   	   starting_right_end_point[0]-path_default_width_pixel/2*np.sin(starting_right_end_orientation-np.pi)])

	global starting_right_end_pixel
	starting_right_end_pixel = find_nearest(nonzero_boundary_map, starting_right_end)
	# start_foremost_pixel = start_foremost.astype(np.int)
	starting_right_end_pixel[0], starting_right_end_pixel[1] = starting_right_end_pixel[1], starting_right_end_pixel[0]

	#-------------------------------------------------------

	# Find end_point's right-end point of boundary:
	ending_right_end_point = pixel_position[len(pixel_position)-1]
	ending_right_end_orientation = poses_array[len(poses_array)-1,2]

	ending_right_end = np.array([ending_right_end_point[1]-path_default_width_pixel/2*np.cos(ending_right_end_orientation-np.pi), \
							  	 ending_right_end_point[0]-path_default_width_pixel/2*np.sin(ending_right_end_orientation-np.pi)])

	global ending_right_end_pixel
	ending_right_end_pixel = find_nearest(nonzero_boundary_map, ending_right_end)
	# start_foremost_pixel = start_foremost.astype(np.int)
	ending_right_end_pixel[0], ending_right_end_pixel[1] = ending_right_end_pixel[1], ending_right_end_pixel[0]

	global seperated_boundary_map
	seperated_boundary_map = boundary_map.copy()
	cv2.circle(seperated_boundary_map, starting_right_end_pixel, 2, (0,0,255), -1)
	cv2.circle(seperated_boundary_map, ending_right_end_pixel, 2, (0,0,255), -1)

	global cnts
	cnts, _ = cv2.findContours(seperated_boundary_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if (len(cnts)!=2):
		print(len(cnts))
	for i in range(len(cnts)):
		cnt = cnts[i]
		if (i==0):
			mid_point_cnt = cnt[int(cnt.shape[0]/3), 0, :]
			global nearest_continous_path_point,index
			nearest_continous_path_point, index = find_nearest(pixel_position, mid_point_cnt, index=True)
			print(mid_point_cnt)
			angle = np.arctan2(mid_point_cnt[1]-nearest_continous_path_point[1], mid_point_cnt[0]-nearest_continous_path_point[0]) \
								+ poses_array[index][2]
		else:
			nearest_cnt_point = find_nearest(cnt[:,0,:], nearest_continous_path_point)
			print(nearest_cnt_point)
			angle = np.arctan2(nearest_cnt_point[1]-nearest_continous_path_point[1], nearest_cnt_point[0]-nearest_continous_path_point[0]) \
								+ poses_array[index][2]
		# print(angle/np.pi*180) 
		if (np.abs(angle-np.pi/2)<0.1):
			right_lane_point = cnt[:,0,:]
			break

	#==================================================================================
	# Create right lane area
	right_side_dilated = np.zeros(map_size)
	right_side_dilated = cv2.polylines(right_side_dilated, np.int32([right_lane_point]), 
							   		   isClosed=False, color=255, thickness=path_default_width_pixel*2+4)
	right_side_dilated_blurred = cv2.blur(right_side_dilated,(path_default_width_pixel*2+4,path_default_width_pixel*2+4))
	print(right_side_dilated_blurred.dtype)
	normalized_blurred_filled_map = np.zeros(map_size)
	#normalized_blurred_filled_map = cv2.bitwise_and(normalized_blurred_filled_map, normalized_blurred_filled_map,
	#													mask = outer_filled_map.astype(np.uint8))
	normalized_blurred_filled_map[outer_filled_map == 0] = np.max(normalized_blurred_filled_map)
	normalized_blurred_filled_map = cv2.normalize(right_side_dilated_blurred, normalized_blurred_filled_map,
												  0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	normalized_blurred_filled_map[outer_filled_map == 0] = 0

	normalized_blurred_filled_map[normalized_blurred_filled_map == 0] = np.max(normalized_blurred_filled_map)
	normalized_blurred_filled_map = cv2.normalize(normalized_blurred_filled_map, normalized_blurred_filled_map,
												  0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	normalized_blurred_filled_map[outer_filled_map == 0] = 0

	#==================================================================================
	# Visualize right end points
	
	viz_right_end_pts = cv2.cvtColor(boundary_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)
	cv2.circle(viz_right_end_pts, starting_right_end_pixel, 10, (0,0,255), -1)
	cv2.circle(viz_right_end_pts, ending_right_end_pixel, 10, (0,0,255), -1)

	# Visualize contours
	viz_cnts = cv2.cvtColor(boundary_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)
	cv2.drawContours(viz_cnts, cnts, 0, (0,0,255), 3)
	cv2.drawContours(viz_cnts, cnts, 1, (0,255,0), 3)
	#cv2.drawContours(viz_cnts, cnts, 2, (255,0,0), 3)


	continous_path_map_dilated = cv2.dilate(continous_path_map, None, iterations=5) 
	cv2.imshow('path', cv2.resize(continous_path_map_dilated, (0,0), fx=0.15, fy=0.15))
	#cv2.imshow('outer_filled_map', cv2.resize(outer_filled_map, (0,0), fx=0.15, fy=0.15))
	# cv2.imshow('viz_right_end_pts', cv2.resize(viz_right_end_pts, (0,0), fx=0.15, fy=0.15))
	# cv2.imshow('viz_cnts', cv2.resize(viz_cnts, (0,0), fx=0.15, fy=0.15))
	cv2.imshow('right_side', cv2.resize(normalized_blurred_filled_map, (0,0), fx=0.15, fy=0.15))
	cv2.waitKey(20)
	#==================================================================================

def create_map(pixel_position, map_size, PATH_DEFAULT_WIDTH, map_resolution):
	path_default_width_pixel = int(PATH_DEFAULT_WIDTH/map_resolution)
	map_matrix = cv2.polylines(map_matrix, np.int32([pixel_position]), 
						  	   isClosed=False, color=255, thickness=path_default_width_pixel)

def find_nearest(arr, value, index=False):
	try:
		newList = arr.transpose() - value.transpose()
		sort = np.sum(np.power(newList, 2), axis=1)
		if (index == False):
			return arr.transpose()[sort.argmin()].transpose()
		else:
			return arr.transpose()[sort.argmin()].transpose(), sort.argmin()
	except ValueError:
		newList = arr - value
		sort = np.sum(np.power(newList, 2), axis=1)
		if (index == False):	
			return arr[sort.argmin(),:]
		else:
			return arr[sort.argmin(),:], sort.argmin()


def process_msg_into_array(msg):
	# The index of this array: 
	# ((position x, position y, angle theta), n_poses)
	# in frame "map"

	poses = msg.poses
	poses_array = np.zeros((len(poses), 3))
	for i in range(len(poses)):
		poses_array[i, 0] = poses[i].pose.position.x
		poses_array[i, 1] = poses[i].pose.position.y

		qz = poses[i].pose.orientation.z
		qw = poses[i].pose.orientation.w
		theta_from_msg = np.arctan2(2*qw*qz, 1-2*qz*qz)
		if (np.abs(theta_from_msg)<0.001) & (i<(len(poses)-1)):
			theta = np.arctan2(poses[i+1].pose.position.y - poses[i].pose.position.y,
							   poses[i+1].pose.position.x - poses[i].pose.position.x)
		poses_array[i, 2] = theta

	return poses_array

#============================================================

def global_map_callback(msg)	:
	print('Map callback check!')
	global map_size, map_resolution
	map_resolution = msg.info.resolution
	map_size = (msg.info.width, msg.info.height)

#============================================================

rospy.init_node('right_side_processing', anonymous=False)
rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, path_callback)
rospy.Subscriber('/map', OccupancyGrid, global_map_callback)
rospy.spin()

cv2.destroyAllWindows