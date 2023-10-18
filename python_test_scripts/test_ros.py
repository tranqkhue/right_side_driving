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
	                      	   		   isClosed=False, color=255, thickness=path_default_width_pixel)

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
	# Find foremost points
	
	# Find start_point's foremost of boundary:
	starting_point = pixel_position[0]
	starting_orientation = poses_array[0,2]

	start_foremost = np.array([starting_point[1]-path_default_width_pixel/2*np.cos(starting_orientation+np.pi/2), \
							   starting_point[0]-path_default_width_pixel/2*np.sin(starting_orientation+np.pi/2)])

	global start_foremost_pixel
	start_foremost_pixel = find_nearest(nonzero_boundary_map, start_foremost)
	# start_foremost_pixel = start_foremost.astype(np.int)
	start_foremost_pixel[0], start_foremost_pixel[1] = start_foremost_pixel[1], start_foremost_pixel[0]

	# Find end_point's foremost of boundary:
	end_point = pixel_position[len(pixel_position)-1]
	end_orientation = poses_array[len(poses_array)-1,2]

	end_foremost = np.array([end_point[1]+path_default_width_pixel/2*np.cos(end_orientation+np.pi/2), \
							 end_point[0]+path_default_width_pixel/2*np.sin(end_orientation+np.pi/2)])
	#print(start_foremost)

	global end_foremost_pixel
	end_foremost_pixel = find_nearest(nonzero_boundary_map, end_foremost)
	# end_foremost_pixel = end_foremost.astype(np.int)
	end_foremost_pixel[0], end_foremost_pixel[1] = end_foremost_pixel[1], end_foremost_pixel[0]

	pixel_position = np.insert(pixel_position, 0, start_foremost_pixel, axis=0)
	pixel_position = np.insert(pixel_position, len(pixel_position), end_foremost_pixel, axis=0)
	boundary_map = cv2.polylines(boundary_map, np.int32([pixel_position]), 
	                      	     isClosed=False, color=255, thickness=1)	
	#==================================================================================
	  
	# Generate blurred filled map from inner_filled_map
	blurred_filled_map = cv2.blur(inner_filled_map.astype(np.uint8),
								  (path_default_width_pixel,path_default_width_pixel), 0)  
	# blur with circular kernel which is consistent with circular dilation from cv2.polylines
	blurred_kernel = np.zeros((path_default_width_pixel, path_default_width_pixel))
	center = int((path_default_width_pixel-1)/2)
	blurred_kernel = cv2.circle(blurred_kernel, (center, center), int((path_default_width_pixel-1)/2-1), 1, -1)
	# cv2.imshow('blurred_kernel', blurred_kernel)
	average_blurred_kernel = blurred_kernel / (path_default_width_pixel * path_default_width_pixel)
	blurred_filled_map = cv2.filter2D(src=inner_filled_map.astype(np.uint8), ddepth=-1, kernel=average_blurred_kernel)

	# Since blur increase number of non-zero elements
	# we use inner_filled_map as mask to keep that number the same!
	blurred_filled_map[np.nonzero(inner_filled_map == 0)] = 0
	masked_blurred_filled_map = cv2.bitwise_and(blurred_filled_map, blurred_filled_map,
	 											mask = inner_filled_map.astype(np.uint8))
	# Normalize masked blurred filled map to range 0-255
	masked_blurred_filled_map[masked_blurred_filled_map==0] = np.max(masked_blurred_filled_map)
	normalized_blurred_filled_map = np.zeros(map_size)
	normalized_blurred_filled_map = cv2.normalize(masked_blurred_filled_map, normalized_blurred_filled_map,
												  0, 255, cv2.NORM_MINMAX)
	normalized_blurred_filled_map = cv2.bitwise_and(normalized_blurred_filled_map, normalized_blurred_filled_map,
														mask = inner_filled_map.astype(np.uint8))
	# print('Nonzero normalized path map: ', np.count_nonzero(normalized_blurred_filled_map==255))

	inversed_blurred_filled_map = 255 - blurred_filled_map
	global masked_inversed_blurred_filled_map
	masked_inversed_blurred_filled_map = cv2.bitwise_and(inversed_blurred_filled_map, inversed_blurred_filled_map,
													 mask = inner_filled_map.astype(np.uint8))

	# kernel = np.ones((15, 15), np.uint8) 
	#img_dilation = cv2.dilate(boundary_map, kernel, iterations=1) 
	#s output = cv2.cvtColor(np.float32(img_dilation), cv2.COLOR_GRAY2BGR)
	# cv2.circle(output, start_foremost_pixel, 10, (0,0,255), -1)
	# cv2.circle(output, end_foremost_pixel, 10, (0,255,0), -1)


	# global distTransform
	# distTransform= cv2.distanceTransform(img_dilation.astype(np.uint8), cv2.DIST_L2, 5)
	# normalized_distTransform = cv2.normalize(distTransform, None,
	# 											  0, 255, cv2.NORM_MINMAX)
	
	# contours, hierarchy = cv2.findContours(boundary_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# cv2.drawContours(img_dilation, contours, -1, (0, 255, 0), 3) 


	#====================================================================================================
	# Find seeding point for cv2.floodFill for right-side lane
	right_side_point = np.array([starting_point[1]-path_default_width_pixel/5*np.cos(starting_orientation-np.pi), \
							    starting_point[0]-path_default_width_pixel/5*np.sin(starting_orientation-np.pi)])
	global right_side_pixel
	right_side_pixel = right_side_point.astype(np.uint32)
	right_side_pixel[0], right_side_pixel[1] = right_side_pixel[1], right_side_pixel[0]

	flood_filled = cv2.floodFill(boundary_map.astype(np.uint8),None,right_side_pixel,255)[1]
	flood_filled = flood_filled - boundary_map

	masked_inversed_blurred_filled_map = masked_inversed_blurred_filled_map.astype(np.int32)
	masked_inversed_blurred_filled_map[flood_filled==255] = np.max(blurred_filled_map)-30+masked_inversed_blurred_filled_map[flood_filled==255]
	masked_inversed_blurred_filled_map[flood_filled==0]   = blurred_filled_map[flood_filled==0]
	normalized = cv2.normalize(masked_inversed_blurred_filled_map, masked_inversed_blurred_filled_map,
								0, 255, cv2.NORM_MINMAX)


	#img_resized = cv2.resize(boundary_map, (0,0), fx=0.15, fy=0.15)
	cv2.imshow('blurred', cv2.resize(normalized_blurred_filled_map, (0,0), fx=0.15, fy=0.15))
	cv2.imshow('flood_filled', cv2.resize(flood_filled.astype(np.uint8), (0,0), fx=0.15, fy=0.15))
	cv2.imshow('masked_inversed', cv2.resize(masked_inversed_blurred_filled_map.astype(np.uint8), (0,0), fx=0.15, fy=0.15))
	cv2.waitKey(20)

def create_map(pixel_position, map_size, PATH_DEFAULT_WIDTH, map_resolution):
	path_default_width_pixel = int(PATH_DEFAULT_WIDTH/map_resolution)
	map_matrix = cv2.polylines(map_matrix, np.int32([pixel_position]), 
	                      	   isClosed=False, color=255, thickness=path_default_width_pixel)

def find_nearest(arr,value):
    newList = arr.transpose() - value.transpose()
    sort = np.sum(np.power(newList, 2), axis=1)
    return arr.transpose()[sort.argmin()].transpose()


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