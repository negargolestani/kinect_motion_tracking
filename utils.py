import cv2
import pickle
import ctypes
import os
import copy
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime


from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime


main_directory = str( Path(__file__).parents[1] )
calibsest_folder_path = main_directory + '/calibration_setting'
records_folder_path = main_directory + '/data/records'
markers_folder_path = main_directory + '/data/markers'
times_folder_path = main_directory + '/data/times'

datime_format = '%y-%m-%d-%H-%M-%S'
delimiter = ';'



####################################################################################################################################################
class FRAME(object):
    ################################################################################################################################################    
    def __init__(self, time, color_image, depth_image, camera_space):
        self.time = time
        self.color_image = color_image
        self.depth_image = depth_image
        self.camera_space = camera_space
    ################################################################################################################################################
    def show(self, contours=None, frame_type='COLOR_VIEW', wait=None):
        if frame_type=='COLOR_VIEW' and self.color_image is not None:
            color_image = copy.deepcopy( self.color_image )
            if contours is not None: cv2.drawContours(color_image, contours, -1, color=(0,0,255), thickness=5) 
            color_image = cv2.resize( color_image, None, fx=0.5,fy=0.5 )       
            cv2.imshow('Color View', color_image)

        elif frame_type=='DEPTH_VIEW' and self.depth_image is not None:
            cv2.imshow('Depth View', self.depth_image)
        
        if wait is not None: cv2.waitKey(wait)
        return  
    ################################################################################################################################################
    def get_colored_circle(self, color_range, n_circles, radius):  
        # Filter color      
        hsv = cv2.cvtColor(self.color_image , cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_range[0], color_range[1])

        # Cluster pixels to n_markers 
        markers_pixels = np.array(np.where(mask)).transpose().tolist()          # find pixels with targeted color 
        kmeans_model = KMeans( n_clusters=n_circles, n_init=5, max_iter=10)    # cluster pixels to n_markers sets
        kmeans_model.fit_predict(markers_pixels)                                    

        # Draw cicle at the center of each cluster 
        mask = np.zeros_like(mask)
        for center in kmeans_model.cluster_centers_.astype(int):
            cv2.circle( mask, (center[1],center[0]), radius, 255, -1)

        # Find contours of circles
        circles , _ = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        return circles
    ################################################################################################################################################
    def get_location(self, contours):
        # Find actual location (camera space) of markers' center
        locations = list()
        for contour in contours:
            camera_points = list()
            for pixel in contour.reshape(-1,2):
                camera_point = self.camera_space[pixel[1], pixel[0], :]
                if np.any(np.isinf(camera_point)): camera_point = [np.nan, np.nan, np.nan]
                camera_points.append( camera_point )
            locations.append( np.mean(camera_points, axis=0) )
        return locations
####################################################################################################################################################
####################################################################################################################################################
class KINECT(object):
    ################################################################################################################################################
    def __init__(self):
        self.kinect = PyKinectRuntime.PyKinectRuntime( PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color) 
    ################################################################################################################################################    
    def get_camera_space(self, depth_frame):
        S = self.kinect.color_frame_desc.Height * self.kinect.color_frame_desc.Width
        L = self.kinect.depth_frame_desc.Height * self.kinect.depth_frame_desc.Width
        TYPE_CameraSpacePointArray = PyKinectV2._CameraSpacePoint * S
        color2camera_points = ctypes.cast(TYPE_CameraSpacePointArray(), ctypes.POINTER(PyKinectV2._CameraSpacePoint))        
        self.kinect._mapper.MapColorFrameToCameraSpace(ctypes.c_uint(L), depth_frame , ctypes.c_uint(S), color2camera_points)
        pf_csps = ctypes.cast(color2camera_points, ctypes.POINTER(ctypes.c_float))
        camera_space = np.ctypeslib.as_array(pf_csps, shape=(self.kinect.color_frame_desc.Height, self.kinect.color_frame_desc.Width, 3))
        return camera_space
    ################################################################################################################################################    
    def read(self, full_data=True):
        while True:
            if self.kinect.has_new_color_frame() and self.kinect.has_new_depth_frame():                    
                time = datetime.now()
                color_frame = self.kinect.get_last_color_frame()
                depth_frame = self.kinect.get_last_depth_frame()
                depth_frame_ = self.kinect._depth_frame_data

                color_image = color_frame.reshape((self.kinect.color_frame_desc.Height, self.kinect.color_frame_desc.Width, 4)).astype(np.uint8)
                depth_image = depth_frame.reshape((self.kinect.depth_frame_desc.Height, self.kinect.depth_frame_desc.Width)).astype(np.uint8)

                if full_data:
                    camera_space = self.get_camera_space(depth_frame_)
                    return FRAME(time, color_image, depth_image, camera_space)
                else:
                   return FRAME(time, color_image, depth_image, depth_frame_) # return depth_frame_ instead of camera_space for faster recording 
    ################################################################################################################################################    
    def record(self):
        print('Recording is Started')
        print('Press "Esc" Key to Stop Recording')

        # Recording loop
        record = list()
        while cv2.waitKey(1) != 27: 
            frame = self.read(full_data=False)
            frame.show()

            record.append(frame)

        print('Recording is Finished')        
        print('Wait for Processing ...')        

        # Get Camera Space after recording loop
        for i, frame in enumerate(record):
            record[i].camera_space = self.get_camera_space( frame.camera_space )
        
        print('Done!')        
        return record
####################################################################################################################################################
####################################################################################################################################################
class MARKERSET(object):
    ################################################################################################################################################
    def __init__(self, color, n_markers, marker_radius, color_setting_filename='color_setting_default'):
        self.color = color
        self.n_markers = n_markers
        self.marker_radius = marker_radius
        self.get_color_range(color_setting_filename)
    ################################################################################################################################################
    def get_color_range(self, color_setting_filename):
        file_path = calibsest_folder_path + '/' + color_setting_filename + '.pickle'
        if os.path.exists(file_path): 
            color_ranges =  pickle.load(open(file_path, 'rb')) 
            self.color_range = color_ranges[self.color]
        return
    ################################################################################################################################################
    def track(self, record, show=False):  
        if type(record) is list: frames = record
        else: frames = [record]

        motion, times = list(), list()
        for frame in frames:
            frame_circles = frame.get_colored_circle(self.color_range, self.n_markers, self.marker_radius)
            locations = frame.get_location(frame_circles)
            motion.append( locations )
            times.append(frame.time)
            if show: frame.show(contours=frame_circles, wait=500)

        return motion, times
####################################################################################################################################################
####################################################################################################################################################
class COIL(object):
    # Defines coil by 3 marker
    def __init__(self, markers_motion):
        v1 = markers_motion[:,0,:] - markers_motion[:,1,:]
        v2 = markers_motion[:,0,:] - markers_motion[:,2,:]
        norm = np.cross(v1, v2)
        self.norm = norm / ( np.linalg.norm(norm) + 1e-12)
        self.center = np.mean( markers_motion, axis=1)
    ################################################################################################################################################
    def get_relative_motion(self, coil):
        dictance = math.sqrt(sum((self.center - coil.center)**2))
        ang_misalign_deg = math.acos(np.dot(self.norm, coil.norm))*180/math.pi
        return dictance, ang_misalign_deg
    ################################################################################################################################################
####################################################################################################################################################



####################################################################################################################################################    
def save_record(record, file_name):
    file_path = file_path = records_folder_path + '/' + file_name + '.pickle'
    create_folder(file_path)                            # Create folder if it does not exist    
    pickle.dump( record, open(file_path, 'wb'))    # Save as .pickle
    return True
####################################################################################################################################################    
def load_record(file_name):
    file_path = file_path = records_folder_path + '/' + file_name + '.pickle'
    if os.path.exists(file_path): 
        return pickle.load(open(file_path, 'rb'))
    return None 
####################################################################################################################################################


####################################################################################################################################################
def save_times(times, file_name):
    file_path = times_folder_path + '/' + file_name + '.txt'
    create_folder(file_path)                                    # Create folder if it does not exist    

    data_txt = ''
    for time in times: 
        data_txt += time.strftime( datime_format )  + '\n'  
    
    with open(file_path, 'w') as f: 
        f.write( data_txt)    
    
    return
####################################################################################################################################################
def load_times(file_name):
    file_path = times_folder_path + '/' + file_name + '.txt'
    
    if os.path.exists(file_path):
        with open(file_path , 'r') as f: 
            lines = f.read().splitlines() 
        
        times = list()
        for line in lines: 
            times.append( datetime.strptime(line, datime_format) )
        
        return times
    return None
####################################################################################################################################################


####################################################################################################################################################
def save_markers(markers_motion, file_name):
    # Save markers' center  (camera space) as "txt" file 
    file_path = markers_folder_path + '/' + file_name + '.txt'
    create_folder(file_path)                                    # Create folder if it does not exist    
    
    motion = np.array(markers_motion)
    motion = motion.reshape(-1, motion.shape[1] * motion.shape[2])
    np.savetxt(file_path, motion, delimiter=delimiter, fmt='%s')

    return
####################################################################################################################################################
def load_markers(file_name):
    file_path = markers_folder_path + '/' + file_name + '.txt'
    if os.path.exists(file_path):
        markers_motion = np.loadtxt(file_path, delimiter=delimiter)
        return markers_motion.reshape(markers_motion.shape[0], -1, 3 )
####################################################################################################################################################


####################################################################################################################################################
def create_folder(file_path):
    # Create folder if it does not exist
    folder_path = str( Path(file_path).parents[0] )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
####################################################################################################################################################
