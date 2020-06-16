import numpy as np
import cv2
import pickle
import ctypes
import os
import copy
import sys
# import math
from datetime import datetime
from pathlib import Path

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime


main_directory = str( Path(__file__).parents[1] )
calibsest_folder_path = main_directory + '/calibration_setting'
records_folder_path = main_directory + '/data'
markers_folder_path = main_directory + '/data'


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
    def color_2_contours(self, color_range, n_Contours=None):        
        #  Get objects contour
        hsv = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_range[0], color_range[1])

        # Find objects contour
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find n=n_markers largest contours 
        if n_Contours is not None and len(contours) > n_Contours: contours = contours[:n_Contours]
        
        return contours            
    ################################################################################################################################################
    def color_2_circle_contours(self, color_range, n_Contours=None):
        # Make contours circle shape
        contours = self.color_2_contours(color_range, n_Contours=n_Contours)        
        img = cv2.cvtColor(np.zeros_like(self.color_image), cv2.COLOR_RGB2GRAY)
        for contour in contours:
            (x,y),radius = cv2.minEnclosingCircle(contour)
            img = cv2.circle(img, (int(x),int(y)), int(radius), 255, -1)            
        circle_contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return circle_contours
    ################################################################################################################################################
    def contour_2_color_points(self, contour):
        # Make contours into circle shape
        frame = np.zeros_like(self.color_image)                        
        cv2.drawContours(frame, [contour], -1, color=250, thickness=-1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        color_points = np.where(frame)
        color_points = np.array(color_points).transpose().tolist()
        return color_points, frame
    ################################################################################################################################################
    def color_points_2_camera_points(self, color_points):        
        camera_points = list()
        for color_point in color_points:
            camera_point = self.camera_space[color_point[0], color_point[1], :] 
            if np.any(np.isinf(camera_point)): camera_point = [np.nan, np.nan, np.nan]
            camera_points.append(camera_point )
        return camera_points
    ################################################################################################################################################
    def save(self, file_path):
        # save_pickle( self.__dict__, file_path)
        frames_dict = self.__dict__
        for key, value in frames_dict.items():
            frames_dict.update( {key: [value] })
        save_pickle( frames_dict, file_path)        
        return 
    ################################################################################################################################################
    def get_markers(self, color_range, n_markers=2):
        contours = self.color_2_circle_contours(color_range, n_Contours=n_markers)    
        # self.show(contours=contours, wait=1000)

        markers_camera_space = list()
        for contour in contours:
            color_points, _ = self.contour_2_color_points(contour)              # Targeted points in color space
            color_center = np.nanmean(color_points, axis=0)                     # Average of targeted points in color space -> Markers center
            
            camera_points = self.color_points_2_camera_points(color_points)     # Targeted points in camera space
            camera_center = np.nanmean(camera_points, axis=0)                   # Average of targeted points in camera space -> Markers center
            markers_camera_space.append( camera_center )

        return markers_camera_space, contours    
####################################################################################################################################################


####################################################################################################################################################
class KINECT(object):
    ################################################################################################################################################
    def __init__(self):
        self.kinect = PyKinectRuntime.PyKinectRuntime( PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color) 
        self.reset()
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
    def read(self):
        while True:
            if self.kinect.has_new_color_frame() and self.kinect.has_new_depth_frame():                    
                time = datetime.now()
                color_frame = self.kinect.get_last_color_frame()
                depth_frame = self.kinect.get_last_depth_frame()
                camera_space = self.get_camera_space(self.kinect._depth_frame_data)

                color_image = color_frame.reshape((self.kinect.color_frame_desc.Height, self.kinect.color_frame_desc.Width, 4)).astype(np.uint8)
                depth_image = depth_frame.reshape((self.kinect.depth_frame_desc.Height, self.kinect.depth_frame_desc.Width)).astype(np.uint8)

                return FRAME(time, color_image, depth_image, camera_space)                               
    ################################################################################################################################################    
    def record(self, file_path, show=True):
        print('Recording is Started')
        print('Press "Esc" Key to Stop Recording')

        # Recording loop
        time, color_image, depth_image, camera_space = list(), list(), list(), list() 
        while cv2.waitKey(1) != 27: 
            if self.kinect.has_new_color_frame() and self.kinect.has_new_depth_frame():   
                time.append( 
                    datetime.now() )
                color_image.append( 
                    self.kinect.get_last_color_frame().reshape((self.kinect.color_frame_desc.Height, self.kinect.color_frame_desc.Width, 4)).astype(np.uint8) )
                depth_image.append( 
                    self.kinect.get_last_depth_frame().reshape((self.kinect.depth_frame_desc.Height, self.kinect.depth_frame_desc.Width)).astype(np.uint8) )
                camera_space.append( 
                    self.kinect._depth_frame_data) # camera space is calculated after recording to reduce delay between frames

                if show: cv2.imshow('Color View',cv2.resize(color_image[-1],None,fx=0.5,fy=0.5))


        # Get Camera Space after recording loop
        for cs in camera_space:
            camera_space[i] = self.get_camera_space(cs)

       # Make dict of lists from data of all recorded frames
        frames_dict = dict(  
            time = time, 
            color_image = color_image, 
            depth_image = depth_image,
            camera_space = camera_space )
        save_pickle( frames_dict, file_path)

        print('Recording is Finished')        
        return frames_dict
####################################################################################################################################################


####################################################################################################################################################
def save_pickle(file, file_path):
    if file_path[-7:] != '.pickle': file_path_ = file_path + '.pickle'
    else: file_path_ = file_path
    
    # Create folder if it does not exist
    folder_path = str( Path(file_path_).parents[0] )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
 
    # Save as .pickle
    pickle.dump( file, open(file_path_, 'wb'))
    return 
####################################################################################################################################################
def load_pickle(file_path):
    if file_path[-7:] != '.pickle': file_path_ = file_path + '.pickle'
    else: file_path_ = file_path
    # return none if file_path does not exist
    if os.path.exists(file_path_):
        return pickle.load(open(file_path_, 'rb'))  
    return None
####################################################################################################################################################
def load_record(file_path):
    record_dict = load_pickle(file_path)
    recorded_frames = list()
    for i in range( len(record_dict['time']) ):
        frame = FRAME(
            record_dict['time'][i], 
            record_dict['color_image'][i], 
            record_dict['depth_image'][i], 
            record_dict['camera_space'][i] )
        recorded_frames.append(frame)
    return recorded_frames
####################################################################################################################################################
   







