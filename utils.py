import numpy as np
import pandas as pd
import time as time_lib
import cv2
import os
import ctypes
import pickle

from datetime import datetime, date, time, timedelta
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime


main_directory = str( Path(__file__).parents[1] )
dataset_folder_name = 'datasets'


####################################################################################################################################################
def get_time_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/kinect/time/' + file_name + '.csv'
####################################################################################################################################################
def get_color_video_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/kinect/color_video/' + file_name + '.avi'
####################################################################################################################################################
def get_camera_space_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/kinect/camera_space/' + file_name + '.pkl'
####################################################################################################################################################
def get_markers_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/markers/' + file_name + '.csv'
####################################################################################################################################################
def get_color_setting_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name +  '/' + dataset_name  + '/calibration_setting/' + file_name + '.pickle'
####################################################################################################################################################
####################################################################################################################################################
def create_folder(file_path):
    # Create folder if it does not exist
    folder_path = str( Path(file_path).parents[0] )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
####################################################################################################################################################



####################################################################################################################################################
class KINECT(object):
    ################################################################################################################################################
    def __init__(self, top_margin=0.15, bottom_margin=0.15, left_margin=0.25, right_margin=0.25):
        self.kinect = PyKinectRuntime.PyKinectRuntime( PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color) 

        ch, cw = self.kinect.color_frame_desc.Height, self.kinect.color_frame_desc.Width
        self.color_mask = np.full((ch, cw), 0, dtype=np.uint8)  
        self.color_mask[int(top_margin*ch): -int(bottom_margin*ch), int(left_margin*cw):-int(right_margin*cw)] = 1    
    ################################################################################################################################################    
    def read(self, margin=True, full=True):
        while True:
            if self.kinect.has_new_color_frame() and self.kinect.has_new_depth_frame():   

                self.time = datetime.now().time()
                self.color_frame = self.kinect.get_last_color_frame()
                self.depth_frame = self.kinect._depth_frame_data
        
                self.color_image = self.color_frame.reshape((self.kinect.color_frame_desc.Height, self.kinect.color_frame_desc.Width, 4)).astype(np.uint8)    
                self.color_image = cv2.bitwise_and(self.color_image, self.color_image, mask=self.color_mask)

                if full: self.get_camera_space()

                break        
    ################################################################################################################################################         
    def get_camera_space(self):
        S = np.int(self.kinect.color_frame_desc.Height * self.kinect.color_frame_desc.Width)
        L = np.int(self.kinect.depth_frame_desc.Height * self.kinect.depth_frame_desc.Width)
        TYPE_CameraSpacePointArray = _CameraSpacePoint * S
        color2camera_points = ctypes.cast(TYPE_CameraSpacePointArray(), ctypes.POINTER(_CameraSpacePoint))        
        self.kinect._mapper.MapColorFrameToCameraSpace(ctypes.c_uint(L), self.depth_frame , ctypes.c_uint(S), color2camera_points)
        pf_csps = ctypes.cast(color2camera_points, ctypes.POINTER(ctypes.c_float))
        self.camera_space = np.ctypeslib.as_array(pf_csps, shape=(self.kinect.color_frame_desc.Height, self.kinect.color_frame_desc.Width, 3))
        return 
    ################################################################################################################################################        
    def show(self, contours=None, wait=None):
        if self.color_frame is not None:
            color_image = self.color_image.copy()
            if contours is not None: cv2.drawContours(color_image, contours, -1, color=(255,255,0), thickness=3) 
            color_image = cv2.resize( color_image, None, fx=0.5,fy=0.5 )       
            cv2.imshow('Color View', color_image)
            if wait is not None: cv2.waitKey(wait)
        return    
####################################################################################################################################################
####################################################################################################################################################
class RECORD(object):
    def __init__(self, dataset_name, file_name, color_setting_filename='color_setting_default'):
        color_setting_file_path = get_color_setting_file_path(dataset_name, color_setting_filename)
        with open(color_setting_file_path, "rb") as f: 
            self.color_setting = pickle.load(f)

        camera_space_file_path = get_camera_space_file_path(dataset_name, file_name) 
        with open(camera_space_file_path, "rb") as f: 
            self.camera_spaces = pickle.load(f)

        video_file_path = get_color_video_file_path(dataset_name, file_name)
        self.color_vid = cv2.VideoCapture(video_file_path)

        self.next_idx = 0 
    ############################################################################################################################################        
    def read(self):
        success, frame = self.color_vid.read()          
        if success and self.next_idx<len(self.camera_spaces):
            self.frame = frame
            self.camera_space = self.camera_spaces[self.next_idx]
            self.next_idx += 1        
            return True
        return False      
    ############################################################################################################################################
    def draw_contours(self, contours, color):
        cnts = list()
        for contour in contours:
            if len(contour): cnts.append(contour)
        cv2.drawContours(self.frame, cnts, -1, color=color, thickness=3) 
        return
    ############################################################################################################################################
    def show(self, contours=None, wait=None):
        if self.frame is not None:
            if contours is not None: cv2.drawContours(self.frame, contours, -1, color=(255,255,0), thickness=3) 
            frame = cv2.resize( self.frame, None, fx=0.5,fy=0.5 )       
            cv2.imshow('Color View', frame)
        
            if wait is not None: cv2.waitKey(wait)
        return  
    ############################################################################################################################################
    def get_colored_circles(self, color, n_circles=3): 
        frame = self.frame.copy() 
        color_range = self.color_setting[color]
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred , cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_range[0], color_range[1])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        pixels = np.transpose(np.nonzero(mask)) 
        circles = [[]]*n_circles

        if len(pixels) > n_circles: 

            kmeans = KMeans(n_clusters=n_circles, random_state=0).fit(pixels)
            for n in range(n_circles):

                mask = np.zeros_like(mask)
                for pixel in pixels[ kmeans.labels_== n, :]: mask[ pixel[0], pixel[1] ] = 255            
                
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
                contour = max(contours, key=cv2.contourArea)
                
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                M = cv2.moments(contour)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                mask = np.zeros_like(mask)            
                cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
                circle = max(contours, key=cv2.contourArea)

                circles[n] = circle

        return circles
    ############################################################################################################################################
    def get_pixels(self, contour):
        if len(contour):
            mask = np.zeros((self.frame.shape[0], self.frame.shape[1],3))
            mask = cv2.drawContours(mask, [contour], 0, (1,0,0), thickness=cv2.FILLED)
            x,y = np.where(mask[:,:,0]==1)
            return np.stack((x,y),axis=-1)
        return list()
    ############################################################################################################################################
    def get_locations(self, contours):        
        locations = list()        
        for contour in contours:
            camera_points = list()            
            for pixel in  self.get_pixels(contour):                 
                camera_point = self.camera_space[pixel[0], pixel[1]]
                if np.all( np.isfinite(camera_point)): camera_points.append( camera_point ) 
            if len(camera_points): location = np.nanmedian(np.array(camera_points), axis=0) 
            else: location = [np.nan, np.nan, np.nan]
            locations = [*locations, *location]

        return locations     
    ############################################################################################################################################
####################################################################################################################################################


