import cv2
import pickle
import ctypes
import os
import copy
import sys
import win32api
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime
from scipy import signal

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime


main_directory = str( Path(__file__).parents[1] )
calibsest_folder_path = main_directory + '/calibration_setting'

time_folder_path = main_directory + '/data/time'
color_video_folder_path = main_directory + '/data/record/color_video'
camera_space_folder_path = main_directory + '/data/record/camera_space'

markers_folder_path = main_directory + '/data/markers'

datime_format = '%y-%m-%d-%H-%M-%S'



####################################################################################################################################################
class KINECT(object):
    ################################################################################################################################################
    def __init__(self, top_margin=0.1, bottom_margin=0.1, left_margin=0.1, right_margin=0.1):
        self.kinect = PyKinectRuntime.PyKinectRuntime( PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color) 
        
        ch, cw = self.kinect.color_frame_desc.Height, self.kinect.color_frame_desc.Width
        self.color_mask = np.full((ch, cw), 0, dtype=np.uint8)  
        self.color_mask[int(top_margin*ch): -int(bottom_margin*ch), int(left_margin*cw):-int(right_margin*cw)] = 1    
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
                # depth_frame = self.kinect.get_last_depth_frame()
                color_image = color_frame.reshape((self.kinect.color_frame_desc.Height, self.kinect.color_frame_desc.Width, 4)).astype(np.uint8)
                color_image = cv2.bitwise_and(color_image, color_image, mask=self.color_mask)

                if full_data:
                    camera_space = self.get_camera_space(self.kinect._depth_frame_data)
                    return FRAME(color_image, camera_space), time
                else:
                   return FRAME(color_image, self.kinect._depth_frame_data), time # return depth_frame_ instead of camera_space for faster recording 
    ################################################################################################################################################    
    def record(self, file_name=None):
        print('Recording is Started')
        print('Press "Esc" Key to Stop Recording')

        # Recording loop
        record, time_list = list(), list()
        while cv2.waitKey(1) != 27 and win32api.GetKeyState(0x01)>-1: 
            frame, time = self.read(full_data=False)
            frame.show()
            record.append(frame)
            time_list.append(time)

        print('Recording is Finished')        
        print('Wait for Processing ...')        

        # Get Camera Space after recording loop
        if file_name is None:
           for i, frame in enumerate(record):
                record[i].camera_space = self.get_camera_space( frame.camera_space )
        
        else:
            print('Now Saving ...!') 
            save_time(file_name, time_list)       
            save_record(file_name, record)
        
        print('Done!')
        return
####################################################################################################################################################
####################################################################################################################################################
class FRAME(object):
    ################################################################################################################################################    
    def __init__(self, color_image, camera_space):
        self.color_image = color_image
        self.camera_space = camera_space
    ################################################################################################################################################
    def show(self, contours=None, wait=None):
        if self.color_image is not None:
            color_image = copy.deepcopy( self.color_image )
            if contours is not None: cv2.drawContours(color_image, contours, -1, color=(255,255,0), thickness=3) 
            color_image = cv2.resize( color_image, None, fx=0.5,fy=0.5 )       
            cv2.imshow('Color View', color_image)
        
            if wait is not None: cv2.waitKey(wait)
        return  
    ################################################################################################################################################
    def get_colored_circle(self, color_range, n_circles=3, radius=10):  
        # Filter color      
        hsv = cv2.cvtColor(self.color_image , cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_range[0], color_range[1])
        mask = cv2.medianBlur(mask,	5)

        # find n_circle largest contours:
        contours , _ = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = contours[:n_circles]
        
        # Draw contours 
        mask = cv2.drawContours(np.zeros_like(mask), contours, -1, color=255, thickness=3) 
        mask_ = mask.copy()

        # Cluster pixels to n_markers 
        markers_pixels = np.array(np.where(mask)).transpose().tolist()          # find pixels with targeted color 
        kmeans_model = KMeans( n_clusters=n_circles, n_init=5, max_iter=10)     # cluster pixels to n_markers sets
        kmeans_model.fit_predict(markers_pixels)                                    

        # Draw cicle at the center of each cluster 
        mask = np.zeros_like(mask)
        for center in kmeans_model.cluster_centers_.astype(int):
            cv2.circle( mask, (center[1],center[0]), radius, 255, -1)

        # Find contours of circles
        circles , _ = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        return circles, mask_
    ################################################################################################################################################
    def get_colored_circle_(self, color_range, n_circles, minDist=20, param1=100, param2=10, minRadius=10, maxRadius=20):  
       # Filter color      
        hsv = cv2.cvtColor(self.color_image , cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_range[0], color_range[1])
        mask = cv2.medianBlur(mask,	5)
        
        # Circles center
        circles	= cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        circles	= np.uint16(np.around(circles))

        mask = np.zeros_like(mask)
        for	circ in circles[0,:]:
            cv2.circle(mask, (circ[0],circ[1]), circ[2], 255, -1)
            
        circles , _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
class COIL(object):
    n_marker = 3
    ################################################################################################################################################
    def __init__(self, file_name):
        # Load
        motions = load_motion(file_name)
        
        # Gap filling
        df = pd.DataFrame(motions)
        df.fillna(method='ffill', axis=0, inplace=True)   
        motions = df.to_numpy()        
        
        # Filter
        # motions = signal.savgol_filter( motions, window_length=11, polyorder=1, axis=0)  

        # reshape to list of markers motion
        self.markers_motion = list()    
        for i in range(self.n_marker): self.markers_motion.append( motions[:,i*3:(i+1)*3] )    

        self.get_params()
        return
    ################################################################################################################################################        
    def get_params(self):    
        v1 = self.markers_motion[1] - self.markers_motion[0]
        v2 = self.markers_motion[2] - self.markers_motion[0]
        norm = np.cross(v1, v2)
        self.norm = norm / ( np.reshape(np.linalg.norm(norm, axis=1) + 1e-12, (-1,1)) * np.ones((1,3)) )
        self.center = np.mean( self.markers_motion, axis=0)     
    ################################################################################################################################################
    def get_relative_motion(self, coil, window_length=11, polyorder=1):
        distance =  np.linalg.norm( self.center - coil.center, axis=1) 
        ang_misalign = np.arccos(np.abs(np.sum(np.multiply(self.norm, coil.norm), axis=1)) ) * 180/np.pi

        distance = signal.savgol_filter( distance, window_length=window_length, polyorder=polyorder)  
        ang_misalign = signal.savgol_filter( ang_misalign, window_length=window_length, polyorder=polyorder)  

        return distance, ang_misalign
####################################################################################################################################################
####################################################################################################################################################
class EXPERIMENT(object):
    # Defines coil by 3 marker
    def __init__(self, file_name, reader_color=None, tags_color=None):
        self.times = load_times(file_name)
        self.reader = COIL(file_name + '_' + reader_color)        
        self.tags = list()
        tags_color_ = tags_color
        if type(tags_color) is not list: tags_color_ = [tags_color]
        for tag_color in tags_color_: self.tags.append( COIL(file_name + '_' + tag_color)   )
        return
    ################################################################################################################################################
    def status(self):
        distance_list, ang_misalign_list = list(), list()
        for tag in self.tags:
            distance, ang_misalign = self.reader.get_relative_motion(tag)
            distance_list.append(distance)
            ang_misalign_list.append(ang_misalign)
        return distance_list, ang_misalign_list
####################################################################################################################################################



####################################################################################################################################################    
def save_record(file_name, record):
    camera_file_path = camera_space_folder_path + '/' + file_name + '.pkl'
    color_file_path = color_video_folder_path + '/' + file_name + '.avi'
    color_vid = cv2.VideoWriter(color_file_path, cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (1920,1080))

    camera_space = list()
    for frame in record:
        color_vid.write( cv2.cvtColor(frame.color_image, cv2.COLOR_RGBA2RGB) )
        camera_space.append(frame.camera_space)

    with open(camera_file_path,"wb") as f:
        pickle.dump(camera_space, f)    

    return True
####################################################################################################################################################    
def load_record(file_name):
    camera_file_path = camera_space_folder_path + '/' + file_name + '.pkl'
    color_file_path = color_video_folder_path + '/' + file_name + '.avi'


    if os.path.exists(color_file_path) and os.path.exists(camera_file_path): 
        
        with open(camera_file_path, "rb") as f: 
            camera_space_list = pickle.load(f)
        color_vid = cv2.VideoCapture(color_file_path)
        
        record = list()
        for camera_space in camera_space_list:
            success, color_image = color_vid.read()            
            if not success: return record
            record.append( FRAME(color_image,camera_space) )        
            
    return record 
####################################################################################################################################################


####################################################################################################################################################
def save_time(file_name, times):
    file_path = time_folder_path + '/' + file_name + '.txt'
    create_folder(file_path)                                    # Create folder if it does not exist    

    data_txt = ''
    for time in times: 
        data_txt += time.strftime( datime_format )  + '\n'  
    
    with open(file_path, 'w') as f: 
        f.write( data_txt)    
    
    return
####################################################################################################################################################
def load_times(file_name):
    file_path = time_folder_path + '/' + file_name + '.txt'
    
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
def save_motion(file_name, motion):
    # Save markers' center  (camera space) as "txt" file 
    file_path = markers_folder_path + '/' + file_name + '.txt'
    create_folder(file_path)                                    # Create folder if it does not exist    
    
    motion = np.array(motion)
    motion = motion.reshape(-1, motion.shape[1] * motion.shape[2])
    np.savetxt(file_path, motion, delimiter="\t", fmt='%s')

    return
####################################################################################################################################################
def load_motion(file_name):
    file_path = markers_folder_path + '/' + file_name + '.txt'
    if os.path.exists(file_path):
        markers_motion = np.loadtxt(file_path)
        # return markers_motion.reshape(markers_motion.shape[0], -1, 3 )
        return markers_motion
####################################################################################################################################################


####################################################################################################################################################
def create_folder(file_path):
    # Create folder if it does not exist
    folder_path = str( Path(file_path).parents[0] )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
####################################################################################################################################################

################################################################################################################################################
def load_color_setting(color_setting_filename='color_setting_default'):
    file_path = calibsest_folder_path + '/' + color_setting_filename + '.pickle'
    if os.path.exists(file_path): 
        return pickle.load(open(file_path, 'rb')) 
    return None
################################################################################################################################################

