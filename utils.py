import numpy as np
import cv2
import pickle
import ctypes
import os
import copy
import sys
import math
from datetime import datetime
from pathlib import Path

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

delimator = ';'
datime_format = '%y-%m-%d-%H-%M-%S'

main_directory = str( Path(__file__).parents[1] )
records_folder_path = main_directory + '/data/records'
markers_folder_path = main_directory + '/data/markers'

calibsest_folder_path = main_directory + '/calibration_setting'
color_range_filename =  'color_ranges_default'



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
                    camera_space = self.get_camera_space(self.kinect._depth_frame_data)
                    FRAME(time, color_image, depth_image, camera_space)
                else:
                   return FRAME(time, color_image, depth_image, depth_frame_) # return depth_frame_ instead of camera_space for faster recording 
    ################################################################################################################################################    
    def record(self):
        print('Recording is Started')
        print('Press "Esc" Key to Stop Recording')

        # Recording loop
        rec = RECORD()
        while cv2.waitKey(1) != 27: 
            frame = self.read(full_data=False)
            frame.show()

            rec.add_value(frame) 

        # Get Camera Space after recording loop
        for i, frame in enumerate(rec.frames):
            rec.frames[i].camera_space = self.get_camera_space( frame.camera_space )
        
        print('Recording is Finished')        
        return rec
####################################################################################################################################################



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
    def get_markers_location(self, color_range, n_markers):
        markers_location = list()
        frame_contours = self.color_2_circle_contours(color_range, n_Contours=n_markers)    
        for contour in frame_contours:
            color_points, _ = self.contour_2_color_points(contour)              # Targeted points in color space
            camera_points = self.color_points_2_camera_points(color_points)     # Targeted points in camera space
            marker_location = np.nanmean(camera_points, axis=0)                 # Average of targeted points in camera space -> Markers center
            markers_location.append( marker_location )
        return markers_location
####################################################################################################################################################
class RECORD(object):
    ################################################################################################################################################    
    def __init__(self, file_name=None):
        self.frames = list()  
        if file_name is not None: self.load(file_name)                   
    ################################################################################################################################################    
    def set_value(self, frames):
        self.frames = frames
    ################################################################################################################################################    
    def add_value(self, frame):
        self.frames.append(frame)
    ################################################################################################################################################    
    def get_makers_motion(self, color_range, n_markers=2):    
        motion = MOTION()
        for frame in self.frames:
            time = frame.time
            locations = frame.get_markers_location(color_range, n_markers) 
            motion.add_value( frame.time, locations)
        return motion
    ################################################################################################################################################    
    def save(self, file_name):
        file_path = file_path = records_folder_path + '/' + file_name + '.pickle'
        create_folder(file_path)                            # Create folder if it does not exist    
        pickle.dump( self.frames, open(file_path, 'wb'))    # Save as .pickle
        return True
    ################################################################################################################################################    
    def load(self, file_name):
        file_path = file_path = records_folder_path + '/' + file_name + '.pickle'
        if os.path.exists(file_path): 
            self.frames = pickle.load(open(file_path, 'rb')) 
################################################################################################################################################    
class MOTION(object):
    ############################################################################################################################################
    def __init__(self, file_name=None):
        self.locations = list()        
        self.times = list()
        if file_name is not None: self.load(file_name)
    ############################################################################################################################################
    def set_value(self, times, locations):
        self.times = times
        self.locations = locations
    ############################################################################################################################################
    def add_value(self, time, locations):
        self.times.append(time)
        self.locations.append(locations)
    ############################################################################################################################################
    def save(self, file_name):
        # Save markers' center  (camera space) as "txt" file 
        file_path = markers_folder_path + '/' + file_name + '.txt'
        create_folder(file_path)                                    # Create folder if it does not exist    

        # Markers (txt)
        data_txt = ''
        for i in range(len(self.times)):  
            data_txt += self.times[i].strftime( datime_format )                    # Markers (txt)
            for m in np.array(self.locations[i]).flatten(): 
                data_txt +=  delimator + str(m)  
            data_txt += '\n'

        # Save 
        with open(file_path, 'w') as f:
            f.write( data_txt )   
            
        return
    ############################################################################################################################################
    def load(self, file_name):
        file_path = markers_folder_path + '/' + file_name + '.txt'
        if os.path.exists(file_path):
            with open(file_path , 'r') as f:
                lines = f.read().splitlines() 

            self.times, self.locations = list(), list()
            for line in lines:
                data_txt = line.split( delimator )
                frame_time = datetime.strptime(data_txt[0], datime_format )
                frame_markers_location = np.array( [ float(x) for x in data_txt[1:]] ).reshape((-1,3))        
                self.times.append(frame_time)
                self.locations.append( frame_markers_location )
####################################################################################################################################################





####################################################################################################################################################
def create_folder(file_path):
    # Create folder if it does not exist
    folder_path = str( Path(file_path).parents[0] )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
####################################################################################################################################################

