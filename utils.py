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
import imutils
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime
from scipy import signal



main_directory = str( Path(__file__).parents[1] )

datime_format = '%y-%m-%d-%H-%M-%S'


####################################################################################################################################################
def get_time_file_path(file_name):
    return main_directory + '/data/time/' + file_name + '.txt'
####################################################################################################################################################
def get_video_file_path(file_name):
    return main_directory + '/data/record/color_video/' + file_name + '.avi'
####################################################################################################################################################
def get_camera_space_file_path(file_name):
    return main_directory + '/data/record/camera_space/' + file_name + '.pkl'
####################################################################################################################################################
def get_color_setting_file_path(file_name):
    return main_directory + '/calibration_setting/' + file_name + '.pickle'
####################################################################################################################################################
def get_motion_file_path(file_name):
    return main_directory + '/data/markers/' + file_name + '.txt'
####################################################################################################################################################



####################################################################################################################################################
def load_times(file_name):
    time_file_path = get_time_file_path(file_name)
    
    if os.path.exists(time_file_path):
        with open(time_file_path , 'r') as f: 
            lines = f.read().splitlines() 

        times = list()
        for line in lines: 
            times.append( datetime.strptime(line, datime_format) )
        
        return times
    return None
####################################################################################################################################################
def create_folder(file_path):
    # Create folder if it does not exist
    folder_path = str( Path(file_path).parents[0] )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
####################################################################################################################################################



