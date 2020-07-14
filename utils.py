import cv2
import pickle
import ctypes
import os
import copy
import sys
# import win32api
import numpy as np
import pandas as pd
import itertools
import imutils
import json
import time as time_
import csv
import glob
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime, date, time
from scipy import signal, interpolate
from collections import defaultdict


main_directory = str( Path(__file__).parents[1] )
# datime_format = '%y-%m-%d-%H-%M-%S-%f'
datime_format = '%H:%M:%S.%f'


####################################################################################################################################################
def get_time_file_path(file_name):
    return main_directory + '/data/kinect/time/' + file_name + '.txt'
####################################################################################################################################################
def get_color_video_file_path(file_name):
    return main_directory + '/data/kinect/color_video/' + file_name + '.avi'
####################################################################################################################################################
def get_camera_space_file_path(file_name):
    return main_directory + '/data/kinect/camera_space/' + file_name + '.pkl'
####################################################################################################################################################
def get_motion_file_path(file_name):
    return main_directory + '/data/kinect/markers/' + file_name + '.txt'
####################################################################################################################################################
def get_rssi_file_path(file_name):
    return main_directory + '/data/rfid/rssi/' + file_name + '.csv'
####################################################################################################################################################
def get_color_setting_file_path(file_name):
    return main_directory + '/data/kinect/calibration_setting/' + file_name + '.pickle'
####################################################################################################################################################
def get_color_image_file_path(file_name):
    return main_directory + '/data/kinect/calibration_setting/' + file_name + '.png'
####################################################################################################################################################


####################################################################################################################################################
def create_folder(file_path):
    # Create folder if it does not exist
    folder_path = str( Path(file_path).parents[0] )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
####################################################################################################################################################



