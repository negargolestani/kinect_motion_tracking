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
datime_format = '%H:%M:%S.%f'
dataset_folder_name = 'dataset'

####################################################################################################################################################
def get_time_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/kinect/time/' + file_name + '.txt'
####################################################################################################################################################
def get_color_video_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/kinect/color_video/' + file_name + '.avi'
####################################################################################################################################################
def get_camera_space_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/kinect/camera_space/' + file_name + '.pkl'
####################################################################################################################################################
def get_markers_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/kinect/markers/' + file_name + '.txt'
####################################################################################################################################################
def get_rssi_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/rfid/' + file_name + '.csv'
####################################################################################################################################################
def get_dataset_file_path(dataset_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/' + dataset_name + '.pkl'
####################################################################################################################################################
def get_color_setting_file_path(file_name):
    return main_directory + '/' + dataset_folder_name + '/calibration_setting/' + file_name + '.pickle'
####################################################################################################################################################




####################################################################################################################################################
def create_folder(file_path):
    # Create folder if it does not exist
    folder_path = str( Path(file_path).parents[0] )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
####################################################################################################################################################



