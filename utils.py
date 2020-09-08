import cv2
import pickle
import ctypes
import os
import copy
import sys
import numpy as np
import pandas as pd
import itertools
import imutils
import json
import time as time_lib
import csv
import glob
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime, date, time, timedelta
from scipy import signal, interpolate, stats
from collections import defaultdict
from math import*
import pywt
from itertools import chain, combinations
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
from ast import literal_eval

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, LSTM, SimpleRNN, Dropout, BatchNormalization, MaxPooling1D, Flatten, Conv1D, Conv2D
from sklearn.metrics import*
from keras import backend as K
from pycaret.regression import*
from IPython.display import display, clear_output

import keras
import tcn
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from keras.optimizers import*





main_directory = str( Path(__file__).parents[1] )
datime_format = '%H:%M:%S.%f'
dataset_folder_name = 'dataset'
synth_dataset_folder_name = 'synthetic_dataset'
eps = 1e-12

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
def get_rfid_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/rfid/' + file_name + '.csv'
####################################################################################################################################################
def get_arduino_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/arduino/' + file_name + '.csv'
####################################################################################################################################################
def get_color_setting_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name +  '/' + dataset_name  + '/calibration_setting/' + file_name + '.pickle'
####################################################################################################################################################
def get_sys_info(dataset_name):
    file_path = main_directory + '/' + dataset_folder_name +  '/' + dataset_name  + '/calibration_setting/sys_info.txt'
    sys_info = pd.read_csv(file_path, delimiter='\t').replace({'None': None})
    return sys_info
####################################################################################################################################################
def get_dataset_folder_path(dataset_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/data' 
####################################################################################################################################################


####################################################################################################################################################
def create_folder(file_path):
    # Create folder if it does not exist
    folder_path = str( Path(file_path).parents[0] )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
####################################################################################################################################################



