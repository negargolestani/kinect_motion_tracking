from utils import* 

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

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
    ################################################################################################################################################
    # def record(self, dataset_name, file_name, record_time=40):   

    #     # Initialization
    #     time_file_path = get_time_file_path(dataset_name, file_name)
    #     create_folder(time_file_path)
    #     time_df = pd.DataFrame(columns=['time'])

    #     color_video_file_path = get_color_video_file_path(dataset_name, file_name)
    #     create_folder(color_video_file_path)
    #     color_vid = cv2.VideoWriter(color_video_file_path, cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (self.kinect.color_frame_desc.Width,self.kinect.color_frame_desc.Height))

    #     camera_space_file_path = get_camera_space_file_path(dataset_name, file_name) 
    #     create_folder(camera_space_file_path)
    #     camera_space_list = list()


    #     # Show when kinect starts recording
    #     self.read(full=False)
    #     print('Recording is Started')
    #     print('Press "Esc" Key to Stop Recording')
    #     time_lib.sleep(2)


    #     # Recording loop
    #     start_time = datetime.combine(date.min, self.time)
    #     while cv2.waitKey(1)!=27 and (datetime.combine(date.min, self.time) - start_time).total_seconds() < record_time:   

    #         self.read(full=True)
    #         self.show()

    #         time_df = time_df.append({'time':self.time}, ignore_index=True)
    #         color_vid.write( cv2.cvtColor(self.color_image, cv2.COLOR_RGBA2RGB) )            
    #         camera_space_list.append(self.camera_space)

    #     cv2.destroyAllWindows()
        
    #     print('Recording is Finished')
    #     print('Wait for Saving ...')        

    #     time_df.to_csv(time_file_path, index=False)        
    #     with open(camera_space_file_path,"wb") as f: pickle.dump(camera_space_list, f)            

    #     print('Done!')           
####################################################################################################################################################


####################################################################################################################################################
if __name__ == '__main__':
    
    dataset_name ='dataset_05'
    file_name = 'record_04'
    record_time = 60


    # Initialization
    kinect = KINECT()    

    time_file_path = get_time_file_path(dataset_name, file_name)
    create_folder(time_file_path)
    time_df = pd.DataFrame(columns=['time'])

    color_video_file_path = get_color_video_file_path(dataset_name, file_name)
    create_folder(color_video_file_path)
    color_vid = cv2.VideoWriter(color_video_file_path, cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (kinect.kinect.color_frame_desc.Width, kinect.kinect.color_frame_desc.Height))

    camera_space_file_path = get_camera_space_file_path(dataset_name, file_name) 
    create_folder(camera_space_file_path)
    camera_space_list = list()


    # Show when kinect starts recording
    kinect.read(full=False)
    print('Recording is Started \n  Press "Esc" Key to Stop Recording')
    time_lib.sleep(2)

    # Recording loop
    start_time = datetime.combine(date.min, kinect.time)
    while cv2.waitKey(1)!=27 and (datetime.combine(date.min, kinect.time) - start_time).total_seconds() < record_time:   
        kinect.read(full=True)
        kinect.show()
        time_df = time_df.append({'time':kinect.time}, ignore_index=True)
        color_vid.write( cv2.cvtColor(kinect.color_image, cv2.COLOR_RGBA2RGB) )            
        camera_space_list.append(kinect.camera_space)
    cv2.destroyAllWindows()    

    print('Recording is Finished \n Wait for Saving ...')    
    time_df.to_csv(time_file_path, index=False)        
    with open(camera_space_file_path,"wb") as f: pickle.dump(camera_space_list, f)            
   
    print('Done!')           
################################################################################################################################################
        