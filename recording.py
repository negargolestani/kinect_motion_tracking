from utils import* 

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime


####################################################################################################################################################
class KINECT(object):
    ################################################################################################################################################
    def __init__(self, top_margin=0.1, bottom_margin=0.1, left_margin=0.15, right_margin=0.15):
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
    def record(self, dataset_name, file_name, record_time=40):
        
        # Initialization
        time_file_path = get_time_file_path(dataset_name, file_name)
        color_video_file_path = get_color_video_file_path(dataset_name, file_name)
        camera_space_file_path = get_camera_space_file_path(dataset_name, file_name) 

        create_folder(time_file_path)
        create_folder(color_video_file_path)
        create_folder(camera_space_file_path)

        time_txt = ''
        color_vid = cv2.VideoWriter(color_video_file_path, cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (self.kinect.color_frame_desc.Width,self.kinect.color_frame_desc.Height))

        # show when kinect starts recording
        self.read(full=False)
        print('Recording is Started')
        print('Press "Esc" Key to Stop Recording')
        time_.sleep(2)

        # Recording loop
        camera_space = list()
        start_time = None

        while cv2.waitKey(1)!=27:                        
            self.read(full=True)
            if start_time is None: start_time = datetime.combine(date.min, self.time)
            elif  (datetime.combine(date.min, self.time) -start_time).total_seconds() > record_time: break
            self.show()
            time_txt += self.time.strftime( datime_format )  + '\n'  
            color_vid.write( cv2.cvtColor(self.color_image, cv2.COLOR_RGBA2RGB) )            
            camera_space.append(self.camera_space)

        cv2.destroyAllWindows()
        print('Recording is Finished')        
        print('Wait for Processing ...')        

        with open(time_file_path, 'w') as f: f.write( time_txt)   
        with open(camera_space_file_path,"wb") as f: pickle.dump(camera_space, f)            

        print('Done!')
        return        
####################################################################################################################################################



################################################################################################################################################
if __name__ == '__main__':
    dataset_name = 'dataset_03'
    file_name='record_18'

    kinect = KINECT( 
        top_margin=0.15, 
        bottom_margin=0.15, 
        left_margin=0.2, 
        right_margin=0.2)            
    kinect.record(dataset_name, file_name, record_time=60)
################################################################################################################################################
        