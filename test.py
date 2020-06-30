from utils import*
from mpl_toolkits.mplot3d import Axes3D
from mapper import*
################################################################################################################################################
def test1():
    kinect = PyKinectRuntime.PyKinectRuntime( PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color) 
    # color_setting = load_color_setting()    
    # color_range = color_setting['blue']
    
    
    while  cv2.waitKey(1) != 27:
    # while True:
        if kinect.has_new_color_frame() and kinect.has_new_depth_frame():                    
            
            time = datetime.now()
            color_frame = kinect.get_last_color_frame()
            depth_frame = kinect.get_last_depth_frame()
            depth_frameD = kinect._depth_frame_data

            color_image = color_frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4))               
            color_image =cv2.cvtColor(color_image, cv2.COLOR_RGBA2RGB)               
            depth_image = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width)).astype(np.uint8)
            # depth_image =cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)               


            # (r, c) = np.shape(depth_image)
            # (X, Y) = np.meshgrid(range(0, c, 1), range(0, r, 1))
            # plt.imshow(depth_image)
            # plt.scatter(X, Y, depth_image[Y,X])
            # plt.show()

            # xx, yy = np.mgrid[0:depth_image.shape[0], 0:depth_image.shape[1]]
            # fig = plt.figure(figsize=(15,15))
            # ax = fig.gca(projection='3d')
            # ax.plot_surface(xx, yy, depth_image ,rstride=1, cstride=1,linewidth=2)
            # ax.view_init(80, 30)
            # plt.show()


         
            # print(np.shape(camera_space))
            cv2.imshow('', cv2.resize( color_image, None, fx=0.5,fy=0.5 ) )
            cv2.waitKey(1)

            # depth_frame_data = np.ctypeslib.as_ctypes(self.depth_frame.flatten())
            # depth_frame_data = np.ctypeslib.as_ctypes(self.kinect_._depth_frame_data)

           
            
################################################################################################################################################
def test2():
    with open('tmp.pkl',"rb") as f:
        frame = pickle.load(f) 


    color_setting = load_color_setting()
    for color in ['red', 'blue']:
        color_range = color_setting[color]
        circles = frame.get_colored_circle_(color_range, n_circles=3)
        locations = np.round(np.multiply( frame.get_location(circles) , 100), 1)
        # dist = list()
        # for location in locations[1:]:
        #     dist.append( np.linalg.norm(location - locations[0]) )
        print(locations)
    
        frame.show(contours=circles, wait=0)   

################################################################################################################################################

################################################################################################################################################
if __name__ == '__main__':
    test1()
################################################################################################################################################
