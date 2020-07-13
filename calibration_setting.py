from utils import*

####################################################################################################################################################
def get_image( image_name ):
    kinect = KINECT()
    frame = kinect.read(full_data=False)
    frame.show( wait=0 )
        
    file_path = calibsest_folder_path + '/' + image_name + '.png'
    create_folder(file_path)                                       

    cv2.imwrite(file_path, frame.color_image) 
    return
####################################################################################################################################################
def get_color_range(hsv_pixels, perc_th=0.1):
    low, high = list(), list()
    for i in range(3):
        counts,bins = np.histogram(hsv_pixels[:,i], bins=np.arange(257))
        bins = bins[:-1]
        bins = bins[ counts/np.max(counts) > perc_th]
        low.append(np.min(bins))
        high.append(np.max(bins))    
    return np.array(low), np.array(high)
####################################################################################################################################################
def get_color_setting(image_name, color_names):
    # Load image and  add ".png"
    image_path =  get_color_image_file_path(image_name)
    image = cv2.imread(image_path)

    # Get gray and hsv image
    n_colors = len( color_names)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    # image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # Find contuors
    contours, _ = cv2.findContours(cv2.inRange(image_gray, 0, 250), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:n_colors]

    # Show image and contours
    image_ = image.copy()
    cv2.drawContours(image_, contours, -1, color=(0,0,255), thickness=2) 
    cv2.imshow(" Image ", image_)
    cv2.waitKey(0)

    # sort contours from left to right
    X = list()
    for i, contour in enumerate(contours):
        (x,_),_ = cv2.minEnclosingCircle(contour)    
        X.append(x)
    contours = [c for _,c in sorted(zip(X,contours))]

    # get color ranges
    color_setting = dict()
    for i in range(n_colors):
        image_ = np.zeros_like(image)
        box = np.int0( cv2.boxPoints( cv2.minAreaRect(contours[i]) ) )
        image_ = cv2.drawContours(image_,[box], 0, (255,255,255), -1)            
        pixels = np.where(cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY))
        hsv_pixels = image_hsv[pixels]
        color_range = get_color_range(hsv_pixels, perc_th=0.01)
        color_setting.update({ color_names[i]: color_range })

        cv2.imshow( color_names[i], cv2.bitwise_and(image, image_) )
        cv2.waitKey(0)

    return color_setting
####################################################################################################################################################
def update_color_setting(image_name, color_names, color_setting_old=None):
    # Load image and  add ".png"
    image_path =  get_color_image_file_path(image_name)
    image = cv2.imread(image_path)

    # Get gray and hsv image
    n_colors = len( color_names)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    # image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # Find contuors
    contours, _ = cv2.findContours(cv2.inRange(image_gray, 0, 250), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:n_colors]

    # Show image and contours
    image_ = image.copy()
    cv2.drawContours(image_, contours, -1, color=(0,0,255), thickness=2) 
    cv2.imshow(" Image ", image_)
    cv2.waitKey(0)

    # sort contours from left to right
    X = list()
    for i, contour in enumerate(contours):
        (x,_),_ = cv2.minEnclosingCircle(contour)    
        X.append(x)
    contours = [c for _,c in sorted(zip(X,contours))]

    # get color ranges

    if color_setting_old is None:
        color_setting_new = dict()
    else:
        color_Setting_new = color_setting_old
        print(type(color_setting_old))

    for i in range(n_colors):
        image_ = np.zeros_like(image)
        box = np.int0( cv2.boxPoints( cv2.minAreaRect(contours[i]) ) )
        image_ = cv2.drawContours(image_,[box], 0, (255,255,255), -1)            
        pixels = np.where(cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY))
        hsv_pixels = image_hsv[pixels]
        color_range = get_color_range(hsv_pixels, perc_th=0.01)
        color_setting_new.update({ color_names[i]: color_range })

        cv2.imshow( color_names[i], cv2.bitwise_and(image, image_) )
        cv2.waitKey(0)

    return color_setting_new
####################################################################################################################################################
def save_color_setting(color_setting, file_name):
    file_path = get_color_setting_file_path(file_name)
    pickle.dump( color_setting, open(file_path, 'wb'))    # Save as .pickle
    return
####################################################################################################################################################
def load_color_setting(file_name):
    color_setting_file_path = get_color_setting_file_path(file_name)
    with open(color_setting_file_path, "rb") as f: 
        return pickle.load(f)
####################################################################################################################################################


####################################################################################################################################################
if __name__ == '__main__':

    # image_name = 'color_setting_ref'                                                            # png file name (input: reference image)
    # color_names = ['yellow', 'red', 'green', 'blue', 'purple', 'orange']                        # color names from left to right of image (input)
    # color_range_file_name = 'color_setting_default'                                             # pickle file name (saved output: color ranges)
        

    color_setting_name = 'color_setting_default'                                               # pickle file name (saved output: color ranges)
    color_setting_old_name = 'color_setting_default'
    
    image_name = 'color_setting_g'                                                                 # png file name (input: reference image)
    color_names = ['green']                                                                        # color names from left to right of image (input)

    # get_image( image_name)                                                                      # Get Image

    new_colors_setting = get_color_setting(image_name,  color_names)  # get dict() of color ranges

    color_setting = load_color_setting(color_setting_old_name)
    for color, setting in new_colors_setting.items(): color_setting.update({ color:setting })

    save_color_setting( color_setting, color_setting_name )    
####################################################################################################################################################
