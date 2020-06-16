from utils import*

####################################################################################################################################################
def get_image(image_path):
    kinect = KINECT()
    frame = kinect.read()
    frame.show( wait=0 )
        
    # add ".png"
    if image_path[-4:] != '.png': image_path_ = image_path + '.png'
    else: image_path_ = image_path

    # Create folder if it does not exists
    folder_path = str( Path(image_path_).parents[0] )
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    cv2.imwrite(image_path_, frame.color_image) 
    return
####################################################################################################################################################
def get_range(hsv_pixels, perc_th=0.1):
    low, high = list(), list()
    for i in range(3):
        counts,bins = np.histogram(hsv_pixels[:,i], bins=np.arange(257))
        bins = bins[:-1]
        bins = bins[ counts/np.max(counts) > perc_th]
        low.append(np.min(bins))
        high.append(np.max(bins))    
    return np.array(low), np.array(high)
####################################################################################################################################################
def get_color_setting(image_path, color_names):
    # Load image and  add ".png"
    if image_path[-4:] != '.png': image_path_ = image_path + '.png'
    else: image_path_ = image_path
    image = cv2.imread(image_path_)

    # Get gray and hsv image
    n_colors = len( color_names)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Find contuors
    contours, _ = cv2.findContours(cv2.inRange(image_gray, 0, 250), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:n_colors]

    # Show image and contours
    image_ = image.copy()
    cv2.drawContours(image_, contours, -1, color=(0,0,255), thickness=5) 
    cv2.imshow(" Image ", image_)
    cv2.waitKey(0)

    # sort contours from left to right
    X = list()
    for i, contour in enumerate(contours):
        (x,_),_ = cv2.minEnclosingCircle(contour)    
        X.append(x)
    contours = [c for _,c in sorted(zip(X,contours))]

    color_setting = dict()
    for i in range(n_colors):
        image_ = np.zeros_like(image)
        box = np.int0( cv2.boxPoints( cv2.minAreaRect(contours[i]) ) )
        image_ = cv2.drawContours(image_,[box], 0, (255,255,255), -1)            
        pixels = np.where(cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY))
        hsv_pixels = image_hsv[pixels]
        low, high = get_range(hsv_pixels)
        color_setting.update({ color_names[i]: (low, high) })

        cv2.imshow( color_names[i], cv2.bitwise_and(image, image_) )
        cv2.waitKey(0)
    return color_setting
####################################################################################################################################################


####################################################################################################################################################
if __name__ == '__main__':
        
    # Get Image
    # get_image( calibsest_folder_path + '/' + image_name)

    image_name = 'color_ranges_ref_image'                                                       # png file name (input: reference image)
    color_names = ['yellow', 'red', 'green', 'blue', 'purple', 'orange']                        # color names from left to right of image (input)
    color_range_file_name = 'color_ranges_default'                                              # pickle file name (saved output: color ranges)
     
    color_ranges = get_color_setting( calibsest_folder_path +'/' + image_name,  color_names )    # get dict() of color ranges
    save_pickle(color_ranges, calibsest_folder_path + '/' + color_range_file_name)              # save as pickle   
####################################################################################################################################################
