from utils import*



################################################################################################################################################
class RECORD(object):
################################################################################################################################################    
    def __init__(self, dataset_name, file_name, color_setting_filename='color_setting_default'):
        color_setting_file_path = get_color_setting_file_path(dataset_name, color_setting_filename)
        with open(color_setting_file_path, "rb") as f: 
            self.color_setting = pickle.load(f)

        camera_space_file_path = get_camera_space_file_path(dataset_name, file_name) 
        with open(camera_space_file_path, "rb") as f: 
            self.camera_spaces = pickle.load(f)

        video_file_path = get_color_video_file_path(dataset_name, file_name)
        self.color_vid = cv2.VideoCapture(video_file_path)

        self.next_idx = 0 
    ############################################################################################################################################        
    def read(self):
        success, frame = self.color_vid.read()          
        if success and self.next_idx<len(self.camera_spaces):
            self.frame = frame
            self.camera_space = self.camera_spaces[self.next_idx]
            self.next_idx += 1        
            return True
        return False      
    ############################################################################################################################################
    def draw_contours(self, contours, color):
        cnts = list()
        for contour in contours:
            if len(contour): cnts.append(contour)
        cv2.drawContours(self.frame, cnts, -1, color=color, thickness=3) 
        return
    ############################################################################################################################################
    def show(self, contours=None, wait=None):
        if self.frame is not None:
            if contours is not None: cv2.drawContours(self.frame, contours, -1, color=(255,255,0), thickness=3) 
            frame = cv2.resize( self.frame, None, fx=0.5,fy=0.5 )       
            cv2.imshow('Color View', frame)
        
            if wait is not None: cv2.waitKey(wait)
        return  
    ############################################################################################################################################
    def get_colored_circles(self, color, n_circles=3): 
        frame = self.frame.copy() 
        color_range = self.color_setting[color]
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred , cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_range[0], color_range[1])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        pixels = np.transpose(np.nonzero(mask)) 
        circles = [[]]*n_circles

        if len(pixels) > n_circles: 

            kmeans = KMeans(n_clusters=n_circles, random_state=0).fit(pixels)
            for n in range(n_circles):

                mask = np.zeros_like(mask)
                for pixel in pixels[ kmeans.labels_== n, :]: mask[ pixel[0], pixel[1] ] = 255            
                
                contours , _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
                contour = max(contours, key=cv2.contourArea)
                
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                M = cv2.moments(contour)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                mask = np.zeros_like(mask)            
                cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
                circle = max(contours, key=cv2.contourArea)

                circles[n] = circle

        return circles
    ############################################################################################################################################
    def get_pixels(self, contour):
        mask = np.zeros((self.frame.shape[0], self.frame.shape[1],3))
        mask = cv2.drawContours(mask, [contour], 0, (1,0,0), thickness=cv2.FILLED)
        x,y = np.where(mask[:,:,0]==1)
        return np.stack((x,y),axis=-1)
    ############################################################################################################################################
    def get_locations(self, contours):
        locations = list()
        for contour in contours:
            if len(contour):               
                pixels = self.get_pixels(contour) 
                camera_points = list()
                for pixel in pixels: 
                    camera_point = self.camera_space[pixel[0], pixel[1]]
                    if not np.any(camera_point) or not np.all( np.isfinite(camera_point)): camera_point = [np.nan, np.nan, np.nan]                    
                    camera_points.append( camera_point ) 
                location = np.nanmean(camera_points, axis=0) 
            else:
                location = [np.nan, np.nan, np.nan]
            locations = [*locations, *location]
        return locations     
    ############################################################################################################################################



################################################################################################################################################
if __name__ == '__main__':

    dataset_name = 'dataset_02'
    colors = ['red','blue','green'] 

    for n in range(5):
        file_name = 'record_' + "{0:0=2d}".format(n)

        record = RECORD( dataset_name, file_name, color_setting_filename='color_setting_default')
        motions_dict = defaultdict(list)
        
        # Loop over frames 
        while True:
            success = record.read()
            if not success: break

            for i, color in enumerate(colors):
                circles = record.get_colored_circles(color, n_circles=3)
                record.draw_contours(circles, color=30*i) 
                locations = record.get_locations(circles)              
                motions_dict[color].append(locations)
            record.show(wait=1)

        # Save
        for color, motion in motions_dict.items():        
            markers_file_path = get_markers_file_path(dataset_name, file_name + '_' + color)
            create_folder(markers_file_path)
            np.savetxt(markers_file_path, np.array(motion), delimiter="\t", fmt='%s')     
        
        print(file_name)   
################################################################################################################################################    
    
