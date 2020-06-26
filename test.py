from utils import*




################################################################################################################################################
def test1():
    file_name = 'record_02'
    
    motions = load_motion(file_name + '_red')
    
    raw = list()    
    for i in range(3): raw.append( motions[:,i*3:(i+1)*3] )    


    df = pd.DataFrame(motions)
    df.fillna(method='ffill', axis=0, inplace=True)                
    motions = df.to_numpy()        
    motions = signal.savgol_filter( motions, window_length=21, polyorder=1, axis=0)  

           
    m = list()    
    for i in range(3): 
        m.append( motions[:,i*3:(i+1)*3] )    
        # plt.plot(m[i])
    # plt.show()


    v1 = m[1] - m[0]
    v2 = m[2] - m[0]
    # norm = np.cross(v1, v2)
    # self.norm = norm / ( np.reshape(np.linalg.norm(norm, axis=1) + 1e-12, (-1,1)) * np.ones((1,3)) )
    # self.center = np.mean( self.markers_motion, axis=0)     
    plt.plot(v1)
    plt.plot(v2)
    plt.show()
################################################################################################################################################ 
def test2():    
    file_name = 'record_03'
    colors = ['blue', 'red']

    # Load
    record = load_record( file_name )
    color_setting = load_color_setting()    

    # Outputs dict
    motions = dict()
    for color in colors: motions.update({color:list()})

    # Tracking loop 
    for frame in record[:-2]:
        contours = list()
        mask = np.zeros((1080, 1920))

        for color in colors:
            frame_circles, mask_ = frame.get_colored_circle( color_setting[color], n_circles=3, radius=10)
            contours = [*contours, *frame_circles]            

            # locations = frame.get_location(frame_circles)
            # motions[color].append( locations )
            mask += mask_

               
        cv2.imshow('', cv2.resize( mask, None, fx=0.5,fy=0.5 ))
        cv2.waitKey(10)
        # frame.show(contours=contours, wait=1)
################################################################################################################################################ 




################################################################################################################################################ 
if __name__ == '__main__':
    test2()
################################################################################################################################################ 
