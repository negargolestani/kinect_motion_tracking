from utils import*

####################################################################################################################################################
class COIL(object):
    n_marker = 3
    ################################################################################################################################################
    def __init__(self, file_name):
        # Load
        motion_file_path = get_motion_file_path(file_name)    
        if os.path.exists(motion_file_path):
            self.motions = np.loadtxt(motion_file_path)
        
        self.clean_motion_data()
        self.get_params()
        return
    ################################################################################################################################################        
    def clean_motion_data(self):
        # Gap filling
        df = pd.DataFrame(self.motions)
        df.fillna(method='ffill', axis=0, inplace=True)   
        self.motions = df.to_numpy()        
        
        # Filter
        # self.motions = signal.savgol_filter( self.motions, window_length=11, polyorder=1, axis=0)  
        return
    ################################################################################################################################################        
    def get_params(self):    
        motions = list()    
        for i in range(self.n_marker): motions.append( self.motions[:,i*3:(i+1)*3] )    

        v1 = motions[1] - motions[0]
        v2 = motions[2] - motions[0]

        norm = np.cross(v1, v2) 
        self.norm = norm / ( np.reshape(np.linalg.norm(norm, axis=1) + 1e-12, (-1,1)) * np.ones((1,3)) )
        self.center = np.mean( motions, axis=0)     
    ################################################################################################################################################
    def get_relative_motion(self, coil, window_length=11, polyorder=1):
        distance =  np.linalg.norm( self.center - coil.center, axis=1) 
        ang_misalign = np.arccos(np.abs(np.sum(np.multiply(self.norm, coil.norm), axis=1)) ) * 180/np.pi

        distance = signal.savgol_filter( distance, window_length=window_length, polyorder=polyorder)  
        ang_misalign = signal.savgol_filter( ang_misalign, window_length=window_length, polyorder=polyorder)  

        return distance, ang_misalign
####################################################################################################################################################




################################################################################################################################################
if __name__ == '__main__':


    file_name = 'test'   
    reader_color = 'blue'
    tags_color = ['red']
        
   
    reader = COIL(file_name + '_' + reader_color)    
    for tag_color in tags_color:
        tag =  COIL(file_name + '_' + tag_color)  
        distance, ang_misalign = reader.get_relative_motion(tag)

        fig, ax = plt.subplots(2, 1)
        
        ax[0].plot( np.round(100*distance,2))
        ax[0].set_ylabel('Distance (cm)')

        ax[1].plot( ang_misalign )
        ax[1].set_ylabel('Misalignment (degree)')
        ax[1].set_ylim([0,90])

        plt.show()

   