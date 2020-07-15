from utils import*



####################################################################################################################################################
def resample_df(df, new_time):
    new_time_df = pd.DataFrame({'time':new_time})
    new_df = df.merge( new_time_df, on='time', how='outer', suffixes=('', ''), sort=True )
    new_df = new_df.interpolate(method='nearest')
    
    new_df = new_df.merge( new_time_df, on='time', how='inner', suffixes=('', ''), sort=True )
    return new_df
####################################################################################################################################################



####################################################################################################################################################
class NODE(object):
    n_marker = 3
    ################################################################################################################################################
    def __init__(self, markers_color, IDD=None):
        self.markers_color = markers_color
        self.IDD = IDD

        self.markers = None
        self.rssi = None
    ################################################################################################################################################
    def load_rssi(self, dataset_name, file_name, window_length=11):
        # Load data 
        rssi_file_path = get_rssi_file_path(dataset_name, file_name)       
        raw_df = pd.read_csv(
            rssi_file_path,                                                     # relative python path to subdirectory
            delimiter  = ';',                                                   # Tab-separated value file.
            usecols = ['IDD', 'Time', 'Ant/RSSI'],                              # Only load the three columns specified.
            parse_dates = ['Time'] )                                            # Intepret the birth_date column as a date      

        # self.rssi.loc[ self.rssi.IDD != self.IDD,'rssi'] = np.nan
        raw_df = raw_df.loc[ raw_df['IDD'] == self.IDD, :]
        self.rssi = pd.DataFrame({ 'rssi': raw_df['Ant/RSSI'].str.replace('Ant.No 1 - RSSI: ', '').astype(int) })

        # Processing
        self.rssi = self.rssi.rolling(window_length, axis=0).median()   # Smoothing
        self.rssi = self.rssi.ffill(axis=0).bfill(axis=0)               # Gap Filling
         
        # Time
        date_time = pd.to_datetime( raw_df['Time'] , format=datime_format)
        self.rssi['time'] = [ np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time]

        return
    ################################################################################################################################################
    def load_markers(self, dataset_name, file_name, window_length=11):   
      
        # Load raw data 
        motion_file_path = get_markers_file_path(dataset_name, file_name + '_' + self.markers_color)  
        self.markers = pd.read_csv( motion_file_path, delimiter="\t", header=None, dtype=np.float64)
                
        # Processing
        self.markers = self.markers.rolling(window_length, axis=0).mean()    # Gap filling
        self.markers = self.markers.ffill(axis=0).bfill(axis=0)              # Smoothing

        # Time
        time_file_path = get_time_file_path(dataset_name, file_name)    
        with open(time_file_path , 'r') as f:  lines = f.read().splitlines() 
        date_time = pd.to_datetime( lines , format=datime_format)
        self.markers['time'] = [ (datetime.combine(date.min, t.time())-datetime.min).total_seconds() for t in date_time]
        
        return
    ################################################################################################################################################
    def shift_time(self, shift):
        self.markers.time -= shift
        if self.rssi is not None: self.rssi.time -= shift
        return
    ################################################################################################################################################
    def center(self, window_length=7, polyorder=1):
        markers_npy = self.markers.drop(['time'], axis=1).to_numpy()
        markers_npy = markers_npy.reshape(np.shape(markers_npy)[0], -1, 3)
        center = np.mean(markers_npy, axis=1) 
        
        center = signal.savgol_filter( center, window_length=window_length, polyorder=polyorder, axis=0)            
        return center
    ################################################################################################################################################    
    def norm(self, window_length=7, polyorder=1):
        markers_npy = self.markers.drop(['time'], axis=1).to_numpy()
        # markers_npy = signal.savgol_filter( markers_npy, window_length=window_length, polyorder=polyorder, axis=0)  

        markers_npy =  markers_npy.reshape(np.shape(markers_npy)[0], -1, 3)
        v1 = markers_npy[:,1,:] - markers_npy[:,0,:]
        v2 = markers_npy[:,2,:] - markers_npy[:,0,:] 
        norm = np.cross( v1, v2)
        norm[ norm[:,2]<0, :] *= -1 
        norm = signal.savgol_filter( norm, window_length=window_length, polyorder=polyorder, axis=0)  
        norm = norm / ( np.reshape(np.linalg.norm(norm, axis=1), (-1,1)) * np.ones((1,3))) 
        return norm
####################################################################################################################################################
class SYSTEM(object):
    ################################################################################################################################################
    def __init__(self, reader_markers_color='red'):
        self.reader = NODE( reader_markers_color )
        self.tags = list()        
    ################################################################################################################################################
    def add_tag(self, markers_color, IDD):
        self.tags.append( NODE(markers_color, IDD) )   
        return               
    ################################################################################################################################################
    def load(self, file_name):
        # Loading + time shift
        
        self.reader.load_markers(file_name) 
        start_time = self.reader.markers.time.iloc[0]

        for i,tag in enumerate(self.tags):            
            self.tags[i].load_markers(file_name)
            self.tags[i].load_rssi(file_name) 
            start_time = min( start_time, tag.rssi.time.iloc[0])

        self.reader.shift_time(start_time)    
        for i,tag in enumerate(self.tags): self.tags[i].shift_time(start_time)
        
        return
    ################################################################################################################################################
    def get_rssi_data(self, smooth=True, window_length=5):
        rssi = pd.DataFrame({'time':self.tags[0].rssi.time})
        for i, tag in enumerate(self.tags):
            tag_rssi = sys.tags[i].rssi.rename({'rssi':'rssi_'+str(i)}, axis=1)
            rssi = rssi.merge( tag_rssi, on='time', how='outer', suffixes=('', ''), sort=True )

        rssi.fillna(method='ffill', axis=0, inplace=True)             

        return rssi
    ################################################################################################################################################
    def get_motion_data(self, smooth=True, window_length=5):   
        # Modify this function for different targeted movements
        motion = pd.DataFrame({'time':self.reader.markers.time})
        ref_center = self.reader.center()
        ref_norm = self.reader.norm()            

        for i, tag in enumerate(self.tags):
            motion['distance_'+str(i)] = np.linalg.norm( ref_center - tag.center(), axis=1)
            motion['misalignment_'+str(i)] = ( np.arccos(np.abs(np.sum(np.multiply( ref_norm, tag.norm()), axis=1)) ) * 180/np.pi )

        motion.fillna(method='ffill', axis=0, inplace=True)  

        return motion
    ################################################################################################################################################
    def get_data(self):
        rssi = self.get_rssi_data()
        motion = self.get_motion_data()

        data = rssi.merge( motion, on='time', how='outer', suffixes=('', ''), sort=True )
        data = data.interpolate(method='nearest')
        
        new_time = rssi.time
        new_time = new_time[ motion.time.iloc[0] < new_time]
        new_time = new_time[ new_time < motion.time.iloc[-1] ]
        data = data.merge( new_time, on='time', how='inner', suffixes=('', ''), sort=True )

        for i in range(len(self.tags)):
            data = data.astype({ 'rssi_'+str(i) : int, 'distance_'+str(i) : float, 'misalignment_'+str(i) : float })                

        return data
####################################################################################################################################################


      

####################################################################################################################################################
if __name__ == '__main__':

    sys = SYSTEM(reader_markers_color='red')
    sys.add_tag('blue', 'E002240002749F45')
    sys.add_tag('green', 'E002240002819E59')
    
    dataset_name = 'dataset_01'

    data = pd.DataFrame()
    for n in range(1,10):
        file_name = 'record_' + "{0:0=2d}".format(n)
        sys.load(file_name) 

        data = data.append( sys.get_data() , ignore_index = True)

    # print(data.head())
    data.to_pickle( get_dataset_file_path(dataset_name) )  
####################################################################################################################################################



# ####################################################################################################################################################
# if __name__ == '__main__':

#     sys = SYSTEM(reader_markers_color='red')
#     sys.add_tag('blue', 'E002240002749F45')
#     sys.add_tag('green', 'E002240002819E59')
    
#     file_name = 'record_01'       
#     sys.load(file_name)    

#     rssi = sys.get_rssi_data()
#     motion = sys.get_motion_data()
#     motion_ = resample_df(motion, rssi.time)

   
#     y = 'misalignment_0'
#     ax = motion.plot(x='time', y=y)
#     motion_.plot(x='time', y=y, ax=ax)

#     rssi.plot(x='time')   
#     plt.show()
# ####################################################################################################################################################

   
    