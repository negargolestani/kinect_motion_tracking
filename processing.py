from utils import*

####################################################################################################################################################
def resample_df(df, new_time, est_kind='linear'):
    # Resample df1 at new sample-times 
    new_df = pd.DataFrame({'time':new_time})
    for column in df.columns[df.columns!='time']:
        func = interpolate.interp1d(df.time, df[column], kind=est_kind)
        new_df[column] = func(new_time)
    return new_df
####################################################################################################################################################
def normalize_(x):
    x_ = np.array(x_)
    return ( x_ - np.nanmean(x_)) / np.nanstd(x_)
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
    def load_rssi(self, file_name):
        # Load data 
        rssi_file_path = get_rssi_file_path(file_name)       
        self.rssi = pd.read_csv(
            rssi_file_path,                                                     # relative python path to subdirectory
            delimiter  = ';',                                                   # Tab-separated value file.
            usecols = ['IDD', 'Time', 'Ant/RSSI'],                              # Only load the three columns specified.
            parse_dates = ['Time'] )                                            # Intepret the birth_date column as a date      

        self.rssi.rename({'Ant/RSSI':'rssi', 'Time':'time'}, axis=1, inplace=True)
        self.rssi.rssi = self.rssi.rssi.str.replace('Ant.No 1 - RSSI: ', '').astype(int)   
        date_time = pd.to_datetime( self.rssi.time , format=datime_format)
        self.rssi.time = [ np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time]
        
        # self.rssi.loc[ self.rssi.IDD != self.IDD,'rssi'] = np.nan
        self.rssi = self.rssi.loc[ self.rssi.IDD == self.IDD]
        self.rssi.drop(['IDD'], axis=1, inplace=True)

        # self.rssi.fillna(method='ffill', inplace=True, axis=0)                                         
        return
    ################################################################################################################################################
    def load_markers(self, file_name):   
      
        # Load clean data 
        motion_file_path = get_motion_file_path(file_name + '_' + self.markers_color)  
        self.markers = pd.read_csv( motion_file_path, delimiter="\t", header=None, dtype=np.float64)
        
        # Time
        time_file_path = get_time_file_path(file_name)    
        with open(time_file_path , 'r') as f:  lines = f.read().splitlines() 
        date_time = pd.to_datetime( lines , format=datime_format)
        self.markers['time'] = [ (datetime.combine(date.min, t.time())-datetime.min).total_seconds() for t in date_time]

        # self.markers.fillna(method='ffill', inplace=True, axis=0)                                
        return
    ################################################################################################################################################
    def shift_time(self, shift):
        self.markers.time -= shift
        if self.rssi is not None: self.rssi.time -= shift
        return
    ################################################################################################################################################
    def center(self, window_length=5, polyorder=1):
        markers_npy = self.markers.drop(['time'], axis=1).to_numpy()
        markers_npy = markers_npy.reshape(np.shape(markers_npy)[0], -1, 3)
        center = np.mean(markers_npy, axis=1) 
        center = signal.savgol_filter( center, window_length=window_length, polyorder=polyorder, axis=0)            
        return center
    ################################################################################################################################################    
    def norm(self, window_length=5, polyorder=1):
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
        start_time = self.reader.markers.time[0]

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
            rssi = rssi.merge( sys.tags[i].rssi.rename({'rssi':'rssi_'+str(i)},axis=1), on='time', how='outer', suffixes=('', ''), sort=True )
        # rssi.fillna(method='ffill', axis=0, inplace=True)  
        return rssi
    ################################################################################################################################################
    def get_motion_data(self, smooth=True, window_length=5):   
    # Modify this function for different targeted movements
        motion = pd.DataFrame({'time':self.reader.markers.time})
        ref_center = self.reader.center()
        ref_norm = self.reader.norm()            

        for i, tag in enumerate(self.tags):
            motion['distance_'+str(i)] = np.linalg.norm( ref_center - tag.center(), axis=1) 
            motion['misalignment_'+str(i)] = np.arccos(np.abs(np.sum(np.multiply( ref_norm, tag.norm()), axis=1)) ) * 180/np.pi
        motion.fillna(method='ffill', axis=0, inplace=True)  
        return motion
####################################################################################################################################################


      


####################################################################################################################################################
if __name__ == '__main__':

    sys = SYSTEM(reader_markers_color='red')
    sys.add_tag('blue', 'E002240002749F45')
    sys.add_tag('green', 'E002240002819E59')
    
    file_name = 'record_06'       
    sys.load(file_name)    

    # _, ax = plt.subplots(1,1)
    # for tag in sys.tags:
    #     x = tag.norm()
    #     # x = (x**2).sum(axis=1)**0.5

    #     print(np.shape(x))
    #     # print(x)
    #     plt.plot(x)
    #     # plt.show()
    # # plt.show()

    # RSSI
    rssi = sys.get_rssi_data()
    # rssi.filter(regex='rssi',axis=1).plot()
    
    # Motion
    y = 'misalignment_1'
    motion = sys.get_motion_data()
    # ax = motion.filter(regex='distance',axis=1).plot()
    ax = motion.plot(x='time', y=y)

    motion = resample_df(motion, rssi.time)
    motion.plot(x='time', y=y, ax=ax)
    plt.show()

    # from sklearn import preprocessing
    # fig, ax = plt.subplots(1,1)

    # for data in [motion, rssi]:
    #     for col in data.columns:
    #         if col == 'time': continue
    #         float_array = data[col].values.astype(float).reshape(-1,1)
    #         scaled_array = normalize_(float_array)
    #         # min_max_scaler = preprocessing.Normalizer()
    #         # scaled_array = min_max_scaler.fit_transform(float_array)
    #         ax.plot(scaled_array)

    plt.show()
####################################################################################################################################################

   
    