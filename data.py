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

        # # Processing
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
        # norm[ norm[:,2]<0, :] *= -1 
        norm = signal.savgol_filter( norm, window_length=window_length, polyorder=polyorder, axis=0)  
        norm = norm / ( np.reshape(np.linalg.norm(norm, axis=1), (-1,1)) * np.ones((1,3))) 
        return norm
####################################################################################################################################################
class SYSTEM(object):
    ################################################################################################################################################
    def __init__(self, system_info = None):
        self.reader = None
        self.tags = list()

        if system_info is not None:
            system_info_ = system_info.copy()
            reader_markers_color = system_info_.pop('reader')
            self.add_reader(reader_markers_color)
            for IDD,markers_color in system_info_.items(): self.add_tag(markers_color, IDD)
        return
    ################################################################################################################################################
    def add_reader(self, reader_markers_color):
        self.reader = NODE( reader_markers_color )
        return
    ################################################################################################################################################
    def add_tag(self, markers_color, IDD):
        self.tags.append( NODE(markers_color, IDD) )   
        return               
    ################################################################################################################################################
    def load(self, dataset_name, file_name):
        self.dataset_name = dataset_name
        self.file_name = file_name

        # Loading + time shift
        self.reader.load_markers(dataset_name, file_name) 
        start_time = self.reader.markers.time.iloc[0]

        for i,tag in enumerate(self.tags):            
            self.tags[i].load_markers(dataset_name, file_name)
            self.tags[i].load_rssi(dataset_name, file_name) 
            start_time = min( start_time, tag.rssi.time.iloc[0])

        self.reader.shift_time(start_time)    
        for i,tag in enumerate(self.tags): self.tags[i].shift_time(start_time)
        
        return
    ################################################################################################################################################
    def get_rssi_data(self, window_length=11):
        rssi = pd.DataFrame({'time':self.tags[0].rssi.time})
        for i, tag in enumerate(self.tags):
            tag_rssi = sys.tags[i].rssi.rename({'rssi':'rssi_'+str(i)}, axis=1)
            rssi = rssi.merge( tag_rssi, on='time', how='outer', suffixes=('', ''), sort=True )               

        return rssi        
    ################################################################################################################################################
    def get_motion_data(self, window_length=11):   
        # Modify this function for different targeted movements
        motion = pd.DataFrame({'time':self.reader.markers.time})

        N = 10
        ref_center = self.reader.center()
        ref_norm = self.reader.norm()            
        ref_center = np.ones(np.shape(ref_center)) * np.mean(ref_center[:N], axis=0)
        ref_norm = np.ones(np.shape(ref_norm)) * np.mean(ref_norm[:N], axis=0)

        for i, tag in enumerate(self.tags):
            motion['distance_'+str(i)] = np.linalg.norm( ref_center - tag.center(), axis=1)
            motion['misalignment_'+str(i)] = np.arcsin(np.linalg.norm(np.cross(ref_norm, tag.norm()), axis=1)) *180/np.pi
            # motion['misalignment_'+str(i)] = np.arccos(np.abs(np.sum(np.multiply( ref_norm, tag.norm()), axis=1))) * 180/np.pi 

        return motion
    ################################################################################################################################################
    def get_data(self, window_length=11, save=False,  data_status=False):
        rssi = self.get_rssi_data()
        motion = self.get_motion_data()
        
        # RSSI data status
        status = pd.DataFrame({'time':rssi.time})
        for column in rssi.columns:
            if column != 'time': status[column.replace('rssi','status')] = np.isnan(rssi[column].values)

        time = rssi.time
        time = time[ motion.time.iloc[0] < time]
        time = time[ time < motion.time.iloc[-1] ]

        data = rssi.merge( motion, on='time', how='outer', suffixes=('', ''), sort=True )
        data = data.interpolate(method='nearest')        
        data = data.merge( time, on='time', how='inner', suffixes=('', ''), sort=True )

        data = data.rolling(window_length, axis=0).mean().fillna(method='ffill', axis=0).bfill(axis=0)      
        data['time'] = time.values
        
        
        data = data.merge(status, on='time', how='inner', suffixes=('', ''), sort=True)

        if save:
            dataset_file_path = get_dataset_file_path(self.dataset_name, self.file_name)
            create_folder(dataset_file_path)
            data.to_pickle( dataset_file_path )  
            print(self.file_name, 'is saved.')
        return data
####################################################################################################################################################



####################################################################################################################################################
if __name__ == '__main__':

    dataset_name = 'dataset_03'    

    rfid_info = get_rfid_info(dataset_name)    
    sys = SYSTEM(system_info=rfid_info)
    
    for n in range(18):
        file_name = 'record_' + "{0:0=2d}".format(n)
        sys.load(dataset_name, file_name)

        data = sys.get_data(save=True)

        # fig, axs = plt.subplots(4,1)     
        # data.plot(x='time', y='distance_0', ax=axs[0])
        # data.plot(x='time', y='distance_1', ax=axs[0])
        # data.plot(x='time', y='misalignment_0', ax=axs[1])
        # data.plot(x='time', y='misalignment_1', ax=axs[1])
        # data.plot(x='time', y='rssi_0', ax=axs[2])
        # data.plot(x='time', y='rssi_1', ax=axs[2])
        # axs[3].plot(data.status_0)
        # axs[3].plot(data.status_1)

        # axs[0].title.set_text(file_name)
        # plt.show()

####################################################################################################################################################

   

# ####################################################################################################################################################
# if __name__ == '__main__':

#     dataset_name = 'dataset_03'    

#     rfid_info = get_rfid_info(dataset_name)
#     sys = SYSTEM( system_info=rfid_info )

#     for n in range(18):
#         file_name = 'record_' + "{0:0=2d}".format(n)
#         sys.load(dataset_name, file_name) 
#         data = sys.get_data(save=True)   
#         print(file_name)
# # ###################################################################################################################################################


# ####################################################################################################################################################
# if __name__ == '__main__':

#     dataset_name = 'dataset_04'    

#     rfid_info = get_rfid_info(dataset_name)    
#     sys = SYSTEM(system_info=rfid_info)
    
#     for n in range(1):
#         file_name = 'record_' + "{0:0=2d}".format(n)
#         print(sys.tags[0].markers_color)
#         sys.reader.load_markers(dataset_name, file_name)
#         sys.tags[0].load_markers(dataset_name, file_name)
#         sys.tags[1].load_markers(dataset_name, file_name)
#         sys.tags[1].markers.plot(x='time')
        # sys.load(dataset_name, file_name)     
        
        # rssi = sys.get_rssi_data()
        # motion = sys.get_motion_data()
        # data = sys.get_data()

        # fig, axs = plt.subplots(3,1)
            
        # motion.plot(x='time', y='distance_0', ax=axs[0])
        # motion.plot(x='time', y='distance_1', ax=axs[0])
        # data.plot(x='time', y='distance_0', ax=axs[0])
        # data.plot(x='time', y='distance_1', ax=axs[0])

        # motion.plot(x='time', y='misalignment_0', ax=axs[1])
        # motion.plot(x='time', y='misalignment_1', ax=axs[1])
        # data.plot(x='time', y='misalignment_0', ax=axs[1])
        # data.plot(x='time', y='misalignment_1', ax=axs[1])

        # sys.tags[0].rssi.plot(x='time', y='rssi', ax=axs[2])
        # sys.tags[1].rssi.plot(x='time', y='rssi', ax=axs[2])
        # rssi.plot(x='time', y='rssi_0', ax=axs[2])
        # rssi.plot(x='time', y='rssi_1', ax=axs[2])
        # data.plot(x='time', y='rssi_0', ax=axs[2])
        # data.plot (x='time', y='rssi_1', ax=axs[2])
        # plt.show()

        # dt0 = np.diff (sys.tags[0].rssi.time.to_numpy()) 
        # plt.scatter( np.arange(len(dt0)), dt0)
        # plt.scatter(np.arange(len(dt1)), dt1)
        # plt.show()
###################################################################################################################################################

   
    