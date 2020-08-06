from utils import*



####################################################################################################################################################
class NODE(object):
    ################################################################################################################################################
    def __init__(self, markers_color, IDD=None, port=None):
        self.markers_color = markers_color
        self.IDD = IDD
        self.port = port
    ################################################################################################################################################
    def get_data(self, dataset_name, file_name, ref_node=None, window_length=11, ref_node_data=None):        
        data = self.get_motion(dataset_name, file_name, window_length=window_length, ref_node_data=ref_node_data)  

        if self.IDD is not None: 
            rssi = self.get_rssi(dataset_name, file_name, window_length=window_length)
            data = data.merge( rssi, on='time', how='outer', suffixes=('', ''), sort=True )

        if self.port is not None: 
            vind = self.get_vind(dataset_name, file_name, window_length=window_length)
            data = data.merge( vind, on='time', how='outer', suffixes=('', ''), sort=True )

        return data
    ################################################################################################################################################
    def get_motion(self, dataset_name, file_name, window_length=11, ref_node_data=None):   
        markers_file_path = get_markers_file_path(dataset_name, file_name)  
        raw_df  = pd.read_csv(
            markers_file_path,                                                  # relative python path to subdirectory
            usecols = ['time', self.markers_color],                             # Only load the three columns specified.
            parse_dates = ['time'] ) 
        

        # Time
        date_time = pd.to_datetime( raw_df['time'] , format=datime_format)
        time = [ np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time]
        
        # Markers
        markers = [list(map(float, l.replace(']','').replace('[','').replace('\n','').split(", "))) for l in raw_df[self.markers_color].values]  
        markers_npy = np.array(markers).reshape(len(time), -1, 3)
        # DON'T Smooth markers. markers can be switched in array and smoothing causes error    

        # Center    
        center = np.mean(markers_npy, axis=1)         
        center = np.nan_to_num(center)
        center = signal.savgol_filter( center, window_length=window_length, polyorder=1, axis=0)     

        # Norm
        norm = np.cross( markers_npy[:,1,:] - markers_npy[:,0,:], markers_npy[:,2,:] - markers_npy[:,0,:])
        # norm[ norm[:,2]<0, :] *= -1   # Don't use ! 
        norm = norm / ( np.reshape(np.linalg.norm(norm, axis=1), (-1,1)) * np.ones((1,3)))
        # DOn't smooth norm 

        if ref_node_data is None:            
            return pd.DataFrame({
                'time': time,
                'markers': markers,
                'center': list(center), 
                'norm': list(norm)
                })    

        ############################
        #### Relative Movements ####

        # Reader center/norm     
        N = 10
        ref_center = np.tile( np.mean(ref_node_data.center.loc[:N], axis=0), (len(ref_node_data),1) )
        ref_norm = np.tile( np.mean(ref_node_data.norm.loc[:N], axis=0), (len(ref_node_data),1) )            
        
        distance_vec = ref_center - center                                                                  # Distance (vector)        
        distance = np.linalg.norm( distance_vec, axis=1)                                                    # Distance
        lat_misalignment = np.sqrt(distance**2 - np.sum(np.multiply( distance_vec, ref_norm), axis=1)**2)   # Lateral Misalignment                   
        ang_misalignment = np.arcsin( np.linalg.norm(np.cross(ref_norm, norm), axis=1) )*180/np.pi          # Angular Misalignment

        # Smoothing (smooth these params after all calculation)               
        distance = signal.savgol_filter( distance, window_length=window_length, polyorder=1, axis=0)                        
        lat_misalignment = signal.savgol_filter( lat_misalignment, window_length=window_length, polyorder=1, axis=0)        
        ang_misalignment = signal.savgol_filter( ang_misalignment, window_length=window_length, polyorder=1, axis=0)  

        return pd.DataFrame({
            'time': time,
            'markers': markers,
            'center': list(center), 
            'norm': list(norm),
            'distance': list(distance),
            'lat_misalignment': list(lat_misalignment),
            'ang_misalignment': list(ang_misalignment)
            })                 
    ################################################################################################################################################
    def get_rssi(self, dataset_name, file_name, window_length=11):
        # Load data 
        rfid_file_path = get_rfid_file_path(dataset_name, file_name)       
        raw_df = pd.read_csv(
            rfid_file_path,                                                     # relative python path to subdirectory
            delimiter  = ';',                                                   # Tab-separated value file.
            usecols = ['IDD', 'Time', 'Ant/RSSI'],                              # Only load the three columns specified.
            parse_dates = ['Time'] )                                            # Intepret the birth_date column as a date      

        # self.rssi.loc[ self.rssi.IDD != self.IDD,'rssi'] = np.nan
        raw_df = raw_df.loc[ raw_df['IDD'] == self.IDD, :]

        # Time
        date_time = pd.to_datetime( raw_df['Time'] , format=datime_format)
        time = [ np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time]
        
        # RSSI
        # rssi_df = pd.DataFrame({ 'rssi': raw_df['Ant/RSSI'].str.replace('Ant.No 1 - RSSI: ', '').astype(int) })
        rssi_df = raw_df['Ant/RSSI'].str.replace('Ant.No 1 - RSSI: ', '').astype(int) 
        rssi_df = rssi_df.rolling(window_length, axis=0).median()   # Smoothing
        rssi_df = rssi_df.ffill(axis=0).bfill(axis=0)               # Gap Filling

        return pd.DataFrame({
            'time':time,
            'rssi':rssi_df.tolist() 
            })
    ################################################################################################################################################
    def get_vind(self, dataset_name, file_name, window_length=11):
        # Load data 
        arduino_file_path = get_arduino_file_path(dataset_name, file_name)               
        raw_df = pd.read_csv(arduino_file_path)

        # Time
        date_time = pd.to_datetime( raw_df['time'] , format=datime_format)
        time = [ np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time]

        # V_induced (Vind)
        vind = raw_df[self.port]
        vind = vind.rolling(window_length, axis=0).median()   # Smoothing
        vind = vind.ffill(axis=0).bfill(axis=0)               # Gap Filling
         
        return pd.DataFrame({
            'time':time,
            'vind':vind.tolist()
        })
########################################################################################################################################################################################################################################################################################################
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
    def add_tag(self, markers_color, IDD=None, port=None):
        self.tags.append( NODE(markers_color, IDD=IDD, port=port) )   
        return               
    ################################################################################################################################################
    def get_data(self, dataset_name, file_name, save=False, window_length=11):
        reader_data = self.reader.get_data(dataset_name, file_name)  
        data = pd.DataFrame({'time':reader_data.time})

        for i, tag in enumerate(self.tags):
            tag_data = tag.get_data(dataset_name, file_name, ref_node_data=reader_data)
            tag_data = tag_data.add_suffix('_'+str(i)).rename({'time_'+str(i):'time'}, axis='columns')
            data = data.merge( tag_data, on='time', how='outer', suffixes=('', '' ), sort=True )

    
        # Select specific time samples 
        target_time = data.time[ np.where(np.any( ~np.isnan(data.filter(regex='rssi', axis=1)), axis=1))[0] ]   # time samples that at least one rssi is not Nan
        data = data.interpolate(method='nearest')        
        data.loc[:, data.columns!='time'] = data.loc[:, data.columns!='time'].rolling(window_length, axis=0).mean().fillna(method='ffill', axis=0).bfill(axis=0)      
        data = data.merge( pd.DataFrame({'time':target_time}), on='time', how='inner', suffixes=('', '' ), sort=True )
     
        # Save
        if save:
            dataset_file_path = get_dataset_file_path(dataset_name, file_name)
            create_folder(dataset_file_path)
            data.to_pickle( dataset_file_path)  
            print(file_name, 'is saved.')

        return data
####################################################################################################################################################



####################################################################################################################################################
if __name__ == '__main__':

    dataset_name = 'dataset_01'    
    doPlot = False
    doSave = True

    rfid_info = get_rfid_info(dataset_name)    
    sys = SYSTEM(system_info=rfid_info)

    for n in range(30):
        file_name = 'record_' + "{0:0=2d}".format(n)
        data = sys.get_data(dataset_name, file_name, save=doSave)

        if doPlot:
            fig, axs = plt.subplots(4,1)  
            axs[0].title.set_text(file_name)

            data.plot(x='time', y='distance_0', ax=axs[0])
            data.plot(x='time', y='distance_1', ax=axs[0])
            axs[0].set_ylim([0, 1])

            data.plot(x='time', y='lat_misalignment_0', ax=axs[1])
            data.plot(x='time', y='lat_misalignment_1', ax=axs[1])
            axs[1].set_ylim([0, .6])

            data.plot(x='time', y='ang_misalignment_0', ax=axs[2])
            data.plot(x='time', y='ang_misalignment_1', ax=axs[2])
            axs[2].set_ylim([0, 90])

            data.plot(x='time', y='rssi_0', ax=axs[3])
            data.plot(x='time', y='rssi_1', ax=axs[3])
            axs[3].set_ylim([50, 150])


            plt.show()
####################################################################################################################################################

   