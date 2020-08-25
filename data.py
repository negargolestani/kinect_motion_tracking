
from utils import*


####################################################################################################################################################
def generate_synth_motion_data(train_dataset_name_list, save_dataset_name=None, N=1000, resample_dt=.1):
    
    train_dataset_list_df = list()
    for train_dataset_name in train_dataset_name_list:
        sys = SYSTEM(train_dataset_name)
        train_dataset_list_df_n = sys.get_data(resample_dt=resample_dt)
        train_dataset_list_df = [*train_dataset_list_df, *train_dataset_list_df_n]
    Nt = min( [len(data) for data in train_dataset_list_df])        
    synthesizer = SYNTHESIZER( Nt )

    time = np.array([ data.time.values[:Nt] for data in train_dataset_list_df])
    distance = np.array([ data.distance.values[:Nt] for data in train_dataset_list_df])
    lat_misalignment = np.array([ data.lat_misalignment.values[:Nt] for data in train_dataset_list_df])
    ang_misalignment = np.array([ data.ang_misalignment.values[:Nt] for data in train_dataset_list_df])
    height = np.sqrt( np.subtract(distance**2, lat_misalignment**2))

    height_synth = synthesizer.generate(N, train_data=height)
    lat_misalignment_synth = synthesizer.generate(N, train_data=lat_misalignment)
    ang_misalignment_synth = synthesizer.generate(N, train_data=ang_misalignment)
    distance_synth = np.sqrt( np.add(height_synth**2, lat_misalignment_synth**2))

    synth_data = dict(
        time = time,        
        distance = distance_synth,
        lat_misalignment = lat_misalignment_synth,
        ang_misalignment = ang_misalignment_synth)
    
    # Save
    if save_dataset_name is not None:
        folder_path = get_dataset_folder_path(save_dataset_name) 
        create_folder(folder_path + '/tmp.csv')
        for key,value in synth_data.items(): np.savetxt(folder_path+'/'+key +'.csv', value) 
        
    return synth_data
####################################################################################################################################################



####################################################################################################################################################
class NODE(object):
    ################################################################################################################################################
    def __init__(self, markers_color, IDD=None, port=None):
        self.markers_color = markers_color
        self.IDD = IDD
        self.port = port
    ################################################################################################################################################
    def get_data(self, dataset_name, file_name, ref_node_data=None, window_length=11, resample_dt=None):        
        
        data = self.get_motion(dataset_name, file_name, window_length=window_length, ref_node_data=ref_node_data)  
        time = data.time    # use kinect time as ref time

        if self.IDD is not None: 
            rssi = self.get_rssi(dataset_name, file_name, window_length=window_length)
            data = data.merge( rssi, on='time', how='outer', suffixes=('', ''), sort=True)            
            # t_start, t_end = max(t_start, rssi.time.iloc[0]), min(t_end,  rssi.time.iloc[-1])
            time = time.loc[time> rssi.time.iloc[0]].loc[time< rssi.time.iloc[-1]]

        if self.port is not None: 
            vind = self.get_vind(dataset_name, file_name, window_length=window_length)
            data = data.merge( vind, on='time', how='outer', suffixes=('', ''), sort=True)
            # t_start, t_end = max(t_start, vind.time.iloc[0]), min(t_end,  vind.time.iloc[-1])
            time = time.loc[time> vind.time.iloc[0]].loc[time< vind.time.iloc[-1]]
        
        data.interpolate(method='nearest', axis=0, inplace=True)      
        data = pd.merge( pd.DataFrame(time), data, on='time', how='inner', suffixes=('', ''), sort=True)


        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)

        if resample_dt is not None:
            resampled_data = pd.DataFrame({'time':np.arange(data.time.iloc[0], data.time.iloc[-1], resample_dt)})
            for column in data.columns:
                if column =='time':continue
                resampler = interpolate.interp1d(data.time.values, data[column].values, kind='linear')
                resampled_data[column] = np.nan_to_num( resampler(resampled_data.time.values) )
                # resampled_data[column] = signal.savgol_filter( resampled_data[column], window_length=window_length, polyorder=1, axis=0)             
            data = resampled_data      

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
            
        N = 10
        ref_center = np.mean(ref_node_data.center.loc[:N], axis=0)
        ref_norm = np.mean(ref_node_data.norm.loc[:N], axis=0)           
                
        distance, lat_misalignment, ang_misalignment = list(), list(), list()
        for i in range(len(center)):
            coilsDistance, xRotAngle = calculate_params(ref_center, ref_norm, center[i,:], norm[i,:])
            distance.append(np.linalg.norm(coilsDistance))
            lat_misalignment.append(np.linalg.norm(coilsDistance[:2]))
            ang_misalignment.append(xRotAngle)

        # # Smoothing (smooth these params after all calculation)               
        distance = signal.savgol_filter( distance, window_length=window_length, polyorder=1, axis=0)                        
        lat_misalignment = signal.savgol_filter( lat_misalignment, window_length=window_length, polyorder=1, axis=0)        
        ang_misalignment = signal.savgol_filter( ang_misalignment, window_length=window_length, polyorder=1, axis=0)  
       
        return pd.DataFrame({
            'time': time,
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
        raw_df = raw_df.loc[ raw_df['IDD'] == self.IDD, :]

        # Time
        date_time = pd.to_datetime( raw_df['Time'] , format=datime_format)
        time = [ np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time]
        
        # RSSI
        # rssi_df = pd.DataFrame({ 'rssi': raw_df['Ant/RSSI'].str.replace('Ant.No 1 - RSSI: ', '').astype(int) })
        rssi_df = raw_df['Ant/RSSI'].str.replace('Ant.No 1 - RSSI: ', '').astype(float) 
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
        raw_df = raw_df.loc[ raw_df['port'] == self.port, :]

        # Time
        date_time = pd.to_datetime( raw_df['time'] , format=datime_format)
        time = [ np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time]
        
        # RSSI
        # rssi_df = pd.DataFrame({ 'rssi': raw_df['Ant/RSSI'].str.replace('Ant.No 1 - RSSI: ', '').astype(int) })
        vind_df = raw_df['vind'].astype(float) 
        vind_df = vind_df.rolling(window_length, axis=0).median()   # Smoothing
        vind_df = vind_df.ffill(axis=0).bfill(axis=0)               # Gap Filling

        return pd.DataFrame({
            'time':time,
            'vind':vind_df.tolist() 
            })
####################################################################################################################################################
class SYSTEM(object):
    ################################################################################################################################################
    def __init__(self, dataset_name=None):
        self.reader = None
        self.tags = list()
        self.dataset_name = None
        
        if dataset_name is not None:
            self.dataset_name = dataset_name
            system_info = get_sys_info(dataset_name)    
            for idx in system_info.index:                
                node_info = system_info.loc[idx, system_info.columns!='type'].to_dict()                
                for key,value in node_info.items():
                    if value == 'None': node_info.update({key: None})
                
                node = NODE( **node_info )
                if system_info.loc[idx, 'type'] == 'tag': self.tags.append(node)
                else: self.reader = node
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
    def get_data(self, file_name=None, save=False, resample_dt=None):
        if file_name is None: file_name_list = [file_path.replace("\\","/").split('/')[-1][:-4] for file_path in glob.glob(get_markers_file_path(self.dataset_name, '')[:-4]+'*.csv')]
        else: file_name_list = [file_name ]
        
        dataset = list()
        for fn in file_name_list:
            start_time, data = np.inf, list()
            
            reader_data = self.reader.get_data(self.dataset_name, fn)                      
            for i, tag in enumerate(self.tags):
                tag_data = tag.get_data(self.dataset_name, fn, ref_node_data=reader_data, resample_dt=resample_dt)
                start_time = min(start_time, tag_data.time.iloc[0]) 
                data.append(tag_data)

            for i, tag_data in enumerate(data): 
                tag_data.time -= start_time
                dataset.append(tag_data)
                        
                if save: 
                    file_path = get_dataset_folder_path(self.dataset_name) + '/' + fn + '_' + self.tags[i].markers_color + '.csv'
                    create_folder(file_path)
                    tag_data.to_csv(file_path, index=None)            

        return dataset
####################################################################################################################################################



####################################################################################################################################################
class KLDivergenceLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)
    ################################################################################################################################################
    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var - K.square(mu) -  K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs
####################################################################################################################################################
class SYNTHESIZER(object):
    ################################################################################################################################################
    def __init__(self, Nt, hiddendim=300, latentdim=100):
        self.hiddendim = hiddendim
        self.latentdim = latentdim
        self.Nt = Nt
        return
    ################################################################################################################################################
    def build(self):
        epsilon_std = 1.0

        x = Input(shape=(self.Nt,))
        h = Dense(self.hiddendim, activation='relu')(x)
        z_mu = Dense(self.latentdim)(h)
        z_log_var = Dense(self.latentdim)(h)
        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
        eps = Input(tensor=K.random_normal(stddev=epsilon_std, shape=(K.shape(x)[0], self.latentdim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])

        self.encoder = Model(x, z_mu)        
        self.decoder = Sequential([
            Dense(self.hiddendim, input_dim=self.latentdim, activation='relu'),
            Dense(self.Nt, activation='sigmoid') ])
        x_pred = self.decoder(z)

        self.vae = Model(inputs=[x, eps], outputs=x_pred)

        def nll(y_true, y_pred): return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
        self.vae.compile(optimizer='rmsprop', loss=nll)
        
        return
    ################################################################################################################################################
    def train(self, train_data, epochs=500):
        self.build()
        self.scale = np.max(train_data)
        train_data_normalized = (train_data - np.min(train_data)) / (np.max(train_data) - np.min(train_data))
        self.vae.fit(train_data_normalized, train_data_normalized, epochs=epochs, validation_data=(train_data_normalized, train_data_normalized), verbose=0)
        return    
    ################################################################################################################################################
    def generate(self, N, train_data=None, window_length=11):
        if train_data is not None: self.train(train_data)

        synth_data = self.decoder.predict(np.random.multivariate_normal( [0]*self.latentdim, np.eye(self.latentdim), N))
        synth_data = signal.savgol_filter( synth_data, window_length=window_length, polyorder=1, axis=1)     

        # synth_data = (synth_data - np.mean(synth_data)) / (np.std(synth_data))
        # synth_data = (synth_data - np.min(synth_data)) / (np.max(synth_data) - np.min(synth_data))
        synth_data = synth_data * self.scale 
        return synth_data
####################################################################################################################################################
####################################################################################################################################################
def calculate_params(loc_1_, align_1_, loc_2_, align_2_):
    loc_1, loc_2, align_1, align_2 = np.array(loc_1_), np.array(loc_2_), np.array(align_1_), np.array(align_2_)
    
    if align_1[1] == 0: thetaX = 0
    else: thetaX = atan( align_1[1]/align_1[2] )    

    thetaY = atan( -align_1[0] / sqrt(align_1[1]**2 + align_1[2]**2) )
    align_2_new = np.matmul(get_rotationMatrix(thetaX, thetaY, 0), np.reshape(align_2,[3,1]))    
        
    if align_2_new[0] == 0: thetaZ = 0
    else: thetaZ = atan(align_2_new[0]/align_2_new[1])    
    Rtot = get_rotationMatrix(thetaX, thetaY , thetaZ)
    loc_2_new = np.matmul(Rtot, np.reshape(loc_2-loc_1, [3,1]))    
    align_2_new = np.matmul(Rtot, np.reshape(align_2, [3,1]))    

    coilsDistance = abs(np.round(np.reshape(loc_2_new, [1,3])[0], 10))
    xRotAngle = np.round( atan(abs( align_2_new[1]/align_2_new[2] )) * 180/pi )
    # xRotAngle = np.round( atan(-align_2_new[1]/align_2_new[2]) * 180/pi )

    return coilsDistance, xRotAngle
####################################################################################################################################################
def get_rotationMatrix(XrotAngle, YrotAngle, ZrotAngle):
    Rx = np.array([ [1, 0,0], [0, cos(XrotAngle), -sin(XrotAngle)], [0, sin(XrotAngle), cos(XrotAngle)] ])
    Ry = np.array([ [cos(YrotAngle), 0, sin(YrotAngle)], [0, 1, 0], [-sin(YrotAngle), 0, cos(YrotAngle)] ])
    Rz = np.array([ [cos(ZrotAngle), -sin(ZrotAngle), 0], [sin(ZrotAngle), cos(ZrotAngle), 0], [0, 0, 1] ])
    Rtotal =  np.matmul(np.matmul(Rz,Ry),Rx)
    return Rtotal
####################################################################################################################################################


################################################################################################################################################
if __name__ == '__main__':
    # Get data from raw measured data and Save as CSV file to load faster for regression    
    sys = SYSTEM('arduino_00')
    # data = sys.get_data(save=False, resample_dt=None)
    # print(data[0])

    # generate_synth_motion_data(train_dataset_name_list=['arduino_00','arduino_01','arduino_02'], save_dataset_name='synth_01', N=2000)

    for n in range(9):
        data = sys.get_data(file_name='record_1'+str(n), save=False, resample_dt=.1)
        c1 = data[0]
        c2 = data[1]

        # feat = 'distance'
        # ax = c1.plot(x='time', y=feat)
        # c2.plot(x='time', y=feat, ax=ax)

        d = abs(c1.distance - c2.distance)
        ld = abs(c1.lat_misalignment-c2.lat_misalignment)

        plt.plot(d)
        plt.plot(ld)

        plt.show()


################################################################################################################################################
        