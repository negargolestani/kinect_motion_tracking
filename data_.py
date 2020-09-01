
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
                node_info = system_info.loc[idx, system_info.columns!='name'].to_dict()                                
                node = NODE( **node_info )
                if system_info.loc[idx, 'name'][:3] == 'tag': self.tags.append(node)
                else: self.reader = node
        return
    ################################################################################################################################################
    def add_reader(self, color):
        self.reader = NODE( color )
        return
    ################################################################################################################################################
    def add_tag(self, color, IDD=None, port=None):
        self.tags.append( NODE(color, IDD=IDD, port=port) )   
        return               
    ################################################################################################################################################
    def get_motion(self, file_name, th=3, window_length=5, sampler_kind='linear'):
      
        # Reader
        reader_markers_df = self.reader.load_markers(self.dataset_name, file_name)
        reader_markers = np.array(reader_markers_df.markers.to_list()).reshape( [ len(reader_markers_df), -1, 3])
        reader_norm = np.nanmedian( get_norm( reader_markers[:10]), axis=0)
        reader_center = np.nanmedian( np.nanmean( reader_markers[:10], axis=1), axis=0)

        # Tags
        tags_markers_df = [tag.load_markers(self.dataset_name, file_name) for tag in self.tags]
        tags_markers = np.array([np.array(tag_markers_df.markers.to_list()).reshape( [ len(tags_markers_df[0]), -1, 3]) for tag_markers_df in tags_markers_df])
        M = np.concatenate(tags_markers, axis=1) 
        Nt, Nm, _ = np.shape(M)

        time = tags_markers_df[0].time.to_numpy()
        dt = np.tile( np.diff(time).reshape(-1,1), (1 , 3))

        # Norm
        tags_norm =  np.array([ get_norm(M[:,[i,j,k],:]) for (i,j,k) in list(combinations(np.arange(Nm), 3))])
        d_tags_norm = np.zeros_like(tags_norm)
        d_tags_norm = np.array([np.abs(np.diff(n, axis=0))/dt for n in tags_norm])
        # Remove jumps
        x, y, z = np.where(d_tags_norm > th)
        for shift in range(window_length): tags_norm[ x[y<Nt-shift], y[y<Nt-shift]+shift, z[y<Nt-shift] ] = np.nan
        tags_norm = signal.medfilt(tags_norm, [1, window_length,1])
        tags_norm = np.nanmedian(tags_norm, axis=0)
        tags_norm = signal.medfilt(tags_norm, [window_length,1])
        
        finite_idxs = np.where(np.all(np.isfinite(tags_norm), axis=1))[0]
        for n in range(3):
            resampler = interpolate.interp1d(time[finite_idxs], tags_norm[finite_idxs,n], kind=sampler_kind)
            tags_norm[5:-5,n] = np.nan_to_num( resampler(time[5:-5]) )         
        tags_norm /= ( np.reshape(np.linalg.norm(tags_norm, axis=1), (-1,1)) * np.ones((1,3)))


        # Center
        plane_point = np.nanmean(M, axis=1)
        for i in range(tags_markers.shape[0]):
            for j in range(tags_markers.shape[2]): 
                dist = np.sum(np.multiply( tags_norm, tags_markers[i,:,j,:]-plane_point), axis=1).reshape(-1,1) * np.ones([1,3])
                tags_markers[i,:,j,:] -= np.multiply( dist, tags_norm )

        tags_centers = np.array([ np.nanmean(tags_markers[i], axis=1) for i in range(2) ])
        tags_centers = np.array([ signal.medfilt(tags_centers[i], [window_length,1])  for i in range(2) ])

        dc = tags_centers[1]-tags_centers[0]
        d = np.linalg.norm(dc, axis=1)
        D = np.nanmedian(d)
        nc = dc / ( d.reshape([-1,1]) * np.ones((1,3)) )

        tags_centers[0] -= np.multiply( (D-d).reshape([-1,1]) * np.ones([1,3]), nc)
        tags_centers[1] += np.multiply( (D-d).reshape([-1,1]) * np.ones([1,3]), nc)


        # Get Relative Motion: makes reader centered at origin (0,0,0) awith surface normal of (0,0,1)
        tags_centers -= reader_center

        if reader_norm[1] == 0: thetaX = 0
        else: thetaX = atan( reader_norm[1]/reader_norm[2] )    
        thetaY = atan( -reader_norm[0] / sqrt(reader_norm[1]**2 + reader_norm[2]**2) )
        R_rot = get_rotationMatrix(thetaX, thetaY, 0)

        for n in range(2): tags_centers[n] = np.matmul(R_rot, tags_centers[n].transpose()).transpose()
        tags_norm = np.matmul(R_rot, tags_norm.transpose()).transpose()

        return pd.DataFrame({
            'time': time,
            'norm': list(tags_norm),
            'center_1': list(tags_centers[0]),
            'center_2': list(tags_centers[1])
            })   
    ################################################################################################################################################    
    # def get_data(self, file_name,  th=4, window_length=5):
    ################################################################################################################################################    
    def get_data_old(self, file_name=None, save=False, resample_dt=None, sampler_kind='linear'):
        
        if file_name is None: 
            file_name_list = [file_path.replace("\\","/").split('/')[-1][:-4] for file_path in glob.glob(get_markers_file_path(self.dataset_name, '')[:-4]+'*.csv')]
        else: 
            file_name_list = [file_name ]
        
        dataset = list()
        for fn in file_name_list:
            start_time, tags_data = 0, list()            
            reader_data = self.reader.get_data(self.dataset_name, fn)                      
            for i, tag in enumerate(self.tags):
                tag_data = tag.get_data(self.dataset_name, fn, ref_node_data=reader_data, resample_dt=resample_dt, sampler_kind=sampler_kind)
                start_time = max(start_time, tag_data.time.iloc[0]) 
                tags_data.append(tag_data)

            for i, tag_data in enumerate(tags_data): 
                tag_data = tag_data.loc[ tag_data.time >= start_time,: ] 
                tag_data.time -= start_time
                dataset.append(tag_data)
                        
                if save: 
                    file_path = get_dataset_folder_path(self.dataset_name) + '/' + fn + '_' + self.tags[i].color + '.csv'
                    create_folder(file_path)
                    tag_data.loc[:,['time','distance','lat_misalignment', 'ang_misalignment']].to_csv(file_path, index=None)            

        return dataset
########################################################################################################################################################
class NODE(object):
    ################################################################################################################################################
    def __init__(self, color, IDD=None, port=None):
        self.color = color
        self.IDD = IDD
        self.port = port
    ################################################################################################################################################
    def load_data_old(self, dataset_name, file_name, resample_dt=None, sampler_kind='linear'):        
        
        data = self.load_markers(dataset_name, file_name)  
        time = data.time    # use kinect time as ref time

        if self.IDD is not None: 
            rssi = self.load_rssi(dataset_name, file_name)
            data = data.merge( rssi, on='time', how='outer', suffixes=('', ''), sort=True)            
            # t_start, t_end = max(t_start, rssi.time.iloc[0]), min(t_end,  rssi.time.iloc[-1])
            time = time.loc[time> rssi.time.iloc[0]].loc[time< rssi.time.iloc[-1]]

        if self.port is not None: 
            vind = self.load_vind(dataset_name, file_name)
            data = data.merge( vind, on='time', how='outer', suffixes=('', ''), sort=True)
            # t_start, t_end = max(t_start, vind.time.iloc[0]), min(t_end,  vind.time.iloc[-1])
            time = time.loc[time> vind.time.iloc[0]].loc[time< vind.time.iloc[-1]]
        
        data.interpolate(method='nearest', axis=0, inplace=True)      
        data = pd.merge( pd.DataFrame(time), data, on='time', how='inner', suffixes=('', ''), sort=True)

        # data.dropna(inplace=True)
        # data.reset_index(drop=True, inplace=True)
        data.reset_index(inplace=True)


        if resample_dt is not None:
            time_new = np.arange(data.time.iloc[0], data.time.iloc[-1], resample_dt)
            resampled_data = pd.DataFrame({'time':time_new})

            for column in data.columns:
                if column =='time':continue
                val_old = np.array(list( data[column].values))                 
                if val_old.ndim > 1:
                    nc = val_old.shape[1]
                    val_new = np.zeros((len(time_new), nc))
                    for n in range(nc):
                        resampler = interpolate.interp1d(time, val_old[:,n], kind=sampler_kind)
                        val_new[:,n] = np.nan_to_num( resampler(time_new) )         
                else:
                    resampler = interpolate.interp1d(time, val_old, kind=sampler_kind)
                    val_new = np.nan_to_num( resampler(time_new) ) 
                resampled_data[column] = list(val_new)                
            return resampled_data

        return data
    ################################################################################################################################################
    def load_markers(self, dataset_name, file_name):   
        markers_file_path = get_markers_file_path(dataset_name, file_name)  
        raw_df  = pd.read_csv(
            markers_file_path,                                                  # relative python path to subdirectory
            usecols = ['time', self.color],                             # Only load the three columns specified.
            parse_dates = ['time'] )         

        # Time
        date_time = pd.to_datetime( raw_df['time'] , format=datime_format)
        time = [ np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time]
        
        # Markers
        markers = [list(map(float, l.replace(']','').replace('[','').replace('\n','').split(", "))) for l in raw_df[self.color].values]  
        markers_npy = np.array(markers).reshape(len(time), -1, 3)
        
        return pd.DataFrame({
            'time': time,
            'markers': list(markers)
            })                
    ################################################################################################################################################
    def load_rssi(self, dataset_name, file_name):
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
        # rssi_df = rssi_df.rolling(window_length, axis=0).median()   # Smoothing
        # rssi_df = rssi_df.ffill(axis=0).bfill(axis=0)               # Gap Filling

        return pd.DataFrame({
            'time':time,
            'rssi':rssi_df.tolist() 
            })
    ################################################################################################################################################
    def load_vind(self, dataset_name, file_name):
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
        # vind_df = vind_df.rolling(window_length, axis=0).median()   # Smoothing
        # vind_df = vind_df.ffill(axis=0).bfill(axis=0)               # Gap Filling

        return pd.DataFrame({
            'time':time,
            'vind':vind_df.tolist() 
            })
################################################################################################################################################


####################################################################################################################################################
def get_norm(markers):
    # Markers: Nt*Nm*3
    # Nm>=3
    norm = np.cross( markers[:,1,:] - markers[:,0,:], markers[:,2,:] - markers[:,0,:])
    idx = norm[:,2]<0
    for i in range(3): norm[idx,i] *= -1
    norm /= ( np.reshape(np.linalg.norm(norm, axis=1), (-1,1)) * np.ones((1,3)))
    return norm
####################################################################################################################################################
def get_rotationMatrix(XrotAngle, YrotAngle, ZrotAngle):
    Rx = np.array([ [1, 0,0], [0, cos(XrotAngle), -sin(XrotAngle)], [0, sin(XrotAngle), cos(XrotAngle)] ])
    Ry = np.array([ [cos(YrotAngle), 0, sin(YrotAngle)], [0, 1, 0], [-sin(YrotAngle), 0, cos(YrotAngle)] ])
    Rz = np.array([ [cos(ZrotAngle), -sin(ZrotAngle), 0], [sin(ZrotAngle), cos(ZrotAngle), 0], [0, 0, 1] ])
    Rtotal =  np.matmul(np.matmul(Rz,Ry),Rx)
    return Rtotal
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
    def generate(self, N, train_data=None, window_length=11, epochs=500):
        if train_data is not None: self.train(train_data, epochs=epochs)

        synth_data = self.decoder.predict(np.random.multivariate_normal( [0]*self.latentdim, np.eye(self.latentdim), N))
        synth_data = signal.savgol_filter( synth_data, window_length=window_length, polyorder=1, axis=1)     

        # synth_data = (synth_data - np.mean(synth_data)) / (np.std(synth_data))
        # synth_data = (synth_data - np.min(synth_data)) / (np.max(synth_data) - np.min(synth_data))
        synth_data = synth_data * self.scale 
        return synth_data
####################################################################################################################################################


################################################################################################################################################
# if __name__ == '__main__':
    # Get data from raw measured data and Save as CSV file to load faster for regression    
    # sys = SYSTEM('arduino_02')
    # data = sys.get_data(save=False, resample_dt=None)
    # print(data[0])

    # generate_synth_motion_data(train_dataset_name_list=['arduino_00','arduino_01','arduino_02'], save_dataset_name='synth_01', N=2000)
################################################################################################################################################
        