
from utils import*


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
        tags_markers_df = [tag.load_markers(self.dataset_name, file_name) for tag in self.tags if tag.color is not None]
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
            # tags_norm[5:-5,n] = np.nan_to_num( resampler(time[5:-5]) )  
            tags_norm[finite_idxs[0]:finite_idxs[-1],n] = np.nan_to_num( resampler(time[finite_idxs[0]:finite_idxs[-1]]) )         
        tags_norm /= ( np.reshape(np.linalg.norm(tags_norm, axis=1), (-1,1)) * np.ones((1,3)) + eps)

        # Center
        plane_point = np.nanmean(M, axis=1)
        for i in range(tags_markers.shape[0]):
            for j in range(tags_markers.shape[2]): 
                dist = np.sum(np.multiply( tags_norm, tags_markers[i,:,j,:]-plane_point), axis=1).reshape(-1,1) * np.ones([1,3])
                tags_markers[i,:,j,:] -= np.multiply( dist, tags_norm )

        tags_centers = np.array([ np.nanmean(tags_markers[i], axis=1) for i in range(len(tags_markers)) ])
        tags_centers = np.array([ signal.medfilt(tags_centers[i], [window_length,1])  for i in range(len(tags_markers)) ])

        center = np.nanmean(tags_centers,axis=0)
        for i, tag_center in enumerate(tags_centers): 
            dc = tag_center - center
            d = np.linalg.norm(dc, axis=1)
            D = np.nanmedian(d)
            nc = dc / ( d.reshape([-1,1]) * np.ones((1,3)) + eps)
            tags_centers[i] = center + D*nc

        tags_centers -= reader_center

        if reader_norm[1] == 0: thetaX = 0
        else: thetaX = atan( reader_norm[1]/reader_norm[2] )    
        thetaY = atan( -reader_norm[0] / sqrt(reader_norm[1]**2 + reader_norm[2]**2 + eps) )
        R_rot = get_rotationMatrix(thetaX, thetaY, 0)

        for n in range(len(tags_markers)): tags_centers[n] = np.matmul(R_rot, tags_centers[n].transpose()).transpose()
        tags_norm = np.matmul(R_rot, tags_norm.transpose()).transpose()

        clean_idx = np.all(np.all(~np.isnan(tags_centers),axis=2), axis=0) * np.all(~np.isnan(tags_norm), axis=1)
        data = dict( time = time[clean_idx], norm = list(tags_norm[clean_idx]) )
        for n in range(len(tags_markers)): data.update({'center_'+str(n+1):list(tags_centers[n,clean_idx])})
        
        return pd.DataFrame(data).dropna()  
    ################################################################################################################################################    
    def get_data(self, file_name, th=3, window_length=5, resample_dt=None, sampler_kind='linear', save=False):
        data = self.get_motion(file_name, th=3, window_length=5)
        for i, tag in enumerate(self.tags):
            tag_data = tag.load_measurements(self.dataset_name, file_name)
            tag_data.columns = [ '{}{}'.format(c,'' if c in ['time'] else '_'+str(i+1)) for c in tag_data.columns]
            data = pd.merge( data, tag_data, on='time', how='outer', suffixes=('', ''), sort=True)

        data.dropna(inplace=True)

        if resample_dt is not None:
            resampling_time = np.arange(data.time.iloc[0], data.time.iloc[-1], resample_dt)
            resampled_data = pd.DataFrame({'time':resampling_time})

            for column in data.columns:
                if column =='time':continue
                val_ts = data[column].dropna()
                val = np.array(val_ts.to_list())
                time = data.time[val_ts.index].to_numpy()
                time_new = copy.deepcopy(resampling_time)
                time_new = time_new[ (time_new > time[0]) * ( time_new < time[-1]) ]

                if val.ndim > 1:
                    val_new = np.zeros((len(time_new), val.shape[1]))
                    for n in range(val.shape[1]):
                        resampler = interpolate.interp1d(time, val[:,n], kind=sampler_kind)
                        val_new[:,n] = resampler(time_new)         
                        # val_new[:,n] = np.nan_to_num( val_new[:,n]  )          
                else:
                    resampler = interpolate.interp1d(time, val, kind=sampler_kind)
                    val_new = resampler(time_new)         
                    # val_new = np.nan_to_num( val_new ) 

                resampled_column = pd.DataFrame({'time':time_new, column:list(val_new)})
                resampled_data = pd.merge( resampled_data, resampled_column, on='time', how='inner', suffixes=('', ''), sort=True)
            data = resampled_data


       # Processing
        data.interpolate(method='nearest', axis=0, inplace=True)      
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)
        data.time -= data.time.iloc[0]

        if save: 
            file_path = get_dataset_folder_path(self.dataset_name) + '/' + file_name + '.csv'
            create_folder(file_path)
            data.to_csv(file_path, index=False, sep=",") 

        return data
    ################################################################################################################################################    
    def get_dataset(self, as_dict=False, **params):
        file_name_list = [file_path.replace("\\","/").split('/')[-1][:-4] for file_path in glob.glob(get_markers_file_path(self.dataset_name, '')[:-4]+'*.csv')]            
        dataset = [ self.get_data(file_name, **params) for file_name in file_name_list]

        if as_dict:
            dataset_dict = defaultdict(list)
            for data in dataset:
                for key, value in data.to_dict('list').items():
                    dataset_dict[key].append(value)
            return dataset_dict

        return dataset
####################################################################################################################################################
class NODE(object):
    ################################################################################################################################################
    def __init__(self, color, IDD=None, port=None):
        self.color = color
        self.IDD = IDD
        self.port = port
    ################################################################################################################################################
    def load_markers(self, dataset_name, file_name):   
        markers_file_path = get_markers_file_path(dataset_name, file_name)  
        raw_df  = pd.read_csv(
            markers_file_path,                                            # relative python path to subdirectory
            usecols = ['time', self.color],                               # Only load the three columns specified.
            parse_dates = ['time'] )         

        # Time
        date_time = pd.to_datetime( raw_df['time'] , format=datime_format)
        time = np.array([np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time])
        
        # Markers
        markers = np.array([list(map(float, l.replace(']','').replace('[','').replace('\n','').split(", "))) for l in raw_df[self.color].values]) 
        
        # markers = raw_df[self.color].apply(literal_eval).to_list()
        # markers_npy = markers.reshape(len(time), -1, 3)        
        clean_idx = np.where(np.all(~np.isnan(markers), axis=1))[0]
        return pd.DataFrame({
            'time': time[clean_idx],
            'markers': list(markers[clean_idx,:])
            })                
    ################################################################################################################################################
    def load_measurements(self, dataset_name, file_name):        
        
        data = pd.DataFrame({'time':list()})
        if self.IDD is not None:  
            data = pd.merge( data, self.load_rssi(dataset_name, file_name), on='time', how='outer', suffixes=('', ''), sort=True)
        if self.port is not None: 
            data = pd.merge( data, self.load_vind(dataset_name, file_name), on='time', how='outer', suffixes=('', ''), sort=True)

        # data.interpolate(method='nearest', axis=0, inplace=True)      
        # data.dropna(inplace=True)
        # data.reset_index(drop=True, inplace=True)
        # data.reset_index(inplace=True)
        return data.dropna()
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

        date_time = pd.to_datetime( raw_df['Time'] , format=datime_format)
        time = [ np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time]
        rssi_df = raw_df['Ant/RSSI'].str.replace('Ant.No 1 - RSSI: ', '').astype(float) 

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

        date_time = pd.to_datetime( raw_df['time'] , format=datime_format)
        time = [ np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time]        
        vind_df = raw_df['vind'].astype(float) 
        
        return pd.DataFrame({
            'time':time,
            'meas_vind':vind_df.tolist() 
            })
####################################################################################################################################################
####################################################################################################################################################
def get_norm(markers):
    # Markers: Nt*Nm*3
    # Nm>=3
    norm = np.cross( markers[:,1,:] - markers[:,0,:], markers[:,2,:] - markers[:,0,:])
    idx = norm[:,2]<0
    for i in range(3): norm[idx,i] *= -1
    norm /= ( np.reshape(np.linalg.norm(norm, axis=1), (-1,1)) * np.ones((1,3)) + eps)
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
    def __init__(self):
        pass
    ################################################################################################################################################
    def build(self, Nt, hiddendim=300, latentdim=300):
        epsilon_std = 1.0

        x = Input(shape=(Nt,))
        h = Dense(hiddendim, activation='tanh')(x)
        z_mu = Dense(latentdim)(h)
        z_log_var = Dense(latentdim)(h)
        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
        eps = Input(tensor=K.random_normal(stddev=epsilon_std, shape=(K.shape(x)[0], latentdim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])

        self.encoder = Model(x, z_mu)        
        self.decoder = Sequential([
            Dense(hiddendim, input_dim=latentdim, activation='tanh'),
            Dense(Nt, activation='sigmoid') ])
        x_pred = self.decoder(z)

        self.vae = Model(inputs=[x, eps], outputs=x_pred)

        def nll(y_true, y_pred): return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
        self.vae.compile(optimizer='rmsprop', loss=nll)
        
        return 
    ################################################################################################################################################
    def synthesize(self, train_data_, N=1000, window_length=15, epochs=500, hiddendim=300, latentdim=300):                
        
        train_data = np.array(train_data_)        
        if train_data.ndim == 2: train_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], 1])
        _, Nt, Nc = np.shape(train_data)
        synth_data = np.zeros((N,Nt,Nc))

        for c in range(Nc):
            self.build(Nt, hiddendim=hiddendim, latentdim=latentdim)

            train_data_c = train_data[:,:,c]
            MIN, MAX = np.nanmin(train_data), np.nanmax(train_data)     
            train_data_c = (train_data_c - MIN) / (MAX-MIN)
            self.vae.fit(train_data_c, train_data_c, epochs=epochs, validation_data=(train_data_c, train_data_c), verbose=0)

            synth_data_c = self.decoder.predict(np.random.multivariate_normal( [0]*latentdim, np.eye(latentdim), N))
            synth_data_c = signal.savgol_filter( synth_data_c, window_length=window_length, polyorder=1, axis=1) 
            synth_data_c = signal.savgol_filter( synth_data_c, window_length=window_length, polyorder=1, axis=1) 
            synth_data_c = (synth_data_c - np.nanmin(synth_data_c)) / (np.nanmax(synth_data_c) - np.nanmin(synth_data_c) )
            synth_data[:,:,c] = synth_data_c * (MAX-MIN) + MIN
        
        return np.array(synth_data)
    ################################################################################################################################################
    def generate(self, train_data_, cond=False, **params):
        train_data = np.array(train_data_)
        if train_data.ndim == 2: train_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], 1])

        if cond:
            d = np.linalg.norm(train_data, axis=2)
            n = train_data / (np.reshape( d, [train_data.shape[0], train_data.shape[1], 1]) * np.ones([1,3]) + eps)
            D = np.nanmedian( np.nanmean(d, axis=1)) 

            theta = np.arcsin( n[:,:,2] )
            phi = np.arccos( n[:,:,0] / (np.linalg.norm(n[:,:,:2], axis=2) + eps)) 
            phi[ n[:,:,1]<0 ] = 2*pi - phi[ n[:,:,1]<0 ]

            phi_synth = self.synthesize(phi,  **params)
            theta_synth = self.synthesize(theta, **params)
            phi_synth, theta_synth = phi_synth[:,:,0], theta_synth[:,:,0]

            data_synth = np.zeros([phi_synth.shape[0], phi_synth.shape[1], 3])
            data_synth[:,:,0] = np.multiply(np.cos(theta_synth), np.cos(phi_synth))
            data_synth[:,:,1] = np.multiply(np.cos(theta_synth), np.sin(phi_synth))
            data_synth[:,:,2] = np.sin(theta_synth)    
            return data_synth * D
        
        return self.synthesize(train_data, **params)
####################################################################################################################################################
####################################################################################################################################################   
def generate_synth_motion_data_(train_dataset_name_list, save_dataset_name=None, resample_dt=.1, Ncoils=1, **params):
    train_dataset = defaultdict(list)
    for train_dataset_name in train_dataset_name_list:
        sys = SYSTEM(train_dataset_name)
        dataset = sys.get_dataset(resample_dt=resample_dt, as_dict=True)
        for key,value in dataset.items(): train_dataset[key].extend(value)
    # Convert all inputs from list of list into into dixed length matrix 
    Nt = min( [len(time) for time in train_dataset['time']])    
    for key, value in train_dataset.items(): train_dataset.update({ key: np.array([v[:Nt] for v in train_dataset[key]]) }) 

    norm = train_dataset['norm']
    centers = np.array([ value for key,value in train_dataset.items() if key[:6] == 'center'])
    center = np.nanmean(centers, axis=0)
    d_centers = np.array([center_n-center for center_n in centers])
    d = np.nanmedian(np.linalg.norm(d_centers, axis=-1)) 
    # phi = np.mod(np.arctan2(d_centers[0,:,:,1], d_centers[0,:,:,0]), 2*pi) 
    phi = np.arctan2(d_centers[0,:,:,1], d_centers[0,:,:,0])

    synthesizer = SYNTHESIZER()
    norm_synth = synthesizer.generate( norm, cond=True, **params)
    phi_synth = synthesizer.generate( phi, **params)
    phi_synth = phi_synth[:,:,0]
    center_synth = synthesizer.generate( center, **params)

    centers_synth = np.zeros((Ncoils, center_synth.shape[0], center_synth.shape[1], center_synth.shape[2]))
    for i in range(centers_synth.shape[1]):
        for j in range(centers_synth.shape[2]):
            R_rot = get_rotationMatrix( 
                np.arctan(norm_synth[i,j,1]/norm_synth[i,j,2] ), 
                -np.arctan(norm_synth[i,j,0]/np.sqrt(norm_synth[i,j,1]**2+norm_synth[i,j,2]**2)), 0).transpose()
            for n in range(centers_synth.shape[0]): 
                phi_n = phi_synth[i,j] + n * 2*pi/Ncoils
                c_n = np.array([ d*np.cos(phi_n), d*np.sin(phi_n), 0 ])             
                centers_synth[n,i,j,:] = np.matmul(R_rot, c_n.reshape(3,1)).reshape(1,3)[0] + center_synth[i,j]

    if save_dataset_name is not None:
        folder_path = get_dataset_folder_path(save_dataset_name) 
        create_folder(folder_path + '/tst.csv')
        time = list(np.arange(Nt)*resample_dt)
        
        for m in range(norm_synth.shape[0]):
            file_path = folder_path + '/record_' + "{0:0=4d}".format(m) + '.csv'
            data =  pd.DataFrame({ 'time': time, 'norm': list(norm_synth[m]) }) 
            for n in range(Ncoils): data['center_'+str(n+1)] = list(centers_synth[n, m]) 
            data.to_csv(file_path, index=None) 
    else:
        return dict(
            time = list(np.arange(Nt)*resample_dt),
            norm = norm_synth,
            centers = centers_synth
        )
####################################################################################################################################################   
def generate_synth_motion_data(train_dataset_name_list, save_dataset_name=None, resample_dt=.1, Ncoils=1, **params):
    train_dataset = defaultdict(list)
    for train_dataset_name in train_dataset_name_list:
        sys = SYSTEM(train_dataset_name)
        dataset = sys.get_dataset(resample_dt=resample_dt, as_dict=True)
        for key,value in dataset.items(): train_dataset[key].extend(value)
    # Convert all inputs from list of list into into dixed length matrix 
    Nt = min( [len(time) for time in train_dataset['time']])    
    for key, value in train_dataset.items(): train_dataset.update({ key: np.array([v[:Nt] for v in train_dataset[key]]) }) 

    norm = train_dataset['norm']
    centers = np.array([ value for key,value in train_dataset.items() if key[:6] == 'center'])
    center = np.nanmean(centers, axis=0)
    
    synthesizer = SYNTHESIZER()
    norm_synth = synthesizer.generate( norm, cond=True, **params)
    center_synth = synthesizer.generate( center, **params)
    centers_synth = np.zeros((Ncoils, center_synth.shape[0], center_synth.shape[1], center_synth.shape[2]))

    if Ncoils==1:
        centers_synth[0,:,:,:] = center_synth

    else:
        d_centers = np.array([center_n-center for center_n in centers])
        d = np.nanmedian(np.linalg.norm(d_centers, axis=-1)) 
        phi = np.arctan2(d_centers[0,:,:,1], d_centers[0,:,:,0])

        phi_synth = synthesizer.generate( phi, **params)
        phi_synth = phi_synth[:,:,0]

        for i in range(centers_synth.shape[1]):
            for j in range(centers_synth.shape[2]):
                R_rot = get_rotationMatrix( 
                    np.arctan(norm_synth[i,j,1]/norm_synth[i,j,2] ), 
                    -np.arctan(norm_synth[i,j,0]/np.sqrt(norm_synth[i,j,1]**2+norm_synth[i,j,2]**2)), 0).transpose()
                for n in range(centers_synth.shape[0]): 
                    phi_n = phi_synth[i,j] + n * 2*pi/Ncoils
                    c_n = np.array([ d*np.cos(phi_n), d*np.sin(phi_n), 0 ])             
                    centers_synth[n,i,j,:] = np.matmul(R_rot, c_n.reshape(3,1)).reshape(1,3)[0] + center_synth[i,j]

    if save_dataset_name is not None:
        folder_path = get_dataset_folder_path(save_dataset_name) 
        create_folder(folder_path + '/tst.csv')
        time = list(np.arange(Nt)*resample_dt)
        
        for m in range(norm_synth.shape[0]):
            file_path = folder_path + '/record_' + "{0:0=4d}".format(m) + '.csv'
            data =  pd.DataFrame({ 'time': time, 'norm': list(norm_synth[m]) }) 
            for n in range(Ncoils): data['center_'+str(n+1)] = list(centers_synth[n, m]) 
            data.to_csv(file_path, index=None) 
    else:
        return dict(
            time = list(np.arange(Nt)*resample_dt),
            norm = norm_synth,
            centers = centers_synth
        )
####################################################################################################################################################   


####################################################################################################################################################
if __name__ == '__main__':
#     # Get data from raw measured data and Save as CSV file to load faster for regression    
    # sys = SYSTEM('arduino_parallel')
    # for n in range(20): data = sys.get_data('record_' + "{0:0=2d}".format(n), save=True)
    synth_dataset = generate_synth_motion_data( ['arduino_parallel'], save_dataset_name='synth_2coils_parallel', Ncoils=1, N=2000, resample_dt=.1, window_length=15, epochs=500, hiddendim=300, latentdim=300)
####################################################################################################################################################
