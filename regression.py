
from utils import*

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from sklearn.metrics import mean_squared_error
from keras import backend as K
from pycaret.regression import*

eps = 1e-12
checkpoints_folderName = 'Checkpoints'
checkpoints_modelName = 'model.ckpt'

# from sklearn import*
# from sklearn.svm import SVR
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error



####################################################################################################################################################
def get_measured_dataset(dataset_name, resample_dt=.1, as_dict=False):
    sys_info = get_sys_info(dataset_name)    
    sys = SYSTEM(system_info=sys_info)

    dataset = list()
    for file_name in glob.glob(get_time_file_path(dataset_name, '')[:-4]+'*.csv'):    
        file_name = file_name.replace("\\","/").split('/')[-1][:-4]
        reader_data = sys.reader.get_data(dataset_name, file_name)  
        for i, tag in enumerate(sys.tags):
            tag_data = tag.get_data(dataset_name, file_name, ref_node_data=reader_data, resample_dt=resample_dt)
            tag_data.time -= tag_data.time.iloc[0]
            dataset.append(tag_data)   

    if as_dict:
        data_dict = dict()
        for column in dataset[0].columns: data_dict.update({ column : np.array([ data[column].values for data in dataset]) })
        return data_dict
    
    return dataset
####################################################################################################################################################
def get_synth_dataset(dataset_name, as_dict=False):
    synth_dataset_folder_path = get_synth_dataset_folder_path(dataset_name)

    dataset_dict = dict()
    for file_path in glob.glob(synth_dataset_folder_path + '/*.csv'):
        key = file_path.replace('\\','/').split('/')[-1][:-4]
        if key == 'vind': value = np.loadtxt(file_path, delimiter=',')
        else: value = np.loadtxt(file_path)
        dataset_dict.update({key:value})
    if as_dict: return dataset_dict

    dataset = list()
    for n in range(len(dataset_dict['time'])):
        data_df = pd.DataFrame()
        for key, value in dataset_dict.items(): data_df[key] = value[n]
        dataset.append(data_df)    
    return dataset
####################################################################################################################################################
def generate_synth_motion_dataset(train_dataset_name, N=1000, save_dataset_name=None, resample_dt=.1):
    
    train_dataset_list_df = get_measured_dataset(train_dataset_name, resample_dt=resample_dt)
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
        folder_path = '../synthetic_dataset/' + save_dataset_name + '/'
        create_folder(folder_path + '.csv')
        for key,value in synth_data.items(): np.savetxt(folder_path+'/'+key +'.csv', value) 
        
    return synth_data
####################################################################################################################################################



####################################################################################################################################################
class REGRESSOR(object):
    ################################################################################################################################################
    def __init__(self, win_size, Nfeatures, step=1, **params ):
        np.random.seed(7)        
        self.win_size = win_size
        self.Nfeatures = Nfeatures
        self.step = step                
        self.build_model(**params)
    ################################################################################################################################################
    def train(self, train_data, epochs=30, verbose=0, show=True):        
        loss = list()        
        for n_epoch in range(epochs):
            display(n_epoch)
            
            ep_loss = list()            
            for i in range(0, np.shape(train_data.X)[1]-self.win_size, self.step):
                history = self.model.fit( 
                    train_data.X[:, i:i+self.win_size,:], 
                    train_data.Y[:, i+self.win_size-1], 
                    epochs = 1, 
                    verbose = verbose
                )    
                ep_loss.append(history.history['loss'])
            
            loss.append(ep_loss)
            clear_output(wait=True)
            
            if show:
                plt.plot( np.ndarray.flatten( np.array(loss)) )
                plt.show()
                
        return np.array(loss)
    ################################################################################################################################################    
    def predict(self, X):
        predictions = list()
        for i in range(0, np.shape(X)[1]-self.win_size, self.step):
            pred = self.model.predict( X[:, i:i+self.win_size,:])
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return predictions.reshape((-1, np.shape(predictions)[1])).transpose()  
####################################################################################################################################################
class RNN(REGRESSOR):
    def build_model(self, Nunits=3, activation='linear'):        
        self.model = Sequential()           
        self.model.add( LSTM(units=Nunits, activation=activation, return_sequences=True, input_shape=(self.win_size, self.Nfeatures)) )        
        self.model.add( LSTM(units=Nunits, activation=activation, return_sequences=True) )        
        self.model.add( LSTM(units=Nunits, activation=activation))   
        # self.model.add(Dropout(0.1))
        self.model.add( Dense(1))          
            
        self.model.compile(loss='mse', optimizer='adam')         
        print(self.model.summary())        
        return 
####################################################################################################################################################
class TCN(REGRESSOR):
    def build_model(self, Nlayers=2, nb_filters=5, Nfeatures=2, activation=None, optimizer='adam', loss='mse' ):  
        i = tf.keras.Input(batch_shape=(None, self.win_size, Nfeatures)) 
        if Nlayers>1:
            o = tcn.TCN(nb_filters=nb_filters, return_sequences=True)(i)
            for n in range(Nlayers-2): tcn.TCN(nb_filters=nb_filters, return_sequences=True)(o)    
            o = tcn.TCN(nb_filters=nb_filters, return_sequences=False)(o)
        else:
            o = tcn.TCN(nb_filters=nb_filters, return_sequences=False)(i)     
            
        o = tf.keras.layers.Dense(1, activation=activation)(o)
        self.model = tf.keras.Model(inputs=[i], outputs=[o])
        
        self.model.compile(loss='mse', optimizer='adam')        
        print(self.model.summary())                
        return 
####################################################################################################################################################


####################################################################################################################################################
class DATA(object):    
    ######################################################################################################
    def __init__(self, X, Y):            
        self.X = np.array(X) 
        self.Y = np.array(Y)
        return
    ######################################################################################################        
    def segment(self, win_size, stride=None, as_df=False):
        X_segmented, Y_segmented = list(), list()
        if stride is None: stride = win_size
        
        for t in range(0, self.X.shape[1] - win_size, stride): 
            X_segmented = [*X_segmented, *self.X[:,t:t+win_size]]
            Y_segmented = [*Y_segmented, *self.Y[:,t+win_size]]   

        if as_df:
            data_df = pd.DataFrame( np.concatenate([X_segmented, np.reshape(Y_segmented,(-1,1))], axis=1) )
            data_df.columns = [*['feature_'+str(i) for i in range(win_size)], 'target']
            return data_df

        return DATA(X_segmented, Y_segmented)                
    ######################################################################################################
    def merge(self, new_dataset):
        merged_dataset = copy.deepcopy(self)
        merged_dataset.X = np.array([*self.X, *new_dataset.X])
        merged_dataset.Y = np.array([*self.Y, *new_dataset.Y])
        return merged_dataset
    ######################################################################################################
    def select(self, idx_list):
        selected_dataset = copy.deepcopy(self)
        selected_dataset.X = self.X[idx_list]
        selected_dataset.Y = self.Y[idx_list]
        return selected_dataset
    ######################################################################################################
    def split(self, ratio):
        N = len(self.X)
        idxs = np.arange(N)
        random.shuffle(idxs)
        
        Ntrain = int(N*ratio)
        data_p1 = self.select(idxs[:Ntrain])
        data_p2 = self.select(idxs[Ntrain:])
        
        return data_p1, data_p2
    ######################################################################################################
    def mtx( self, Nt_mtx='max' ):  
        # This function padds or cuts all input data (X) to make them same length and generate matrix data(X_mtx)
        # it also nomalize data X-mean(X)
        data_mtx = copy.deepcopy(self)
        if len(np.shape(data_mtx.X))>1:  return data_mtx    

        Nt_list = [np.shape(x)[0] for x in self.X]
        Nt = int( eval('np.' + Nt_mtx)(Nt_list) )
        Nd, Nf = len(self.X),  np.shape(self.X[0])[1]
        
        data_mtx.X = np.zeros( (Nd,Nt,Nf) )
        data_mtx.Y = np.zeros( (Nd,Nt) )
        
        for idx, x in enumerate(self.X): 
            nt = np.shape(x)[0]
            
            if Nt >= nt:
                data_mtx.X[idx,:,:] = np.pad( x, ((0,Nt-nt),(0,0)),'constant')
                data_mtx.Y[idx,:nt] = self.Y[idx]
                
            else:
                data_mtx.X[idx,:,:] = x[:Nt,:]
                data_mtx.Y[idx, :] = self.Y[idx][:Nt]
        return data_mtx
    ######################################################################################################
    def bound(self, min_value=None, max_value=None):
        # This function limits the amplitude value 
        
        bounded_data = copy.deepcopy(self)
        if min_value is not None:
            for x in bounded_data.X: x[ x<min_value ] = min_value
        if max_value is not None:                
            for x in bounded_data.X: x[ x>max_value ] = max_value
        
        return bounded_data
    ######################################################################################################
    def trim(self, keep_ratio=None):
        trimmed_data = copy.deepcopy(self)
        trimmed_data.X = list()
        
        if keep_ratio is None:
            dt = 20   
            for x in self.X:     
                N = len(x)
                n1, n2 = dt, N-dt 
                xx = abs( np.diff(x))
                xx = np.sum(xx, axis=1)    
                xx = abs(np.diff(xx))
                xx /= ( np.nanmax(xx) + eps )                 
                idxs = np.where( xx > 0.5 )[0]    
                idxs1 = idxs[idxs < 0.5*N] 
                idxs2 = idxs[idxs > 0.5*N]      
                if np.any(idxs1): n1 = np.min(idxs1) + dt
                if np.any(idxs2): n2 = np.max(idxs2) - dt   
                if (n2-n1) < 0.5*N: n1, n2 = 0, N            
                trimmed_data.X.append( x[n1:n2,:] )
        else:   
            for x in self.X:
                L = int( len(x) * keep_ratio)
                trimmed_data.X.append( x[:L,:] ) 

        trimmed_data.X = np.array(trimmed_data.X)    
        return trimmed_data    
    ######################################################################################################
    def quantize(self, Qstep):        
        quantized_data = copy.deepcopy(self)
        for idx, x in enumerate(quantized_data.X): 
            quantized_data.X[idx] = Qstep * np.floor(x/Qstep)
        return quantized_data   
    ######################################################################################################
    def clean(self):
        # cleans data from NANs ! 
        cleaned_data = copy.deepcopy(self)
        for idx, x in enumerate(cleaned_data.X):
            if np.any(np.isnan(x)):
                df = pd.DataFrame(x)
                df = df.fillna(method='ffill', axis=0).bfill(axis=0)      
                cleaned_data.X[idx] = df.as_matrix()

        return cleaned_data                
    ######################################################################################################
    def filter_noise(self, window_length=5, polyorder=2):
        filtered_data = copy.deepcopy(self)
        for n, x in enumerate(self.X):
            for i in range(8):
                filtered_data.X[n][:,i] = signal.savgol_filter(x[:,i], window_length, polyorder)        
        return filtered_data
    ######################################################################################################
    def MinMax(self):
        # Rescale data value to (0,1)
        MIN, MAX = np.inf, -np.inf 
        normalized_data = copy.deepcopy(self)
        for idx, x in enumerate(normalized_data.X): 
            MIN = min(MIN,np.nanmin(x,axis=0))
            MAX = max(MAX, np.nanmax(x,axis=0))

        for idx, x in enumerate(normalized_data.X): 
            normalized_data.X[idx] = np.subtract(x,MIN) / ( np.subtract(MAX,MIN) + eps )
        # normalized_data.X = (self.X - np.min(self.X))/(np.max(self.X)-np.min(self.X))
        return normalized_data    
    ######################################################################################################
    def standardize(self, scale=True):
        normalized_data = copy.deepcopy(self)
        STD = 1
        for idx, x in enumerate(normalized_data.X): 
            MEAN = np.mean(x,axis=0)
            if scale: STD = np.std(x,axis=0) + eps
            normalized_data.X[idx] = np.subtract(x,MEAN) / STD    
        return normalized_data         
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
    def __init__(self, system_info = None):
        self.reader = None
        self.tags = list()

        if system_info is not None:
            
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
    def get_data_old(self, dataset_name, file_name, save=False, window_length=11, resample_dt=None):
        reader_data = self.reader.get_data(dataset_name, file_name)  
        data = pd.DataFrame({'time':reader_data.time})

        for i, tag in enumerate(self.tags):
            tag_data = tag.get_data(dataset_name, file_name, ref_node_data=reader_data)
            tag_data = tag_data.add_suffix('_'+str(i)).rename({'time_'+str(i):'time'}, axis='columns')
            data = data.merge( tag_data, on='time', how='outer', suffixes=('', '' ), sort=True )

        data.interpolate(method='nearest', inplace=True)  
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)
        data.time -= data.time.iloc[0]

        if resample_dt is not None:
            resampled_data = pd.DataFrame({'time':np.arange(data.time.iloc[0], data.time.iloc[-1], resample_dt)})
            for column in data.columns:
                if column =='time':continue
                resampler = interpolate.interp1d(data.time.values, data[column].values, kind='linear')
                resampled_data[column] = np.nan_to_num( resampler(resampled_data.time.values) )
                # resampled_data[column] = signal.savgol_filter( resampled_data[column], window_length=window_length, polyorder=1, axis=0)     
            data = resampled_data
        # Save
        if save:
            dataset_file_path = get_dataset_file_path(dataset_name, file_name)
            create_folder(dataset_file_path)
            data.to_pickle( dataset_file_path)  
            print(file_name, 'is saved.')

        return data
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







# ####################################################################################################################################################
# class RNN(object):
#     g = None
#     sess = None    
#     # ---------------------------------------------------------------------------------------------------   
#     def __init__(self, NNtype, Nlayers, Nunits, TBlength, Nclasses, Nfeatures, **optimizerParams):        
#         self.__class__ = eval(NNtype)       
#         self.NNtype = NNtype                # Type of RNN  
#         self.Nlayers = int(Nlayers)         # Number of layers
#         self.Nunits = int(Nunits)           # Number of units
#         self.TBlength = int(TBlength)       # Truncated backpropagation length
#         self.Nclasses = int(Nclasses)       # Number of classes
#         self.Nfeatures = int(Nfeatures)     # Number of features     

#         self.optimizerParams = optimizerParams       
#     # ---------------------------------------------------------------------------------------------------   
#     def build(self):
#          if self.g is None: 
#             if 'sess' in globals() and sess: sess.close() 
#             tf.reset_default_graph() # reset graph
#             self.g = dict()
#             self.build_inputs()
#             self.build_model()
#             self.build_optimizer()
#             self.reset()     
#     # ---------------------------------------------------------------------------------------------------   
#     def build_inputs(self):
#         # Input placeholders
#         self.g['inputs'] = tf.placeholder(tf.float32, shape=[None, self.TBlength, self.Nfeatures], name='inputs_placeholder')
#         self.g['labels'] = tf.placeholder(tf.int32, shape=[None], name='labels_placeholder')
#         self.g['keepProb'] = tf.placeholder(tf.float32, name='keepProb_placeholder')    
#     # ---------------------------------------------------------------------------------------------------   
#     def build_model(self):  
#         # Batch normalization 
#         inputs_norm = tf.contrib.layers.batch_norm(self.g['inputs'])  
#         # Build Network
#         net, init_state_tuple = self.get_net()    
#         # RNN network output
#         outputs, self.g['current_state'] = tf.nn.dynamic_rnn(net, inputs_norm, initial_state=init_state_tuple)       
#         # Logits:  [batchSize, TBlength, Nunits] --> [batchSize, TBlength,  Nclasses] 
#         with tf.variable_scope('Conv_mtx'):
#             W = tf.get_variable('W', [self.Nunits, self.Nclasses])
#             b = tf.get_variable('b', [1, self.Nclasses], initializer=tf.constant_initializer(0.0))                        
#         logits = tf.reshape(tf.matmul(tf.reshape(outputs, [-1, self.Nunits]), W) + b, [-1, self.TBlength, self.Nclasses])
#         logits = tf.reduce_mean(logits,axis=1) # average over a window for logits         

#         # Predictions
#         probabilities = tf.nn.softmax(logits) # Sofmax 
#         self.g['predictions'] = tf.cast(tf.argmax(probabilities,1),tf.int32)    
        
#         # Accuracy
#         equality = tf.equal( self.g['predictions'], self.g['labels'] )
#         self.g['accuracy'] = tf.cast(equality,tf.float32)

#         # Loss
#         self.g['loss'] = tf.nn.sparse_softmax_cross_entropy_with_logits( labels=self.g['labels'], logits=logits)         
#     # ---------------------------------------------------------------------------------------------------   
#     def build_optimizer(self):  
#         # Optimizer
#         optPars = copy.deepcopy(self.optimizerParams)
#         optimizerType = optPars.pop('optimizerType')
#         learningRate = optPars.pop('learningRate')
#         globalStep = tf.Variable(0, trainable=False)  
#         expDecaySteps = 100
#         if 'expDecaySteps' in optPars: expDecaySteps = optPars.pop('expDecaySteps')
#         expDecayRate = 1
#         if 'expDecayRate' in optPars: expDecayRate = optPars.pop('expDecayRate')

#         # decayed_learningRate = learningRate * decayRate ^ (globalStep / decaySteps)      
#         decayed_learningRate = tf.train.exponential_decay(learningRate, globalStep, expDecaySteps, expDecayRate)       
        
#         # Optimizer
#         optPars.update({'learning_rate':decayed_learningRate})
#         optimizer = eval( 'tf.train.' + optimizerType )( **optPars )        

#         # Train
#         loss_avg = tf.reduce_mean(self.g['loss'])        
#         self.g['trainStep'] = optimizer.minimize(loss_avg, global_step=globalStep)      
#     # ---------------------------------------------------------------------------------------------------   
#     def train_validate( self, TrainValidation_data, isTraining=True, stride=None, batchSize=None, keepProb=1):
#         # OUTPUTS:
#         #         loss_list (vector of size Nepochs):  list of losses for all epochs 
#         #         accuracy_list (vector of size Nepochs): list of accuracy for all epochs 

#         mtx_data = TrainValidation_data.mtx()
#         if batchSize is None: batchSize = mtx_data.X.shape[0]
#         if stride is None: stride = self.TBlength 
#         if isTraining: keepProb_ = keepProb
#         else: keepProb_ = 1

#         # Train / validate
#         current_state = self.get_init_state(batchSize)
#         ep_loss_list, ep_accuracy_list, ep_prediction_list, ep_label_list = list(), list(), list(), list()
#         # results = RESULTS()                    
#         for i in range( 0,  mtx_data.X.shape[0], batchSize ):
#             for j in range( 0,  mtx_data.X.shape[1]-self.TBlength, stride ):
#                 batch_x = mtx_data.X[i:i+batchSize, j:j+self.TBlength]
#                 batch_y = mtx_data.Y[i:i+batchSize] 
#                 # Feeds
#                 feed_dict = { 
#                     self.g['init_state']:current_state, 
#                     self.g['inputs']:batch_x, 
#                     self.g['labels']:batch_y, 
#                     self.g['keepProb']:keepProb_
#                     }
#                 # Outputs
#                 requested_outputs = [ 
#                     self.g['current_state'], 
#                     self.g['loss'],  
#                     self.g['accuracy'], 
#                     self.g['predictions']
#                     ]
#                 if isTraining: requested_outputs.append(self.g['trainStep'])
#                 # Run session & get results
#                 outputs = self.sess.run(requested_outputs, feed_dict=feed_dict )                                  
#                 current_state, batch_loss_list, batch_accuracy_list, batch_prediction_list = outputs[:4]

#                 ep_loss_list.append(batch_loss_list)     
#                 ep_accuracy_list.append(batch_accuracy_list)
#                 ep_prediction_list.append(batch_prediction_list)
#                 ep_label_list.append(batch_y)
        
#         ep_loss_list = np.array(ep_loss_list).flatten()
#         ep_accuracy_list = np.array(ep_accuracy_list).flatten()
#         ep_prediction_list = np.array(ep_prediction_list).flatten()
#         ep_label_list = np.array(ep_label_list).flatten()

#         return ep_loss_list, ep_accuracy_list, ep_prediction_list, ep_label_list    
#     # ---------------------------------------------------------------------------------------------------   
#     def run(self, training_data, validation_data, Nepochs=200, **trainingParams):
        
#         training_results = dict({  'loss' : list(), 'accuracy' : list(), 'predictions': list(), 'labels' : list() })
#         validation_results = dict({  'loss' : list(), 'accuracy' : list(), 'predictions': list(), 'labels' : list() })

#         for ep in range( Nepochs ): 
#             # Training
#             ep_loss_list, ep_accuracy_list, ep_prediction_list, ep_label_list = self.train_validate(training_data, isTraining=True, **trainingParams)
#             training_results['loss'].append(ep_loss_list)
#             training_results['accuracy'].append(ep_accuracy_list)
#             training_results['predictions'].append(ep_prediction_list)
#             training_results['labels'].append(ep_label_list)

#             # Validation                
#             ep_loss_list, ep_accuracy_list, ep_prediction_list, ep_label_list = self.train_validate(validation_data, isTraining=False, **trainingParams)
#             validation_results['loss'].append(ep_loss_list)
#             validation_results['accuracy'].append(ep_accuracy_list)
#             validation_results['predictions'].append(ep_prediction_list)
#             validation_results['labels'].append(ep_label_list)
            
#         return training_results, validation_results                     
#     # ---------------------------------------------------------------------------------------------------       
#     def reset(self):
#         self.sess = tf.Session() 
#         self.sess.run(tf.global_variables_initializer()) # initialize the graph
#     # ---------------------------------------------------------------------------------------------------       
#     def save_model(self, folderPath):  
#         # save checkpoints 
#         checkpoints_folderPath = folderPath + '/' + checkpoints_folderName
#         create_folder(checkpoints_folderPath) 
#         saver = tf.train.Saver()       
#         saver.save(self.sess, checkpoints_folderPath + '/' + checkpoints_modelName)        
#     # ---------------------------------------------------------------------------------------------------       
#     def restore_model(self, folderPath):
#         # folderPath: '.../NNtype LSTM, ....Nepoch 10'
#         checkpoints_folderPath = folderPath + '/' + checkpoints_folderName
#         saver = tf.train.Saver()        
#         saver.restore(self.sess, checkpoints_folderPath + '/' + checkpoints_modelName)                    
# ####################################################################################################################################################
# class vanillaRNN(RNN):
#     # ---------------------------------------------------------------------------------------------------       
#     def get_net(self):
#         net = []
#         self.g['init_state'] = tf.placeholder(tf.float32, [self.Nlayers, None, self.Nunits], name='initState_placeholder')
#         l = tf.unstack(self.g['init_state'], axis=0)
#         init_state_tuple = tuple([l[idx] for idx in range(self.Nlayers)])
#         for i in range(self.Nlayers): 
#             cell = tf.contrib.rnn.BasicRNNCell(self.Nunits) 
#             # cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.g['keepProb'])  
#             cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.g['keepProb'], output_keep_prob=self.g['keepProb'], state_keep_prob=self.g['keepProb'] )
#             net.append(cell)
#         net = tf.contrib.rnn.MultiRNNCell(net)          
#         return net, init_state_tuple
#     # ---------------------------------------------------------------------------------------------------       
#     def get_init_state(self, batchSize):
#         return np.zeros((self.Nlayers, batchSize, self.Nunits))
# ####################################################################################################################################################
# class LSTM(RNN):
#     # ---------------------------------------------------------------------------------------------------       
#     def get_net(self):
#         net = []
#         self.g['init_state'] = tf.placeholder(tf.float32, [self.Nlayers, 2, None, self.Nunits], name='initState_placeholder')
#         l = tf.unstack(self.g['init_state'], axis=0)    
#         init_state_tuple = tuple([tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1]) for idx in range(self.Nlayers)])        
#         for i in range(self.Nlayers): 
#             cell = tf.contrib.rnn.BasicLSTMCell(self.Nunits, state_is_tuple=True)
#             # cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.g['keepProb'])    
#             cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.g['keepProb'], output_keep_prob=self.g['keepProb'], state_keep_prob=self.g['keepProb'] )
#             net.append(cell)
#         net = tf.contrib.rnn.MultiRNNCell(net, state_is_tuple=True)    
#         return net, init_state_tuple
#     # ---------------------------------------------------------------------------------------------------       
#     def get_init_state(self, batchSize):
#         return np.zeros((self.Nlayers, 2, batchSize, self.Nunits))
# ####################################################################################################################################################

