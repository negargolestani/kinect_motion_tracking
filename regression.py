
from utils import*




####################################################################################################################################################
def load_dataset(dataset_name, resample_dt=None, as_dict=True):
    file_path_list = glob.glob(get_dataset_folder_path(dataset_name) +'/*.csv')
    
    data = pd.read_csv(file_path_list[0])
    columns = list(set(['time', *data.filter(regex='norm').columns, *data.filter(regex='center_').columns, *data.filter(regex='vind').columns]))
    
    converters = dict()
    for column in columns:
        val = data.loc[0,column]
        if type(val) == str:
            if ',' in val: converters.update({ column: literal_eval })
            else: converters.update({ column:lambda x: list(map(float, x.strip('[]').split())) })
    
    # converters = {key:lambda x: list(map(float, x.strip('[]').split())) for key in ['norm', 'center_1', 'center_2']}
    # converters = {key:literal_eval for key in ['norm', 'center_1', 'center_2']}
    
    dataset = list()
    for file_path in file_path_list:        
        data = pd.read_csv(file_path, converters=converters)

        if resample_dt is not None:
            resampled_data = pd.DataFrame({'time':np.arange(data.time.iloc[0], data.time.iloc[-1], resample_dt)})
            for column in data.columns:
                if column =='time':continue
                resampler = interpolate.interp1d(data.time.values, data[column].values, kind='linear')
                resampled_data[column] = np.nan_to_num( resampler(resampled_data.time.values) )
                # resampled_data[column] = signal.savgol_filter( resampled_data[column], window_length=window_length, polyorder=1, axis=0)             
            data = resampled_data        
        dataset.append(data)

    return dataset
####################################################################################################################################################
####################################################################################################################################################
def get_delay(x, y):
    # delay < 0 means that y starts 'delay' time steps before x 
    # delay > 0 means that y starts 'delay' time steps after x
    assert len(x) == len(y)
    c = fftshift( np.real(ifft(fft(x) * fft(np.flipud(y)))) )   # cross correlation using fft
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    delay = zero_index - np.argmax(c)
    return delay
####################################################################################################################################################


####################################################################################################################################################
class DATA(object):    
    ######################################################################################################
    def __init__(self, X=[], Y=[], dataset_name=None, **params): 
        if dataset_name is not None:
            self.load( dataset_name, **params)   
        else:
            self.X = np.array(X)  
            self.Y = np.array(Y)        
        return
    ######################################################################################################
    def load( self, dataset_name, features=['synth_vind_1', 'synth_vind_2'], target='center_1'):
        
        dataset_df_list = load_dataset(dataset_name, as_dict=False)
        self.X, self.Y = list(), list()

        for data in dataset_df_list:
            
            x = np.zeros((len(data), len(features)))
            for i, feature in enumerate(features): x[:,i] = data[feature].to_list()

            y = np.array( data[target].to_list() )
            if target == 'center_1' or target =='center_2': y = np.linalg.norm( y, axis=1)
            elif target == 'norm': y = np.arccos( y[:,2]) * 180/pi

            self.X.append(x)
            self.Y.append(y)
        
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        return         
    ######################################################################################################
    def segment(self, win_size, step=None):
        # returns Nsample list of n * win_size * Nf   where n is number of segments extracted from Nt samples 
        if step is None: step = win_size                
        X, Y = list(), list()
        for (x,y) in zip(self.X, self.Y):
            Nt = np.shape(x)[0]
            # x_s = [ x[t:t+win_size,:] for t in range(0, Nt-win_size, step) ]
            # y_s = [ y[t+win_size] for t in range(0, Nt-win_size, step) ]
            x_s = np.array([ x[t:t+win_size,:] for t in range(0, Nt-win_size+1, step) ])
            y_s = np.array([ y[t+win_size-1] for t in range(0, Nt-win_size+1, step) ])
            X.append(x_s)
            Y.append(y_s)
        return DATA(X, Y)         
    ######################################################################################################
    def get_features(self):
        features = np.array([ get_features_sample(x) for x in self.X ])
        return DATA(X=features, Y=self.Y)
    ######################################################################################################
    def get_df(self, merge=False):
        if merge:
            X = np.concatenate( self.X, axis=0)
            Nf = X.shape[1]*X.shape[2]
            X = X.reshape(-1, Nf)
            data_df = pd.DataFrame( X, columns=['feature_'+str(i) for i in range(Nf)] )
            data_df['target'] = np.concatenate( self.Y, axis=0)  

            return data_df

        else:
            data_df_list= list()
            for (x,y) in zip(self.X, self.Y):
                
                Nf = x.shape[1]*x.shape[2]
                x = x.reshape(-1, Nf)
                data_df = pd.DataFrame( x, columns=['feature_'+str(i) for i in range(Nf)] )
                data_df['target'] = y
                data_df_list.append(data_df)

            return data_df_list
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
        # random.shuffle(idxs)
        
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

        Nd, Nf = len(self.X),  np.shape(self.X[0])[1]
        Nt_list = list()
        for x in self.X: Nt_list.append( np.shape(x)[0] )
        if type(Nt_mtx) is str: Nt = int( eval('np.' + Nt_mtx)(Nt_list) )
        else:  Nt = Nt_mtx
        data_mtx.X = np.zeros( (Nd,Nt,Nf) )
        data_mtx.Y = np.zeros( (Nd,Nt) )
        for idx, (x,y) in enumerate(zip(self.X, self.Y)): 
            # x = np.subtract(x,np.mean(x,axis=0))        
            nt = np.shape(x)[0]
            if Nt >= nt:
                # data_mtx.X[idx,:,:] = np.pad( x, ((0,Nt-nt),(0,0)),'constant')
                data_mtx.X[idx,:nt,:] = x
                data_mtx.Y[idx,:nt] = y
            else:
                data_mtx.X[idx,:] = x[:Nt,:]
                data_mtx.Y[idx,:] = y[:Nt]
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
            for i in range(np.shape(x)[1]):
                filtered_data.X[n][:,i] = signal.savgol_filter(x[:,i], window_length, polyorder)        
        return filtered_data
    ######################################################################################################
    def MinMax(self):
        # Rescale data value to (0,1)
        normalized_data = copy.deepcopy(self)
        for idx, x in enumerate(normalized_data.X): 
            MIN = np.nanmin(x,axis=0)
            MAX = np.nanmax(x,axis=0)
            normalized_data.X[idx] = np.subtract(x,MIN) / ( np.subtract(MAX,MIN) + eps )
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
def get_features_sample(x_t):
    features = list()   
    axis = -2
    x_f = np.real( np.fft.fft(x_t, axis=axis) )
    x_wA, x_wD = pywt.dwt(x_t, 'db1', axis=axis)
    dx_t = np.diff( x_t, axis=axis )
    for x_ in [x_t, x_f, x_wA, x_wD, dx_t]:                                    
        features.append( np.mean( x_, axis=axis)) 
        features.append( np.std( x_, axis=axis ))                               
        features.append( np.median( x_, axis=axis ))               
        features.append( np.min( x_, axis=axis ))               
        features.append( np.max( x_, axis=axis ))               
        features.append( np.var( x_, axis=axis ))               
        features.append( np.percentile( x_, 25, axis=axis ))               
        features.append( np.percentile( x_, 75, axis=axis ))               
        features.append( stats.skew( x_, axis=axis))               
        features.append( stats.kurtosis( x_, axis=axis))               
        features.append( stats.iqr( x_, axis=axis))               
        features.append( np.sqrt(np.mean(np.power(x_,2), axis=axis)))   
    features = np.concatenate(np.array(features), axis=-1)

    if np.ndim(features) == 1: return features.reshape(60,2)
    return features.reshape(-1, 60, 2)

    # if np.ndim(features) == 1: return features.reshape(-1,1)
    # N, Nf = np.shape(features)
    # return features.reshape(N, Nf, 1)
####################################################################################################################################################



####################################################################################################################################################
class RNN(object):
    ######################################################################################################    
    def __init__(self, win_size, Nfeatures, **params ):
        np.random.seed(7)        
        self.win_size = win_size
        self.Nfeatures = Nfeatures
        self.build_model(**params)
    ######################################################################################################
    def build_model(self, Nunits=3, NhiddenLayers=1, activation='softmax', optimizer='RMSprop', loss='mean_squared_error', dropout=0):        
        self.model = Sequential()           
        self.model.add( LSTM(units=Nunits, input_shape=(self.win_size, self.Nfeatures), return_sequences=True, dropout=dropout) )    
        for n in range(NhiddenLayers): 
            self.model.add( LSTM(units=Nunits, return_sequences=True, dropout=dropout) )        
        self.model.add( LSTM(units=Nunits, dropout=dropout) ) 
        # self.model.add( BatchNormalization() )          
        self.model.add( Dense(1, activation=activation) )                      
        self.model.compile(loss=loss, optimizer=optimizer)  
        print(self.model.summary())        
        return 
    ######################################################################################################
    def train(self, train_data, step=None, **params): 
        if step is None: step = self.win_size            
        train_data_sg = train_data.segment(self.win_size, step)
        history = self.model.fit( 
            np.concatenate( train_data_sg.X, axis=0), 
            np.concatenate( train_data_sg.Y, axis=0).reshape(-1,1), 
            **params) 
        return history
    ######################################################################################################    
    def predict(self, X, window_length=None):
        predictions = list()
        for x in X:
            pred = np.array([self.model.predict( np.reshape(x[t:t+self.win_size,:],[1,self.win_size, -1]) ) for t in range(np.shape(x)[0]-self.win_size)]).flatten()
            if window_length is not None: pred = signal.savgol_filter( pred, window_length=window_length, polyorder=1)       
            predictions.append( pred )  
        return predictions 
####################################################################################################################################################
class CNN(object):
        ######################################################################################################    
    def __init__(self, win_size, Nfeatures, **params ):
        np.random.seed(7)        
        self.win_size = win_size
        self.Nfeatures = Nfeatures
        self.build_model(**params)
    ######################################################################################################
    def build_model(self, stride=None, activation='softmax', optimizer='RMSprop', loss='mean_squared_error', dropout=0):                   
        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=3, stride=stride, activation='linear', padding='same', input_shape=(self.win_size, self.Nfeatures)))
        # self.model.add(Conv2D(64, kernel_size=3, activation='linear', padding='same'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(512, activation=activation))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(1))                     
        self.model.compile(loss=loss, optimizer=optimizer)  
        print(self.model.summary())        
        return 
    ######################################################################################################
    def train(self, train_data, **params): 
        history = self.model.fit( train_data.X, train_data.Y, **params) 
        return history
    ######################################################################################################    
    def predict(self, X, window_length=None):
        predictions = list()
        for x in X:
            pred = np.array([self.model.predict( np.reshape(x[t:t+self.win_size,:],[1,self.win_size, -1]) ) for t in range(np.shape(x)[0]-self.win_size)]).flatten()
            if window_length is not None: pred = signal.savgol_filter( pred, window_length=window_length, polyorder=1)       
            predictions.append( pred )  
        return predictions 
####################################################################################################################################################
class TCN(object):
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
