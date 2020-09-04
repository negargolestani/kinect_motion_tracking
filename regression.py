
from utils import*




####################################################################################################################################################
def load_dataset(dataset_name, resample_dt=None, as_dict=True):
    converters = {key:lambda x: list(map(float, x.strip('[]').split())) for key in ['norm', 'center_1', 'center_2']}
    
    dataset = list()
    for file_path in glob.glob(get_dataset_folder_path(dataset_name) +'/*.csv'):        
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
            y = np.linalg.norm( np.array( data[target].to_list() ), axis=1)
            
            self.X.append(x)
            self.Y.append(y)
        
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        return         
    ######################################################################################################
    def segment(self, win_size, step=None, as_df=False, merge=True):
        # returns Nsample list of n * win_size * Nf   where n is number of segments extracted from Nt samples 
        if step is None: step = win_size                
        X, Y = list(), list()
        for (x,y) in zip(self.X, self.Y):
            Nt = np.shape(x)[0]
            # x_s = [ x[t:t+win_size,:] for t in range(0, Nt-win_size, step) ]
            # y_s = [ y[t+win_size] for t in range(0, Nt-win_size, step) ]
            x_s = [ x[t:t+win_size,:] for t in range(0, Nt-win_size+1, step) ]
            y_s = [ y[t+win_size-1] for t in range(0, Nt-win_size+1, step) ]
            X.append(x_s)
            Y.append(y_s)
        data_segmented = DATA(X, Y)

        if as_df:            
            if data_segmented.X.ndim == 1:            
                Nf = np.shape(data_segmented.X[0])[2]
                data_segmented_list = list()
                for (x,y) in zip(data_segmented.X, data_segmented.Y):
                    data_df = pd.DataFrame( np.concatenate( [np.reshape(x, (-1, Nf*win_size)), np.reshape(y, (-1,1))], axis=1) )
                    data_df.columns = [*['feature_'+str(i) for i in range(win_size*Nf)], 'target']
                    data_segmented_list.append(data_df)
                return data_segmented_list

            if data_segmented.X.ndim == 4:            
                Nf = np.shape(data_segmented.X)[3]                
                data_segmented_df = pd.DataFrame( np.concatenate( [data_segmented.X.reshape(-1, Nf*win_size), data_segmented.Y.reshape(-1,1)], axis=1) )
                data_segmented_df.columns = [*['feature_'+str(i) for i in range(win_size*Nf)], 'target']
                return data_segmented_df

        return data_segmented         
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

####################################################################################################################################################
class REGRESSOR(object):
    ################################################################################################################################################
    def __init__(self, win_size, step=1, **params ):
        np.random.seed(7)        
        self.win_size = win_size
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
        self.model.add( LSTM(units=Nunits, activation=activation, return_sequences=True, input_shape=(self.win_size, )) )        
        self.model.add( LSTM(units=Nunits, activation=activation, return_sequences=True) )        
        self.model.add( LSTM(units=Nunits, activation=activation))   
        # self.model.add(Dropout(0.1))
        self.model.add( Dense(1))          
            
        self.model.compile(loss='mse', optimizer='adam')         
        print(self.model.summary())        
        return 
####################################################################################################################################################
class TCN(REGRESSOR):
    def build_model(self, Nlayers=2, nb_filters=5, activation=None, optimizer='adam', loss='mse' ):  
        i = tf.keras.Input(batch_shape=(None, self.win_size)) 
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
