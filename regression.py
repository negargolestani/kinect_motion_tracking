
from utils import*

eps = 1e-12

####################################################################################################################################################
def load_dataset(dataset_name, resample_dt=None, as_dict=True):
    dataset_type = dataset_name.split('_')[0]

    if dataset_type == 'arduino' or dataset_type =='meas':
        return load_dataset_meas(dataset_name, resample_dt=resample_dt, as_dict=as_dict)
    
    elif dataset_type == 'synth':
        return load_dataset_synth(dataset_name, resample_dt=resample_dt, as_dict=as_dict)
        
    # return globals()['load_dataset_'+dataset_name.split('_')[0]](dataset_name, resample_dt=resample_dt, as_dict=as_dict)
####################################################################################################################################################
def load_dataset_synth(dataset_name, resample_dt=None, as_dict=True):
    synth_dataset_folder_path = get_dataset_folder_path(dataset_name)

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
def load_dataset_meas(dataset_name, resample_dt=None, as_dict=True):
    dataset = list()
    for file_path in glob.glob(get_dataset_folder_path(dataset_name) +'/*.csv'):        
        data = pd.read_csv(file_path)

        if resample_dt is not None:
            resampled_data = pd.DataFrame({'time':np.arange(data.time.iloc[0], data.time.iloc[-1], resample_dt)})
            for column in data.columns:
                if column =='time':continue
                resampler = interpolate.interp1d(data.time.values, data[column].values, kind='linear')
                resampled_data[column] = np.nan_to_num( resampler(resampled_data.time.values) )
                # resampled_data[column] = signal.savgol_filter( resampled_data[column], window_length=window_length, polyorder=1, axis=0)             
            data = resampled_data        
        dataset.append(data)

    if as_dict:
        dataset_dict = dict()
        for column in dataset[0].columns: dataset_dict.update({ column : np.array([ data[column].values for data in dataset]) })
        return dataset_dict

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
    def __init__(self, X=[], Y=[]):            
        # self.X = np.array(X)  
        self.X = np.array(X) / (1+eps)  # Don't know why !!!  have to divide to handle pycaret error in training 
        self.Y = np.array(Y)        
        return
    ######################################################################################################        
    def segment(self, win_size, step=None, as_df=False):
        if step is None: step = win_size        
        
        X, Y = list(), list()
        for t in range(0, self.X.shape[1] - win_size, step): 
            X = [*X, *self.X[:,t:t+win_size]]
            Y = [*Y, *self.Y[:,t+win_size]]           
        data_segmented = DATA(X, Y)
       
        if as_df:
            data_df = pd.DataFrame( np.concatenate([data_segmented.X, np.reshape(data_segmented.Y,(-1,1))], axis=1) )
            data_df.columns = [*['feature_'+str(i) for i in range(win_size)], 'target']
            return data_df

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

        Nt_list = [np.shape(x)[0] for x in self.X]
        Nt = int( eval('np.' + Nt_mtx)(Nt_list) )
        Nd = len(self.X)
        
        data_mtx.X = np.zeros( (Nd,Nt) )
        data_mtx.Y = np.zeros( (Nd,Nt) )
        
        for idx, x in enumerate(self.X): 
            nt = np.shape(x)[0]
            
            if Nt >= nt:
                data_mtx.X[idx,:nt] = x
                data_mtx.Y[idx,:nt] = self.Y[idx]
                
            else:
                data_mtx.X[idx,:] = x[:Nt]
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
            MEAN = np.nanmean(x,axis=0)
            if scale: STD = np.nanstd(x,axis=0) + eps
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

