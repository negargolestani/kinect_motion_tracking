from utils import*

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras import backend as K

####################################################################################################################################################
def load_data(markers_file_path, markers_color, window_length=11):
    raw_df  = pd.read_csv(
        markers_file_path,                                             # relative python path to subdirectory
        usecols = ['time', markers_color],                             # Only load the three columns specified.
        parse_dates = ['time'] )         

    # Time
    date_time = pd.to_datetime( raw_df['time'] , format=datime_format)
    time = [ np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time]
    time = time-time[0]

    # Markers
    markers = [list(map(float, l.replace(']','').replace('[','').replace('\n','').split(", "))) for l in raw_df[markers_color].values]  
    markers_npy = np.array(markers).reshape(len(time), -1, 3)
    # DON'T Smooth markers. markers can be switched in array and smoothing causes error    

    # Center    
    center = np.mean(markers_npy, axis=1)         
    center = np.nan_to_num(center)
    center = signal.savgol_filter( center, window_length=window_length, polyorder=1, axis=0)     

   # Norm
    norm = np.cross( markers_npy[:,1,:] - markers_npy[:,0,:], markers_npy[:,2,:] - markers_npy[:,0,:])
    norm = norm / ( np.reshape(np.linalg.norm(norm, axis=1), (-1,1)) * np.ones((1,3)))
    # DOn't smooth norm 
    
    time_new = np.arange(0, 35, .1)
    center_new, norm_new = np.zeros((len(time_new),3)), np.zeros((len(time_new),3))
    resampler_kind = 'linear'
    for i in range(3):
        resampler = interpolate.interp1d(time, center[:,i], kind=resampler_kind)
        c_new =  np.nan_to_num( resampler(time_new) )
        center_new[:,i] = signal.savgol_filter( c_new, window_length=window_length, polyorder=1, axis=0)     

        resampler = interpolate.interp1d(time, norm[:,i], kind=resampler_kind)
        n_new =  np.nan_to_num( resampler(time_new) )
        n_new = signal.savgol_filter( n_new, window_length=window_length, polyorder=1, axis=0)    
        norm_new[:,i] = n_new / np.linalg.norm(n_new)

    return center_new, norm_new
####################################################################################################################################################
def get_rotationMatrix(XrotAngle, YrotAngle, ZrotAngle):
    Rx = np.array([ [1, 0,0], [0, cos(XrotAngle), -sin(XrotAngle)], [0, sin(XrotAngle), cos(XrotAngle)] ])
    Ry = np.array([ [cos(YrotAngle), 0, sin(YrotAngle)], [0, 1, 0], [-sin(YrotAngle), 0, cos(YrotAngle)] ])
    Rz = np.array([ [cos(ZrotAngle), -sin(ZrotAngle), 0], [sin(ZrotAngle), cos(ZrotAngle), 0], [0, 0, 1] ])
    Rtotal =  np.matmul(np.matmul(Rz,Ry),Rx)
    return Rtotal
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
def get_dataset(dataset_name):
    N = 10
    motion_data = list()
    folder_path = os.path.dirname( get_markers_file_path(dataset_name,'') )   
    for file_path in list(Path(folder_path).glob('*.csv')):  
        ref_center, ref_norm = load_data(file_path, 'red')
        ref_center, ref_norm = np.mean(ref_center[:N], axis=0), np.mean(ref_norm[:N], axis=0)                   
        
        for color in ['blue','green']:
            center, norm = load_data(file_path, color)
            motion_data_n = list()
            for t in range(center.shape[0]):
                coilsDistance, xRotAngle = calculate_params(ref_center, ref_norm, center[t], norm[t])
                motion_data_n.append( [*coilsDistance, xRotAngle]) 
            motion_data_n = signal.savgol_filter( motion_data_n, window_length=11, polyorder=1, axis=0)     
           
            motion_data.append(motion_data_n)
    return np.array(motion_data)
####################################################################################################################################################
def get_dataset_(dataset_name):
    motion_data = list()
    folder_path = os.path.dirname( get_markers_file_path(dataset_name,'') )   
    for file_path in list(Path(folder_path).glob('*.csv')):  
        for color in ['blue','green']:
            center, norm = load_data(file_path, color)
            motion_data.append( np.concatenate([center,norm], axis=1) )
    return np.array(motion_data)
####################################################################################################################################################



####################################################################################################################################################
def nll(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
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
    def __init__(self, hiddendim, latentdim, Nt):
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
        self.vae.compile(optimizer='rmsprop', loss=nll)
        
        return
    ################################################################################################################################################
    def train(self, train_data, epochs=200):
        self.build()
        self.data_range = [np.min(train_data), np.max(train_data)]

        train_data_normalized = (train_data - np.min(train_data)) / (np.max(train_data) - np.min(train_data))
        self.vae.fit(train_data_normalized, train_data_normalized, epochs=epochs, validation_data=(train_data_normalized, train_data_normalized), verbose=0)
        return    
    ################################################################################################################################################
    def generate(self, N, window_length=11):
        codes = np.random.multivariate_normal( [0]*self.latentdim, np.eye(self.latentdim), N)
        synth_data = self.decoder.predict(codes)

        synth_data = signal.savgol_filter( synth_data, window_length=window_length, polyorder=1, axis=1)     

        # synth_data = (synth_data - np.mean(synth_data)) / (np.std(synth_data))
        # synth_data = (synth_data - np.min(synth_data)) / (np.max(synth_data) - np.min(synth_data))
        # synth_data = synth_data * (self.data_range[1]-self.data_range[0]) + self.data_range[0] 
        return synth_data[:,50:]
####################################################################################################################################################



####################################################################################################################################################
if __name__ == '__main__':

    motion_data = get_dataset('dataset_07')
    N = 1000
    batch_Size, Nt, Nf = motion_data.shape

    synth_data = np.zeros((N, Nt-50, Nf))
    synthesizer = SYNTHESIZER( hiddendim=300, latentdim=100, Nt=Nt)

    for nf in range(Nf):
        train_data = motion_data[:,:,nf]
        synthesizer.train(train_data, epochs=500)
        synth_data[:,:,nf] = synthesizer.generate(N)
    synth_data[:,:,-1] *= 90


    # Save 
    dataset_name = 'dataset_00'
    folder_path = '../synthetic_data/' + dataset_name 
    create_folder(folder_path+'/motion/record_00.csv')
    # .npy
    np.save(folder_path + '/motion_data.npy' , synth_data)  
    # .csv
    for n in range(1000):
        file_name = 'synth_record_' + "{0:0=3d}".format(n) + '.csv'
        np.savetxt( folder_path + '/motion/' + file_name, np.asarray(synth_data[n,:,:]))