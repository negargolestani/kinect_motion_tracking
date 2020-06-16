from utils import*

####################################################################################################################################################
class COIL(object):
    ################################################################################################################################################
    def __init__(self, markers_color):   
        self.markers_color = markers_color     
        self.time = list()
        self.center = list()
        self.norm = list()
    ################################################################################################################################################
    def update(self, time, markers):
        if len(markers) ==2:
            center = ( markers[0]  + markers[1] ) /2    
            norm = np.abs( markers[0] - markers[1]) 
            norm =  norm/np.sqrt(sum(norm**2))            
            
            self.time.append( time )
            self.center.append( center )
            self.norm.append( norm )
        return 
    ################################################################################################################################################
    def save(self, file_path):        
        pickle.dump(self.__dict__, open(file_path, 'wb'))
        return
    ################################################################################################################################################
    def distance(self, coil):      
        dist = sqrt(sum((self.center - coil.center)**2))
        ang_misalign_deg = acos(np.dot(self.norm, coil.norm))*180/pi
        return dist, ang_misalign_deg
####################################################################################################################################################

 