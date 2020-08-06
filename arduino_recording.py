from utils import* 

import serial

####################################################################################################################################################
class ARDUINO(object):
    def __init__(self, ports, baudrate=9600):
        self.ports = ports
        self.sers = list()
        for i, port in enumerate(ports):
            ser = serial.Serial(port, baudrate=baudrate)
            ser.close()
            ser.open()
            self.sers.append(ser)
        return
    ################################################################################################################################################
    def read(self, scale=5/1024):
        self.time = datetime.now().time()
        self.vind = [float(ser.readline().decode()) * scale for ser in self.sers]
        return 
####################################################################################################################################################
        


####################################################################################################################################################
if __name__ == '__main__':
    
    dataset_name ='dataset_05'
    file_name = 'record_00'
    record_time = 40

    # Initialization
    arduino = ARDUINO([ 
        '/dev/cu.usbserial-1410',
        '/dev/cu.usbserial-1420'
        ])   
    arduino_file_path = get_arduino_file_path(dataset_name, file_name)
    create_folder(arduino_file_path)
    data_df = pd.DataFrame()

    # Show when arduino starts recording
    print('Recording is Started \n  Press "Esc" Key to Stop Recording')
    arduino.read()
    
    # Recording loop
    start_time = datetime.combine(date.min, arduino.time)
    while cv2.waitKey(1)!=27 and (datetime.combine(date.min, arduino.time) - start_time).total_seconds() < record_time:  
        arduino.read()

        data_dict = dict({'time':arduino.time})
        for (port,vind) in zip(arduino.ports, arduino.vind): data_dict.update({ port:vind }) 
        data_df = data_df.append( data_dict , ignore_index=True)

    print('Recording is Finished \n Wait for Saving ...')    
    data_df.to_csv(arduino_file_path, index=False)        
    print('Done!')           
################################################################################################################################################
