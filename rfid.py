from utils import*


################################################################################################################################################
class TAG(object):
    ############################################################################################################################################
    def __init__(self, IDD, time=None, rssi=None):
        self.IDD = IDD
        if time is None or rssi is None:
            self.time = list()
            self.rssi = list()
        else:
            self.time = time
            self.rssi = rssi        
    ############################################################################################################################################
    def show(self, tag_label=None):
        # Plot RSSI data of all tags in inventory 
        fs = 12
        if tag_label is None: tag_label = self.IDD
        plt.plot(self.time, self.rssi,  label=tag_label)
        plt.legend(fontsize=fs)
        plt.xlabel( 'Time', fontsize=fs )
        plt.ylabel( 'RSSI', fontsize=fs )
        plt.show()
    #######################################################################################################
################################################################################################################################################
class RFID(object):
    ############################################################################################################################################
    def __init__(self, file_name):
        rssi_file_path = get_rssi_file_path(file_name)
       
        # Read IDD, Time, RSSI values from file
        df = pd.read_csv(
            rssi_file_path,                                                     # relative python path to subdirectory
            delimiter  = ';',                                                   # Tab-separated value file.
            usecols = ['IDD', 'Time', 'Ant/RSSI'],                              # Only load the three columns specified.
            parse_dates = ['Time']                                              # Intepret the birth_date column as a date
        )

        # Processing
        idd = df['IDD'].astype(str).to_numpy()
        rssi = df['Ant/RSSI'].str.replace('Ant.No 1 - RSSI: ', '').astype(int).to_numpy()                  
        time = list()
        for t in df['Time'].astype(str): time.append( datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') )
        time = np.array(time)

        self.inventory = list()
        for IDD in list(set(idd)):
            tag = TAG(IDD, time=time[idd==IDD], rssi=rssi[idd==IDD])
            self.inventory.append(tag)
        return
    ############################################################################################################################################
    # def show(Self):
    #     for tag in self.inventory:
    #         plt.figure()

################################################################################################################################################
  


#######################################################################################################
if __name__ == "__main__":     
    # file_directory_list = glob.glob( '../data/*.csv')
    # file_directory = file_directory_list[0]
#     tags_label = dict({'E00700001ED1AADA':'TX1', 'E00700001ED1AA59':'TX2'})

    rfid = RFID('record_01')

    for tag in rfid.inventory:
        tag.show()
        plt.hold(True)
#######################################################################################################
