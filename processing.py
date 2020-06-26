from utils import*


################################################################################################################################################
if __name__ == '__main__':


    file_name = 'record_02'   

    exp = EXPERIMENT(file_name, reader_color='blue', tags_color='red')
    distance_list, ang_misalign_list = exp.status()

    
    plt.figure()
    plt.plot( distance_list[0] )
    plt.figure()
    plt.plot( ang_misalign_list[0] )
    plt.show()
   