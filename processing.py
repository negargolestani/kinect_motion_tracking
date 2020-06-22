from utils import*


################################################################################################################################################
if __name__ == '__main__':


    # file_name = 'record_04_green'    
    file_name = 'record_05'    
    markers_motion = load_markers(file_name + '_blue')
    times = load_times(file_name)

    coil = COIL(markers_motion)

    plt.plot( coil.center )
    plt.show()


    