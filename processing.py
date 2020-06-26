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
    
    # print(np.shape(distance_list[0]))
    
    # plt.plot(distance_list)
    # plt.show()
    
#     coil1 = exp.reader
#     coil2 = exp.tags[0]

#     ang_misalign = np.arccos(np.sum(np.multiply(coil1.norm, coil2.norm), axis=1)) * 180/np.pi
# # 
#     print(np.shape(ang_misalign))

    # print( np.linalg.norm([3,4,0]) )