from utils import*





################################################################################################################################################
if __name__ == '__main__':

    # a = [[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]]
    
    # file_name = 'record_05'
    # record = load_record( file_name )
    

    # n_markers = 3
    # marker_radius = 15
    # color_setting_filename = 'color_setting_default'
    # color = 'blue'
    # markerset = MARKERSET( color, n_markers, marker_radius, color_setting_filename)
    


    # frame = record[0]

    # circles = frame.get_colored_circle(markerset.color_range, 3, 15)

    # frame.show(contours=circles, wait=0)
    # cv2.imwrite('image.png', record[0].color_image)

    # cv2.imwrite(file_path, frame.color_image) 
    file_path = markers_folder_path + '/' + 'record_05_blue' + '.txt'
    # f = open(file_path, 'r+')
    data = np.loadtxt(file_path, delimiter=delimiter)
    # data = f.read()
    # f.close()

    print(data)