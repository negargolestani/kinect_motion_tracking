from utils import*





################################################################################################################################################
if __name__ == '__main__':

    # n_markers = 3
    # marker_radius = 15
    # color_setting_filename = 'color_setting_default'
    # color = 'blue'
    # markerset = MARKERSET( color, n_markers, marker_radius, color_setting_filename)
    
    file_name = 'record_06'
    record = load_record( file_name )
    frame = record[0]

    frame.show( wait=0 )
        
    # file_path = calibsest_folder_path + '/' + image_name + '.png'
    # create_folder(file_path)                                       

    cv2.imwrite('new.png', frame.color_image)     
