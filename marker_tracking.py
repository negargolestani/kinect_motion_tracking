from utils import*


################################################################################################################################################
if __name__ == '__main__':

    color_setting = load_color_Setting(file_name='color_setting_default')


    file_name = 'record_04'
    record = RECORD( file_name )
    
 
    color = 'purple'
    color_range = color_setting[color]  


    for frame in record.frames:
        contours = frame.color_2_circle_contours(color_range, n_Contours=2)  
        frame.show(contours=contours, wait=1000)  
   
    # motion = record.get_makers_motion(color_range,n_markers=2)
    # motion.save(file_name)
    


