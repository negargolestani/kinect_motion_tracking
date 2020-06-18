from utils import*


################################################################################################################################################
if __name__ == '__main__':

    color_setting = load_color_Setting(file_name='color_setting_default')


    file_name = 'record_04'
    record = RECORD( file_name )
    
 
    color = 'green'
    color_range = color_setting[color]  
   
    motion = record.get_makers_motion(color_range,n_markers=2, show=False)
    motion.save(file_name + '_' + color)
    


