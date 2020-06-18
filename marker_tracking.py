from utils import*


################################################################################################################################################
if __name__ == '__main__':

    file_name = 'record_04'
    record = RECORD( file_name )
    
    color = 'green'
    color_ranges = pickle.load( open( calibsest_folder_path + '/' + color_range_filename + '.pickle' , 'rb') )  
    color_range = color_ranges[color]  


    motion = record.get_makers_motion(color_range,n_markers=2)
    motion.save(file_name)
    


