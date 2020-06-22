from utils import*




################################################################################################################################################
if __name__ == '__main__':


    n_markers = 3
    marker_radius = 10
    color_setting_filename = 'color_setting_default'
    color = 'blue'
    markerset = MARKERSET( color, n_markers, marker_radius, color_setting_filename)
    
    file_name = 'record_01'
    record = load_record( file_name )
    

    markers_motion, times = markerset.track(record, show=True)

    save_times(times, file_name)
    save_markers(markers_motion, file_name + '_' + color)
    
