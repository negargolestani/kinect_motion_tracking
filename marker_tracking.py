from utils import*




################################################################################################################################################
if __name__ == '__main__':


    n_markers = 2
    marker_radius = 15
    color_setting_filename = 'color_setting_default'
    color = 'green'
    markerset = MARKERSET( color, n_markers, marker_radius, color_setting_filename)
    
    
    file_name = 'record_04'
    record = load_record( file_name )
    
    markers_motion, times = markerset.track(record, show=True)

    save_times(times, file_name)
    save_markers(markers_motion, file_name + '_' + color)
    
