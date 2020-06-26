from utils import*




################################################################################################################################################
if __name__ == '__main__':


    file_name = 'record_03'
    record = load_record( file_name )

    # for frame in record: frame.show(wait=10)

    for color in ['blue', 'red']:
        markerset = MARKERSET( color=color, n_markers=3, marker_radius=10, color_setting_filename='color_setting_default')
        markers_motion, times = markerset.track(record, show=True)
        save_motion(markers_motion, file_name + '_' + color)
        
    save_times(times, file_name)
    
