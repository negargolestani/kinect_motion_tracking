from utils import*



################################################################################################################################################
if __name__ == '__main__':

    calib_color_range_filename = 'color_ranges_default'
    color_ranges = load_pickle( calibsest_folder_path + '/' + calib_color_range_filename )

    color = 'green'
    color_range = color_ranges[color]

    record_filename = 'record_01'
    recorded_frames = load_record(file_path = records_folder_path + '/' + record_filename )

    n_markers = 2
    n_time = len(recorded_frames)
    data_save = np.zeros((n_time,n_markers))
     
    for frame in recorded_frames[:1]:
        markers_camera_space, contours = frame.get_markers(color_range, n_markers=n_markers)
        frame.show(contours=contours, wait=0)
        print( markers_camera_space )