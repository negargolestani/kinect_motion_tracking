from utils import*


################################################################################################################################################
if __name__ == '__main__':

    color = 'green'
    n_markers = 2
    color_range_filename='color_ranges_default'
    target = TARGET(color, n_markers=n_markers, color_range_filename=color_range_filename)
    

    file_name = 'record_02'
    frames = load_record(file_name)
    # times, markers, _ = target.tarck(frames)

    # print(np.shape(markers))
    # save_markers(times, markers, file_name)
    times, markers = load_markers(file_name)


