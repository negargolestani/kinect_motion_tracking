from utils import* 


################################################################################################################################################
if __name__ == '__main__':

    file_name = 'record_03'

    kinect = KINECT( top_margin=0.2, bottom_margin=0.2, left_margin=0.2, right_margin=0.2)
    # record = kinect.record(full_data=False)
    record = kinect.record()
    save_record(record, file_name)

################################################################################################################################################
        