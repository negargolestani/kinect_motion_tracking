from utils import* 

if __name__ == '__main__':
    
    # dataset_name ='arduino_orthogonal'
    file_name = 'record_19'
    record_time = 40


    # Initialization
    kinect = KINECT()    

    time_file_path = get_time_file_path(dataset_name, file_name)
    create_folder(time_file_path)
    time_df = pd.DataFrame(columns=['time'])

    color_video_file_path = get_color_video_file_path(dataset_name, file_name)
    create_folder(color_video_file_path)
    color_vid = cv2.VideoWriter(color_video_file_path, cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (kinect.kinect.color_frame_desc.Width, kinect.kinect.color_frame_desc.Height))

    camera_space_file_path = get_camera_space_file_path(dataset_name, file_name) 
    create_folder(camera_space_file_path)
    camera_space_list = list()


    # Show when kinect starts recording
    kinect.read(full=False)
    print('Recording is Started \n  Press "Esc" Key to Stop Recording')
    time_lib.sleep(2)

    # Recording loop
    start_time = datetime.combine(date.min, kinect.time)
    while cv2.waitKey(1)!=27 and (datetime.combine(date.min, kinect.time) - start_time).total_seconds() < record_time:   
        kinect.read(full=True)
        kinect.show()
        time_df = time_df.append({'time':kinect.time}, ignore_index=True)
        color_vid.write( cv2.cvtColor(kinect.color_image, cv2.COLOR_RGBA2RGB) )            
        camera_space_list.append(kinect.camera_space)
    cv2.destroyAllWindows()    

    print('Recording is Finished \n Wait for Saving ...')    
    time_df.to_csv(time_file_path, index=False)        
    with open(camera_space_file_path,"wb") as f: pickle.dump(camera_space_list, f)            
   
    print('Done!')