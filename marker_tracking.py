from utils import*


if __name__ == '__main__':

    dataset_name = 'arduino_parallel_new'
    colors = ['red','blue'] 
    n_circles = [4,3]


    for n in range(20):
        file_name = 'record_' + "{0:0=2d}".format(n)

        # Record
        record = RECORD( dataset_name, file_name, color_setting_filename='color_setting_default')
        locations_df = pd.DataFrame(columns=['time', *colors])
        locations_dict = defaultdict(list)  

        # Loop over frames 
        while True:
            success = record.read()
            if not success: break

            for i, color in enumerate(colors):
                circles = record.get_colored_circles(color, n_circles=n_circles[i])
                record.draw_contours(circles, color=30*i) 
                locations = record.get_locations(circles)  
                locations_dict[color].append(locations)                            
            record.show(wait=1)

        # Time
        time_file_path = get_time_file_path(dataset_name, file_name)    
        time = pd.read_csv(time_file_path)

        # MArkers (time and locations)
        markers = pd.concat([time, pd.DataFrame(locations_dict)], axis=1)
        
        # Save 
        markers_file_path = get_markers_file_path(dataset_name, file_name)
        create_folder(markers_file_path)
        markers.to_csv(markers_file_path, index=False)
        
        print(file_name, 'is saved.')   
    

