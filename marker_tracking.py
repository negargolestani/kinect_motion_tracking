from utils import*




################################################################################################################################################
if __name__ == '__main__':

    # Initialization
    file_name = 'record_03'
    colors = ['blue', 'red']

    # Load
    record = load_record( file_name )
    color_setting = load_color_setting()    

    # Outputs dict
    motions = dict()
    for color in colors: motions.update({color:list()})

    # Tracking loop 
    for frame in record[:-2]:
        contours = list()

        for color in colors:
            frame_circles = frame.get_colored_circle( color_setting[color], n_circles=3, radius=10)
            locations = frame.get_location(frame_circles)

            contours = [*contours, *frame_circles]            
            motions[color].append( locations )

        frame.show(contours=contours, wait=1)

    
    # Save motions
    # for color in colors: save_motion(file_name + '_' + color,  motions[color])