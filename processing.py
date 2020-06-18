from utils import*


################################################################################################################################################
if __name__ == '__main__':


    file_name = 'record_04'    
    motion = MOTION(file_name)
    

    print(np.shape(motion.locations))

    # center = np.mean(frame_marker, axis=0)
    # align = np.diff(frame_marker, axis=0)
    # align = align / ( np.linalg.norm(align) + 1e-12)
    # print(align)


    # for frame_markers in markers:
        # dist = sqrt(sum((self.center - coil.center)**2))
        # ang_misalign_deg = acos(np.dot(self.norm, coil.norm))*180/pi
        # return dist, ang_misalign_deg


