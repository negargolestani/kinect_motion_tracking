from utils import*


################################################################################################################################################
if __name__ == '__main__':


    file_name = 'record_04_green'    
    motion = MOTION(file_name)
    

    coil_center = np.mean( motion.locations, axis=1)
    coil_align = np.diff( motion.locations, axis=1).reshape(-1,3)
    coil_align = coil_align / ( np.linalg.norm(coil_align) + 1e-12)
    print(coil_align)


    # for frame_markers in markers:
        # dist = sqrt(sum((self.center - coil.center)**2))
        # ang_misalign_deg = acos(np.dot(self.norm, coil.norm))*180/pi
        # return dist, ang_misalign_deg


