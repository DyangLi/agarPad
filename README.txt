    agarPad calculates cellular size information and fluorescence information from single cells imaged on agar pads or slides. 

    This script is called with the path to the image files with the first
    argument, and then three other arguments that filter cells out based on
    their size.

    > python measure_cell_intensity.py 'path/to/TIFFs/' pc_int_thr wth_cut_up wth_cut_low
    e.g.
    > python measure_cell_intensity.py './TIFF/' 0.75 1.4 0.6

    It saves two files, one with size information and one with
    fluorescent intensity information.

    INPUTS
    path	 path to the directory where the .tif files are
    pc_int_thr  threshold from 0 to 1 to decide where cell boundary is during
                phase contrast sizing
    wth_cut_up  upper bound of acceptable cell width in microns
    wth_cut_low lower bound of acceptable cell width in microns

    OUTPUTS
    path/pathname_pc.txt        size information per cell in three columns:
                                length, width, volume
    path/pathname_fluor.txt     fluorescence intensity information per cell in
                                three columns: average intensity, total
                                intensity, and background intensity
