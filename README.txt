agarPad calculates cellular size information and fluorescence information
    from single cells imaged on agar pads or slides. Size information comes
    from brightfield/phase contrast (pc) images, while fluorecence comes from a
    second channel. A brightfield/pc image is always required to find initial
    cells, but fluorecence imaging is optional. Calculating size information
    based on the pc profile is also opitional, in which case simply the box
    size around the cell will be used to calculate length and width.
    This script is called with reference to a parameters file which holds both
    the path to the image files as well as other parameters and switches.
    > python agarPad.py -f './agarPad_parameters.yaml'
    It saves two files, one with size information and one with
    fluorescent intensity information.
    INPUTS
    parameters.yaml     Holds all parameter and path information. Description of
                        the parameters are in that file.
    OUTPUTS
    path/pathname_pc.txt        size information per cell in three columns:
                                length, width, volume
    path/pathname_fluor.txt     fluorescence intensity information per cell in
                                three columns: average intensity, total
                                intensity, and background intensity
    Additionally there are some hard coded parameters.
