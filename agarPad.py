"""
Created on Tue Jan 13 11:06:17 2015
Edited on Fri Sep 18 by jt
Edited on Fri Sep 29 by jt -- added pc size analysis
Edited on Thr Oct 15 by jt -- added parameter file support
@author: yonggun
"""

# import modules
#from __future__ import division
import os, sys, getopt, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import ndimage
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage import measure
from skimage import exposure
import cv2 # openCV
from termcolor import colored
from os import listdir
from os.path import isfile, join
import glob # for path name finding
import yaml # for parameter loading

# if this is being done in ipython
try:
    from IPython.core.display import display, clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

################################################################################
### functions
def find_threshold(profile, thr):
    '''This travels along a profile and finds the point at which a certain threshold is met. It then uses a linear fit to find where the threhold was met with sub pixel resolution.
    INPUTS
    profile     linear intensity profile
    thr         threshold from 0 to 1 from which to find a point
    OUTPUTS
    thr_pt      point along the profile that the threshold is met
    '''
    thr_pt = 0 # initialize position of where threhold is met
    n = 0 # holds position in profile array

    # loop over intensity profile
    for i in (profile):
        if i > thr: # look for first pixel value over threshold
            ax1 = n-1 # pixel array position before threshold met
            ay1 = profile[n-1] # intesity value at that position
            ax2 = n # pixxel array position after threshold met
            ay2 = i # intesity value at that threshold
            k = (ay2-ay1)/(ax2-ax1) # calculate slope
            # the slope should be postive and don't choose the first few points
            if k < 0 or n < 3:
                continue
            b = ay2 - k*ax2 # find intercept
            try:
                thr_pt = (thr - b) / k # find exact point thr met
            except:
                print('Threshold on profile not found.')
                continue
            break # break if this is found
        n += 1 # update position
    return thr_pt

def CellBox(img, imgfluo, fpath):
    '''This function finds boxes around the cells and calculates sizes and
    intensities.
    INPUTS
    img         phase contrast or bright field image
    imgfluo     fluorescent channel image
    fpath       file path of img
    pc_int_thr  threshold from 0 to 1 to decide where cell boundary is during
                phase contrast sizing
    wth_cut_up  upper bound of acceptable cell width in microns
    wth_cut_low lower bound of acceptable cell width in microns
    meas_pc     boolean for if sizes should be measured from phase image
    meas_int    boolean for if average and total fluorecence intensity should
                be measured
    params['qc']          for quality control, aka verbose mode
    OUTPUTS
    '''
    # define some variables from the params dictionary
    pxl2um = params['pxl2um']

    ############################################################################
    # find all the cells using contouring
    kernel = np.ones((3,3), np.uint8)
    img_preprocess = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img8 = img_as_ubyte(img_preprocess)

    # process the imaging for contour finding
    blur = cv2.GaussianBlur(img8,(5,5),0)
    th3 = cv2.adaptiveThreshold(blur, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,121,2)
    ret2,global_otsu_inv = cv2.threshold(th3, 128, 255,
                                         cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(global_otsu_inv, cv2.MORPH_OPEN, kernel)
    closing = cv2.erode(opening, kernel, iterations = 1)
    closing = cv2.dilate(closing, kernel, iterations = 1)
    img_inv = closing

    # find the contours of all the cells or possible cells.
    # CHAIN_APPROX_SIMPLE saves the whole contour information in a compressed
    # way, saving memory.
    contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE, (0,0))

    if params['qc']:
        print('Number of found contours:', len(contours))

    ############################################################################
    # filter contours for location, strange shapes, and sizes

    # these numpy arrays hold the output data for the remaining cells
    wth_box = [] # crude width of cell box
    hgt_box = [] # crude height of cell box

    # get rid of cells that are in the boundary of the image
    im_height, im_width = img.shape
    # boundary = 20 this variable is globals from the .yaml
    for h, cnt in reversed(list(enumerate(contours))):
        for tmp in cnt:
            if (tmp[0][1] < params['boundary'] or tmp[0][1] > im_height - params['boundary'] or tmp[0][0] < params['boundary'] or tmp[0][0] > im_width - params['boundary']):
                del contours[h]
                break

    # get rid of cells based on shape and size
    for h, cnt in reversed(list(enumerate(contours))):
        # find shape around cell
        rect = cv2.minAreaRect(cnt) # create rectangle around cell
        box = cv2.cv.BoxPoints(rect) # turn that rect object into 4 points
        box = np.int0(box)
        area = cv2.contourArea(cnt) # area of the cell based on contour

        # determine the angle the cell is pointed as a rough estimate of
        # length and width based on the bounding box
        if rect[1][0] > rect[1][1]:
            length = np.int0(rect[1][0])
            width = np.int0(rect[1][1])
            angle = rect[2]
        else:
            length = np.int0(rect[1][1])
            width = np.int0(rect[1][0])
            angle = rect[2] + 90

        ### filter contours that are not cell
        # if the area is too small then delete that cell
        if area < params['min_area']:
            del contours[h]
            continue

        # if the cell has a crazy shape delete it
        rectangle = length * width
        if rectangle / area > params['rect_to_area']:
            del contours[h]
            continue

        # delete cells with weird aspect ratio
        if length / width < params['min_aspect_ratio'] or length / width > params['max_aspect_ratio']:
           del contours[h]
           continue

        # excluding short and long cells
        if length > params['con_max_l'] or length < params['con_min_l']:
            del contours[h]
            continue

        # excluding based on width limits
        if width > params['con_max_w'] or width < params['con_min_w']:
            del contours[h]
            continue

        # record height and width of cell if it made the cut
        hgt_box = np.append(hgt_box, length)
        wth_box = np.append(wth_box, width)

    ### end loop finding throwing out contours

    # print number of filtered contours per image
    if params['qc']:
        print('Number of found cells after initial filtering:', len(contours))

    ############################################################################
    ### Go through all the contours (the found cells), and calculate values

    ### phase contrast analysis
    if params['meas_pc']:

        # define some variables from the global parameter dictionary
        pc_int_thr = params['pc_int_thr']

        wth_pc = [] # width of cell pc
        hgt_pc = [] # height of cell pc

        for h, cnt in reversed(list(enumerate(contours))):
            # reset image mask
            img_mask = np.zeros(img.shape,np.uint8)

            # find cell box
            rect = cv2.minAreaRect(cnt)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)

            # Check the size and angle
            if rect[1][0]>rect[1][1]:
                length=np.int0(rect[1][0])
                width=np.int0(rect[1][1])
                angle=rect[2]
            else:
                length=np.int0(rect[1][1])
                width=np.int0(rect[1][0])
                angle=rect[2]+270

            # make a larger box around the contour
            sz_img = length + 30
            sz_lth = length + 25
            sz_wth = width + 10

            # crop out a rectangle around the cell
            s_piece_pc = img[np.int0(rect[0][1]-sz_img/2):np.int0(rect[0][1]+sz_img/2), np.int0(rect[0][0]-sz_img/2):np.int0(rect[0][0]+sz_img/2)]
            # rotate image so cell is horizontal
            b_pc = ndimage.rotate(s_piece_pc, angle)

            ####################################################################
            ### determine the length of cell

            # find center line of cell.
            sz_imgR = b_pc.shape[0]
            d_pc = b_pc[sz_imgR/2-3:sz_imgR/2+3, sz_imgR/2-sz_lth/2:sz_imgR/2+sz_lth/2]
            kkk_pc = np.mean(d_pc,0) #profile of 6px

            # make sure you actually got something here
            len_kkk = len(kkk_pc)
            if (len_kkk)==0:
                del contours[h]
                continue

            # x position in profile
            xxx = np.arange(0, len_kkk)

            # find min and max intesity values
            mx_pc = max(kkk_pc)
            mn_pc = min(kkk_pc)

            # try to do bg subtraction and normalization
            try:
                kkk_pc_r=1-(kkk_pc-mn_pc)/(mx_pc-mn_pc)
            except:
                del contours[h]
                continue

            # find mean max of center of length
            mx_avg = np.mean(kkk_pc_r[len_kkk/2-10:len_kkk/2+11])
            mx_avg = mx_avg*pc_int_thr
            # line with value of that avg
            avg = np.zeros((len_kkk,1))
            avg[:] = mx_avg

            ### Determine the length based on PC image
            # left side threshold point
            l_th = find_threshold(kkk_pc_r, pc_int_thr)

            # right side threshold point
            # flip the profile to look from the other side
            kkk_pc_r_reversed = kkk_pc_r[::-1]
            r_th = find_threshold(kkk_pc_r_reversed, pc_int_thr)
            # the point must be subtracted from other side
            r_th = len_kkk-r_th-1

            # filter out cells if the length makes no sense
            if np.isnan(r_th) or np.isnan(l_th) or np.isnan(r_th-l_th) or r_th==0 or l_th==0:
                del contours[h]
                continue
            else:
                ll = r_th - l_th # length

            # save variables for plotting later
            if params['qc']:
                xxx_l = xxx
                kkk_pc_r_l = kkk_pc_r
                avg_l = avg
                l_th_l = l_th
                mx_avg_l = mx_avg
                r_th_l = r_th

            ####################################################################
            ### determine the width of cell
            wd = width-2

            # split width into two parts to avoid contricted centers
            try:
                ll_b=int(l_th+wd/2)
                lr_b=int((l_th+r_th)/2-wd/2)
                rl_b=int((l_th+r_th)/2+wd/2)
                rr_b=int(r_th-wd/2)
            except:
                del contours[h]
                continue

            f_pc = b_pc[sz_imgR/2-sz_wth/2-3:sz_imgR/2+sz_wth/2+5, sz_imgR/2-sz_lth/2:sz_imgR/2+sz_lth/2]

            r,c = f_pc.shape
            if r <= 0 or c <= 0:
                del contours[h]
                continue

            l_img=f_pc[0:r, ll_b:lr_b]
            r_img=f_pc[0:r, rl_b:rr_b]
            if False:
                plt.imshow(l_img, cmap=plt.cm.gray, interpolation='nearest')
                plt.show()

            l_th_mean = 0
            r_th_mean = 0
            kkk_all = np.zeros(r)

            # Define the range to average the width
            img_piece=f_pc[0:r, ll_b:rr_b]
            rr, cc = img_piece.shape
            llength = ll_b
            rlength = rr_b

            if False:
                plt.imshow(img_piece, cmap=plt.cm.gray, interpolation='nearest')
                plt.show()

            # Calculate the mean width across all middle sections from above
            for pxl in range(cc):
                try:
                    kkk_pc_w=img_piece[0:r, pxl]
                except:
                    del contours[h]
                    continue
                #kkk_pc_w=np.mean(s_pc,1)
                len_kkk=len(kkk_pc_w)
                if (len_kkk)==0:
                    del contours[h]
                    continue

                xxx_w = np.arange(0, len_kkk)

                mx_pc = max(kkk_pc_w)
                mn_pc = min(kkk_pc_w)

                # normalize data
                try:
                    kkk_pc_r = 1 - (kkk_pc_w-mn_pc) / (mx_pc-mn_pc)
                except:
                    del contours[h]
                    continue
                #mx_avg=np.mean(kkk_pc_r[len_kkk/2-1:len_kkk/2+2])
                mx_avg=np.max(kkk_pc_r[len_kkk/2-1:len_kkk/2+2])
                avg=np.zeros((len_kkk,1))
                mx_avg=mx_avg*pc_int_thr
                avg[:]=pc_int_thr

                if False:
                    plt.plot(xxx_w, kkk_pc_r, 'b-', xxx_w, avg, 'g-')
                    plt.show()

                # Determine the width based on threshold
                # left side threshold point
                l_th = find_threshold(kkk_pc_r, pc_int_thr)

                # right side threshold point
                # flip the profile to look from the other side
                kkk_pc_r_reversed = kkk_pc_r[::-1]
                r_th = find_threshold(kkk_pc_r_reversed, pc_int_thr)
                # the point must be subtracted from other side
                r_th = len_kkk-r_th-1

                # update average on fly
                l_th_mean = (l_th_mean*pxl + l_th) / (pxl+1)
                r_th_mean = (r_th_mean*pxl + r_th) / (pxl+1)

            # calculate width as distance between means
            ww = r_th_mean - l_th_mean

            # filtering data once again
            if np.isnan(l_th_mean) or np.isnan(r_th_mean) or np.isnan(r_th_mean-l_th_mean) or l_th_mean==0 or r_th_mean==0:
                del contours[h]
                continue
            else:
                ww=r_th_mean-l_th_mean

                if np.isnan(ll) or np.isnan(ww):
                    del contours[h]
                    continue
                elif ll <= 1 or ww <= params['wth_cut_low']/pxl2um:
                    del contours[h]
                    continue
                elif ll > 150 or ww > params['wth_cut_up']/pxl2um:
                    del contours[h]
                    continue
                else:
                    hgt_pc=np.append(hgt_pc, ll)
                    wth_pc=np.append(wth_pc, ww)

            # plot the cell and associated profiles
            if params['qc']:
                # draw contour on image mask for plotting
                cv2.drawContours(img_mask, [cnt], -1, 100, 3)

                plt.figure(figsize=(10,10), dpi=80)

                plt.subplot(2,2,1)
                plt.title('Cell (or should be)')
                plt.xlabel('pixels')
                plt.ylabel('pixels')
                plt.imshow(b_pc, cmap=plt.cm.gray, interpolation='nearest')

                plt.subplot(2,2,3)
                plt.title('Length profile intensity')
                plt.plot(xxx_l, kkk_pc_r_l, 'b-', xxx_l, avg_l, 'g-', l_th_l, mx_avg_l, 'bo', r_th_l, mx_avg_l, 'bo')
                plt.xlabel('pixels')
                plt.ylabel('normalized intensity')

                plt.subplot(2,2,2)
                plt.title('Width profile intensity')
                plt.plot(xxx_w, kkk_pc_r, 'b-', xxx_w, avg, 'g-', l_th, pc_int_thr, 'bo', r_th, pc_int_thr, 'bo')
                plt.xlabel('pixels')
                plt.ylabel('normalized intensity')

                plt.subplot(2,2,4)
                plt.title('Location of cell in image')
                plt.imshow(img_mask, cmap=plt.cm.gray, interpolation='nearest')
                plt.xlabel('pixels')
                plt.ylabel('pixels')
                plt.show()

        ### end of looping through contours
    ### end of phase contrast sizing

    ############################################################################
    # measure fluorescence intensities
    # this goes after pc analysis, because pc analysis will throw out cells
    if params['meas_int']:
        avg_int_cells = [] # numpy array with average intensity per cell
        tot_int_cells = [] # numpy array with total intensity per cell

        # find background intensities of the whole image
        img_mask = np.zeros(img.shape, np.uint8)
        background_mask = np.zeros(img.shape, np.uint8)
        background_mask = 255 - background_mask
        mean_background = cv2.mean(imgfluo, mask = background_mask)

        for h, cnt in reversed(list(enumerate(contours))):
            # find intensity of the cell
            int_mask = np.zeros(img.shape, np.uint8)

            # contour is a demarcation of the cell
            # this changes the mask to just the cell of interest, [cnt]
            cv2.drawContours(int_mask,[cnt],0,255,-1)
            cv2.drawContours(background_mask,[cnt],0,255,-1)

            # average intensity
            avg_int = cv2.mean(imgfluo, mask = int_mask)

            # total intensity
            area = cv2.contourArea(cnt) # area of the cell based on contour
            tot_int = avg_int[0] * area # is just avgerage times area

            # append data
            avg_int_cells = np.append(avg_int_cells, avg_int[0])
            tot_int_cells = np.append(tot_int_cells, tot_int)

            # draw contour with current contour, adding it to img_mask
            cv2.drawContours(img_mask,[cnt],-1,100,2)

    # display contours for whole picture
    if params['qc']:
        plt.figure(figsize=(16,16), dpi=80)
        plt.title(fpath)
        plt.imshow(img_mask, cmap=plt.cm.gray, interpolation='nearest')
        plt.show()

    ############################################################################
    ### return data based on what analysis was done
    if params['meas_pc'] and not params['meas_int']:
        return hgt_pc, wth_pc, 0, 0, 0, img_mask
    elif not params['meas_pc'] and params['meas_int']:
        return hgt_box, wth_box, avg_int_cells, tot_int_cells, mean_background[0], img_mask
    elif params['meas_pc'] and params['meas_int']:
        return hgt_pc, wth_pc, avg_int_cells, tot_int_cells, mean_background[0], img_mask

################################################################################
### main script
if __name__ == "__main__":
    """Main program.
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
    """

    # get arguments and parameters
    # switches
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:")
    except getopt.GetoptError:
        print('No arguments detected.')
    for opt, arg in opts:
        if opt == '-f':
            param_file = arg

    # load the parameter file
    f = open(param_file)
    global params
    params = yaml.safe_load(f)
    f.close()

    print(params)

    path = params['path_to_tiffs'] # path to image folder
    BFfiles = glob.glob(path+"/*c1.tif") # phase or bright field images
    Fluofiles = glob.glob(path+"/*c2.tif") # fluorescent images


    # # used to save the profiles
    # profile_path = path+"/profile"
    # if not(os.path.isdir(profile_path)):
    #     os.makedirs(profile_path)

    # in case the files were made using NIS elements viewer
    if len(BFfiles) == 0:
        BFfiles = glob.glob(path+"/*C1.tif")
        Fluofiles = glob.glob(path+"/*C2.tif")

    # get rid of the .DS_Store file if it exists
    if BFfiles[0]=='.DS_Store':
        del BFfiles[0]
    if Fluofiles[0]=='.DS_Store':
        del Fluofiles[0]

    print colored("========================================================", 'red')
    print "     ", colored(params['experiment_name'], 'green')
    print('Found %d files:' % len(BFfiles))
    print colored("========================================================\n", 'red')

    # user defined thesholds and whatnot
    pxl2um = params['pxl2um'] # pixel to um conversion... seems crude
    #pc_int_thr = params['pc_int_thr'] # threshold for profile
    #wth_cut_up = params['wth_cut_up'] #
    #wth_cut_low = params['wth_cut_low']

    # these array hold all data for all pictures
    a_wth=[]
    a_hgt=[]

    avg_intensity = []
    tot_intensity = []
    background = []
    #relative_intensity = []

    # loop through all pictures, based on phase images
    for i, bfname in enumerate(BFfiles):
        fluoname = Fluofiles[i] # get the fluorecent image name

        print bfname
        #
        # progress = i / len(BFfiles) * 100
        # #sys.stdout.write(bfname)
        # if have_ipython:
        #     clear_output(wait=True)
        #     print('Progress: %3.0f%% \r' % progress)
        # else:
        #     sys.stdout.write('Progress: %3.0f%% \r' % progress)
        # sys.stdout.flush()

        # read images
        img = plt.imread(bfname) # read phase image
        img_fluo = plt.imread(fluoname) # read fuorescen image

        # crop image, both images should be the same size
        if params['crop_image']:
            rr, cc = img.shape
            # rr, cc = img.shape
            # img = img[.25*rr:.75*rr, .25*cc:.75*cc]
            # rr, cc = img.shape
            # img_fluo = img_fluo[0:rr, 0:cc]
            img = img[params['row_start']*rr:params['row_end']*rr,
             params['col_start']*cc:params['col_end']*cc]
            img_fluo = img_fluo[params['row_start']*rr:params['row_end']*rr,
             params['col_start']*cc:params['col_end']*cc]

        # equalize
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.005)

        ### Run analysis
        hgt_pc, wth_pc, avg_int_cells, tot_int_cells, mean_bg, contour = CellBox(img_adapteq, img_fluo, bfname)

        # append data phase
        #print '===> Number of cells found is', len(hgt_pc), '<==='
        if len(hgt_pc)==0:
            continue

        # print number of cells
        print '===> Number of cells found is', len(hgt_pc), '<==='

        a_hgt = np.append(a_hgt, hgt_pc * pxl2um)
        a_wth = np.append(a_wth, wth_pc * pxl2um)

        # append data fluo
        avg_intensity = np.append(avg_intensity, avg_int_cells)
        tot_intensity = np.append(tot_intensity, tot_int_cells)
        background = np.append(background, mean_bg)

    # calculate volume of cylinder with hemispherical ends
    a_vol = (a_hgt-a_wth)*np.pi*(a_wth/2)**2+(4/3)*np.pi*(a_wth/2)**3

    # calculate average values for printing
    l = np.mean(a_hgt)
    w = np.mean(a_wth)
    v = np.mean(a_vol)

    # print info terminal
    print "Height"
    print a_hgt
    print "Width"
    print a_wth

    text = 'length = ' + str(l)[:6] + 'um, width = ' + str(w)[:6] + 'um, volume = ' + str(v)[:6] + 'um^3'
    print colored("========================================================", 'red')
    print "     ", colored(params['experiment_name'], 'green')
    print colored("========================================================", 'red')
    print colored(text,'yellow')
    print colored("========================================================", 'red')

    # print file
    print('Writing file')
    array=np.zeros([len(a_hgt),3])
    array[:,0] = a_hgt
    array[:,1] = a_wth
    array[:,2] = a_vol

    fname = params['path_analysis'] + params['experiment_name'] + '_pc.txt'
    np.savetxt(fname, array, delimiter=',')

    # shape data into array
    array = np.zeros([len(avg_intensity),3])
    array[:,0] = avg_intensity
    array[:,1] = tot_intensity
    array[:,2] = np.mean(background)
    #array[:,2] = a_relative_intensity

    # save file as csv
    fname = params['path_analysis'] + params['experiment_name'] + '_fluor.txt'
    np.savetxt(fname, array, delimiter=',')

    # additional print files
    if False:
        savefile = bfname + '_contour.tif'
        #contour = rgb2gray(contour)
        plt.imsave(savefile, contour, cmap=plt.cm.gray)

    print('Finished')
