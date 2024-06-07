import argparse
import copy
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import skimage
import tifffile as tif
import torch
from pytictoc import TicToc
from scipy import ndimage
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image


class SAM_cyst_workflow:
    """
    Class to completely process a histology image and obtain a segmentation mask of the cysts,
    together with an estimate mask of the whole kidney. All of this using a SAM-based workflow.
    """

    def __init__(self, imName = False, outputFolder = False, fullKidneyOutputFolder =False, args=False, checkpointPath = "sam_vit_h_4b8939.pth", winProportion=5, largeThresProportion=20,
                 model_type='vit_h', SAM_points_per_side=32, patchBorderScanRange=6, maxNumKidneyPieces=3,
                 propBorderForStraingthObjects=0.7, bboxSizeThres=100, save=True):
        '''
        Loading all the global variables from the inputed arguments
        '''

        warnings.filterwarnings('ignore')
        #If the arguments are not passed by argparser, the independent input values are used instead

        self.imName = args.imName if (args and (args.imName is not None)) else imName
        self.outputFolder = args.outputFolder if (args and (args.outputFolder is not None)) else outputFolder
        self.fullKidneyOutputFolder = args.fullKidneyOutputFolder if (args and (args.fullKidneyOutputFolder is not None)) else fullKidneyOutputFolder
        if not self.fullKidneyOutputFolder:
            self.fullKidneyOutputFolder = self.outputFolder

        self.checkpointPath = args.checkpointPath if (args and (args.checkpointPath is not None)) else checkpointPath
        self.winProportion = args.winProportion if (args and (args.winProportion is not None)) else winProportion
        self.largeThresProportion = args.largeThresProportion if (args is not None and (args.largeThresProportion is not None)) else largeThresProportion
        self.model_type = args.model_type if (args and (args.model_type is not None)) else model_type
        self.SAM_points_per_side = args.SAM_points_per_side if (args and (args.SAM_points_per_side is not None)) else SAM_points_per_side
        self.patchBorderScanRange = args.patchBorderScanRange if (args and (args.patchBorderScanRange is not None)) else patchBorderScanRange
        self.maxNumKidneyPieces = args.maxNumKidneyPieces if (args and (args.maxNumKidneyPieces is not None)) else maxNumKidneyPieces
        self.propBorderForStraingthObjects = args.propBorderForStraingthObjects if (args is not None and (args.propBorderForStraingthObjects is not None))\
                                            else propBorderForStraingthObjects
        self.bboxSizeThres = args.bboxSizeThres if (args and (args.bboxSizeThres is not None)) else bboxSizeThres
        self.save = args.save if (args and (args.save is not None)) else save

        
        

    def load_im(self, imName):
        '''
        Load an image as a numpy array from a tiff file name.
        '''
        im = tif.imread(imName)
        return im

    def update_mask_labels(self, anns, initInt, largeThres):
        """
        It updates the mask labels from different patches of the images so that they are all unique.
        """
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)

        img = np.zeros(
            (
                sorted_anns[0]["segmentation"].shape[0],
                sorted_anns[0]["segmentation"].shape[1],
            ),
            dtype=int,
        )

        count = 0

        for ann in sorted_anns:
            m = ann["segmentation"]
            if np.sum(m) >= largeThres:
                img[m] = 0
            else:
                count += 1
                img[m] = initInt + count

        return img

    def applySAMHistoSeg(self, im):
        """
        Applying pretrained SAM to the original image
        """
        t = TicToc()
        t.tic()

        winProportion = self.winProportion
        largeThresProportion = self.largeThresProportion
        model_type = self.model_type
        checkpointPath = self.checkpointPath
        SAM_points_per_side = self.SAM_points_per_side

        # Use the GPU
        if torch.cuda.is_available():
            deviceUsed = torch.device("cuda:0")
            print("GPU available")
        else:
            deviceUsed = torch.device("cpu")
            print("GPU unavailable, running on CPU")

        sam = sam_model_registry[model_type](checkpointPath).to(deviceUsed)
        mask_generator = SamAutomaticMaskGenerator(sam, SAM_points_per_side)

        # Scan the image window by window
        # Window size, obtained from dividing the image's width over winProportion (removing decimals)
        winSize = np.floor(im.shape[1] / winProportion).astype(int)

        # When the image is fully scanned this variable will be set to true
        scanned = False

        # These 2 variables check which column and row you are in (in the image divided by windows)
        dim0 = 0
        dim1 = 0
        imMask = np.zeros([im.shape[0], im.shape[1]], dtype="int")

        # Initial intensity for the current patch mask
        initInt = 0

        # Threshod to remove too large objects. Calculated as the number of pixels in the window
        # divided by largeThresProportion
        largeThres = np.floor((winSize**2) / largeThresProportion).astype(int)

        xDimEnd = False
        yDimEnd = False

        # Scanning the image, dividing it into windows and applying SAM
        # to each of them
        while not scanned:
            windowStart0 = dim0 * winSize
            windowStart1 = dim1 * winSize

            # If the window fits inside the image (horizonal dimension)
            if im.shape[1] >= (windowStart1 + 2 * winSize):
                winFinalSize1 = winSize - 1
                dim1 += 1
                xDimEnd = False
            # Otherwise
            else:
                winFinalSize1 = im.shape[1] - windowStart1 - 1
                dim1 = 0
                dim0 += 1
                xDimEnd = True

            # If the window fits inside the image (vertical dimension)
            if im.shape[0] >= (windowStart0 + 2 * winSize):
                winFinalSize0 = winSize - 1
                yDimEnd = False
            # Otherwise
            else:
                winFinalSize0 = im.shape[0] - windowStart0 - 1
                yDimEnd = True

            # Extract the patch
            patch = im[
                windowStart0 : windowStart0 + winFinalSize0,
                windowStart1 : windowStart1 + winFinalSize1,
                :,
            ]

            # Process it with SAM and update the mask labels to ensure they are unique
            patchData = mask_generator.generate(patch)
            patchMask = self.update_mask_labels(patchData, initInt, largeThres)

            initInt = np.max(patchMask)

            # Store the patch's mask in the complete output mask
            imMask[
                windowStart0 : windowStart0 + winFinalSize0,
                windowStart1 : windowStart1 + winFinalSize1,
            ] = patchMask

            # When all rows have been scanned
            if xDimEnd and yDimEnd:
                scanned = True
        print('SAM application by patches:')
        t.toc()
        print('\n')
        

        return imMask

    def correctPatchBorders(self, mask):
        """
        Scan each of the patches to correct their borders. Pixels mistakenly labeled as background in
        the scan range of the patch's border will be corrected to the intensity of the closest non-background
        label. Objects that have been split between patches will be combined back.
        """
        t = TicToc()
        t.tic()

        winProportion = self.winProportion
        scanRange = self.patchBorderScanRange
        # Define window size
        wS = np.floor(mask.shape[1] / winProportion).astype(int)

        # Define maximum dimensions (maximum number of patches in each direction)
        maxdim0 = np.ceil(mask.shape[0] / wS).astype(int)
        maxdim1 = np.ceil(mask.shape[1] / wS).astype(int)

        # Scan all the patches (dim0 and dim1 are the patch coordinates)
        for dim0 in range(maxdim0):
            for dim1 in range(maxdim1):
                if not dim0 == maxdim0 - 1:
                    # Regular tile
                    if not dim1 == maxdim1 - 1:
                        in0 = dim0 * wS
                        fin0 = (dim0 + 1) * wS
                        in1 = dim1 * wS
                        fin1 = (dim1 + 1) * wS

                    # If dim1 is over
                    else:
                        in0 = dim0 * wS
                        fin0 = (dim0 + 1) * wS
                        in1 = dim1 * wS
                        fin1 = mask.shape[1]

                else:
                    # If only dim0 is over
                    if not dim1 == maxdim1 - 1:
                        in0 = dim0 * wS
                        fin0 = mask.shape[0]
                        in1 = dim1 * wS
                        fin1 = (dim1 + 1) * wS

                    # If both dim0 and dim1 are over
                    else:
                        in0 = dim0 * wS
                        fin0 = mask.shape[0]
                        in1 = dim1 * wS
                        fin1 = mask.shape[1]

                currPatch = mask[in0:fin0, in1:fin1]

                # If there is a patch below
                if fin0 < mask.shape[0]:
                    # print('patch below')

                    # Extract the patch below
                    bottPatch = mask[fin0 : fin0 + wS, in1:fin1]

                    # The lower border of your patch

                    # Scan the last rows. For each row whose entries are 0 at the border, give them the same intensity as the previous
                    # row of pixels instead (correct them).
                    for h in reversed(range(scanRange)):
                        # So that you get [3,2,1]
                        ind = h + 1
                        if ind <= (currPatch.shape[0]-1):
                            if np.all(currPatch[-ind, :] == 0):
                                currPatch[-ind, :] = currPatch[-ind - 1, :]

                    bottBorder = currPatch[-1, :]

                    # The upper border of the patch below

                    # Scan the first rows. For all entries that are 0 at the border, give them the same intensity as the next
                    # row of pixels instead (correct them).
                    for h in reversed(range(scanRange)):
                        ind = h
                        if ind <= (bottPatch.shape[0]-2):
                            if np.all(bottPatch[ind, :] == 0):
                                bottPatch[ind, :] = bottPatch[ind + 1, :]

                    topBorderNext = bottPatch[0, :]

                    # For each pixel in the current patch's bottom line,
                    # check if it is not 0, the corresponding pixel in
                    # the lower patch is also not 0 and both are different.
                    # If this is fulfilled, paint all pixels with that
                    # intensity from the bottom patch with the current
                    # patch's intensity

                    for pixel, nextPixel in zip(bottBorder, topBorderNext):
                        if (pixel != 0) and (nextPixel != 0) and (pixel != nextPixel):
                            bottPatch[bottPatch == nextPixel] = pixel

                    # Save the bottom patch in the mask
                    mask[fin0 : fin0 + wS, in1:fin1] = bottPatch

                    # If there is a patch to the right
                if fin1 < mask.shape[1]:
                    # Extract the patch to the right
                    rightPatch = mask[in0:fin0, fin1 : fin1 + wS]

                    # The right border of your patch

                    # Scan the last columns. For each column whose entries are 0 at the border, give them the same intensity as the previous
                    # column of pixels instead (correct them).

                    for h in reversed(range(scanRange)):
                        ind = h + 1
                        if ind <= (currPatch.shape[1]-1):
                            if np.all(currPatch[:, -ind] == 0):
                                currPatch[:, -ind] = currPatch[:, -ind - 1]

                    rightBorder = currPatch[:, -1]

                    # The left border of the patch to the right

                    # Scan the first columns. For all entries that are 0 at the border, give them the same intensity as the next
                    # column of pixels instead (correct them).

                    for h in reversed(range(scanRange)):
                        ind = h
                        if ind <= (rightPatch.shape[1]-2):
                            if np.all(rightPatch[:, ind] == 0):
                                rightPatch[:, ind] = rightPatch[:, ind + 1]

                    leftBorderNext = rightPatch[:, 0]

                    # For each pixel in the current patch's right line,
                    # check if it is not 0, the corresponding pixel in
                    # the patch to the right is also not 0 and both are
                    # different. If this is fulfilled, paint all pixels
                    # with that intensity from the bottom patch with the
                    # current patch's intensity

                    for pixel, nextPixel in zip(rightBorder, leftBorderNext):
                        if (pixel != 0) and (nextPixel != 0) and (pixel != nextPixel):
                            rightPatch[rightPatch == nextPixel] = pixel

                    # Save the right patch in the mask
                    mask[in0:fin0, fin1 : fin1 + wS] = rightPatch

                # Save the current patch in the mask
                mask[in0:fin0, in1:fin1] = currPatch

        
        print('Borders of the patches corrected:')
        t.toc()
        print('\n')
        return mask

    # ------------------------------------------------------------------------------------------------------------------

    def getFullKidneyMask(self, im):
        """
        Get an approximate mask of the whole kidney using Otsu's method and the convex hull of the largest
        connected components in the image.
        """
        t = TicToc()
        t.tic()

        maxNumKidneyPieces = self.maxNumKidneyPieces

        # Transform the image from RGB to grayscale
        im = np.mean(im, -1)

        # Threshold the image using Otsu's method
        thresh = skimage.filters.threshold_otsu(im)
        im = 1 * (im <= thresh)

        # Look for the largest connected components (kidney cuts)
        labels = label(im)
        props = regionprops(labels)

        # Filter regions based on major axis length (greater than (1/6)*width) and sort them
        largest_regions = sorted(
            [
                prop
                for prop in props
                if prop.major_axis_length > (im.shape[1] * (1 / 6))
            ],
            key=lambda x: x.area,
            reverse=True,
        )

        if len(largest_regions) > maxNumKidneyPieces:
            largest_regions = largest_regions[:maxNumKidneyPieces]

        # Calculate the convex hull of each kidney cut
        kidneyMask = np.zeros_like(im)
        for region in largest_regions:
            mask = labels == region.label
            # To use a convex hull
            hull = convex_hull_image(mask)
            kidneyMask[hull] = 1
        
        print('Full kidney mask created:')
        t.toc()
        print('\n')

        return kidneyMask

    def refineMask(self, im, mask, fullKidneyMask):
        """
        Function to process the mask and get rid of objects with horizontal and vertical straight borders,
        as well as objects with low homogeneity (not constant pixel intensity)
        """
        t = TicToc()
        t.tic()

        winProportion = self.winProportion
        proportionBorder = self.propBorderForStraingthObjects
        bboxSizeThres = self.bboxSizeThres

        im = np.mean(im, axis=2)

        # Get rid of all cysts outside the kidney by multiplying with the full kidney mask
        mask = mask * fullKidneyMask

        # Define kernels for edge detection
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        # Get the large threshold based on windows size
        winSize = np.floor(mask.shape[1] / winProportion).astype(int)

        # Loop over all the values different from 0 in the mask
        for value in np.sort(np.unique(mask))[1:]:
            maskInd = np.where(mask == value)

            # Get the intensity of the pixels in the mask and normalize them ([0,1] interval)
            maskValues = im[maskInd]

            # Modify the values of the mask so that they are in the [0,1] interval and compute
            # their standard deviation
            maskValues = maskValues - np.min(maskValues) / np.max(maskValues)
            homStd = np.std(maskValues)

            bbox = copy.deepcopy(
                mask[
                    int(np.min(maskInd[0])) : int(np.max(maskInd[0])),
                    int(np.min(maskInd[1])) : int(np.max(maskInd[1])),
                ]
            )

            bbox[bbox != value] = 0
            bbox[bbox == value] = 1

            # If there is a low homogeneity in the mask, delete it
            if homStd >= 45:
                mask[mask == value] = 0

            else:
                if (bbox.shape[0] != 0) and (bbox.shape[1] != 0):
                    # Bounding box with a row/column of 0s added at the borders
                    bboxPadded = np.zeros([bbox.shape[0] + 2, bbox.shape[1] + 2])
                    bboxPadded[1:-1, 1:-1] = bbox

                    bordersAlong0 = ndimage.filters.convolve(bboxPadded, Ky)[1:-1, 1:-1]
                    bordersAlong1 = ndimage.filters.convolve(bboxPadded, Kx)[1:-1, 1:-1]

                    size0 = bbox.shape[0]
                    size1 = bbox.shape[1]

                    # If many pixels with a high border absolute value are in a row/column, that means
                    # there is a border. We will scan the bounding box:
                    # For every row
                    for i in range(size1):
                        # If the bounding box is large enough
                        largeEnough = (size0 >= bboxSizeThres) and (
                            size1 >= bboxSizeThres
                        )

                        # And you find some border larger than the specified proportion of that dimension (dimension 0 or 1)
                        if largeEnough and (
                            np.count_nonzero(bordersAlong1[:, i] == 4)
                            >= np.floor(size0 * proportionBorder)
                            or (
                                np.count_nonzero(bordersAlong1[:, i] == -4)
                                >= np.floor(size0 * proportionBorder)
                            )
                        ):
                            # Delete the mask
                            mask[mask == value] = 0

                    # For every column
                    for i in range(size0):
                        # If there are many pixels (more than half the height) with a strong border
                        if (
                            np.count_nonzero(bordersAlong0[i, :] == 4)
                            >= np.floor(winSize / 12)
                        ) or (
                            np.count_nonzero(bordersAlong0[i, :] == -4)
                            >= np.floor(winSize / 12)
                        ):
                            # Delete the mask
                            mask[mask == value] = 0
        
        print('Mask refined:')
        t.toc()
        print('\n')
        return mask

    def displayImages(self, im, mask, fullKidneyMask):
        '''
        Display the original image, the full kidney mask and the cysts mask
        '''
        plt.figure()
        #plt.subplot(3, 1, 1)
        plt.imshow(im)
        plt.title('Original image')
        plt.show()

        plt.figure()
        #plt.subplot(3, 1, 2)
        plt.imshow(fullKidneyMask)
        plt.title('Full kidney mask')
        plt.show()

        plt.figure()
        #plt.subplot(3, 1, 3)
        plt.imshow(mask)
        plt.title('Cysts mask')
        plt.show()

    def runFullProcess(self):
        """
        Apply the whole processing workflow to an image
        """

        t= TicToc()
        t.tic()

        imName = self.imName
        outputFolder = self.outputFolder
        fullKidneyOutputFolder = self.fullKidneyOutputFolder
        save = self.save

        imRawName = os.path.basename(imName)

        print('\n\nProcessing image {}\n'.format(imRawName))

        # Read the image
        im = self.load_im(imName)

        # Apply SAM in patches to get a preliminary cysts mask
        imMask = self.applySAMHistoSeg(im)

        # Correct the patches' borders
        imMask = self.correctPatchBorders(imMask)

        # Extract the full kidney mask
        fullKidneyMask = self.getFullKidneyMask(im)

        # Refine the cysts mask
        imMask = self.refineMask(im, imMask, fullKidneyMask)

        # If the save option is activated, save the masks
        if save:
            tif.imsave("{}/cystsMask-{}".format(outputFolder, imRawName), imMask)
            tif.imsave(
                "{}/kidneyMask-{}".format(fullKidneyOutputFolder, imRawName), fullKidneyMask
            )
  
        print('Full processing workflow completed:')
        t.toc()

        #Display the images
        self.displayImages(im, imMask, fullKidneyMask)

        return imMask, fullKidneyMask


if __name__ == "__main__":
    """
    If the script is run directly
    """
    print("Initializing")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--imName",
        type=str,
        help="Full path to the image to be processed"
    )

    parser.add_argument(
        "--outputFolder",
        type=str,
        help="Path to the folder where the masks will be saved"
    )

    parser.add_argument(
        "--fullKidneyOutputFolder",
        type=str,
        help="Path to the folder where the full kidney masks will be saved"
    )
    
    parser.add_argument(
        "--checkpointPath",
        type=str,
        help="Path to the SAM checkpoint to load the pretrained model"
    )
    parser.add_argument(
        "--winProportion",
        type=int,
        help="Proportion of window size"
    )
    parser.add_argument(
        "--largeThresProportion",
        type=int,
        help="Proportion of large threshold",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of model to be used"
    )
    parser.add_argument(
        "--SAM_points_per_side",
        type=int,
        help="Number of SAM points per side"
    )

    parser.add_argument(
        "--patchBorderScanRange",
        type=int,
        help="Number of pixel rows to scan around the patch border for the patch border correction"
    )
    parser.add_argument(
        "--maxNumKidneyPieces",
        type=int,
        help="Maximum number of kidney pieces to be considered for the full kidney mask"
    )
    parser.add_argument(
        "--propBorderForStraingthObjects",
        type=float,
        help="Proportion of an object's bounding box width/height that must be occupied by a strong border"+
        "in order to delete the object during mask refinement"
    )
    parser.add_argument(
        "--bboxSizeThres",
        type=int,
        help="Minimum size of the height/width of a bounding box to be considered for the strong border correction"
    )
    parser.add_argument(
        "--save",
        type=bool,
        help="If true, the masks will be saved in the corresponding output folder"
    )

    print("Parsing the arguments")
    args = parser.parse_args()

    #Create an instance of the SAM_cyst_workflow class
    workflow = SAM_cyst_workflow(args=args)

    #Apply the full processing workflow
    cystMask, kineyMask = workflow.runFullProcess()
