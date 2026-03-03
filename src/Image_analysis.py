import json
import os
from math import sqrt

import numpy as np
import pandas as pd
import xtiff
from tifffile import imread
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.segmentation import clear_border, watershed, relabel_sequential
from skimage.morphology import remove_small_objects
from porespy.metrics import regionprops_3D

# Input and output directories
# INPUT_DIR = '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/CTCF-AID_AUX-CTL'
# INPUT_DIR = '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/CTCF-AID_AUX'
# INPUT_DIR = '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC'
# INPUT_DIR = '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA'
# INPUT_DIR = '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA-CTL'
# INPUT_DIR = '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ncxNPC'
INPUT_DIR = '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/NPC'
# INPUT_DIR = '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/RAD21-AID_AUX'
# INPUT_DIR = '/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/RAD21-AID_AUX-CTL'
OUTPUT_DIR = f'{INPUT_DIR}'

# Properties to measure
DOMAIN_PROPERTIES = (
    'label',
    'area',
    'filled_area',
    'major_axis_length',
    'centroid',
    'weighted_centroid',
    'equivalent_diameter',
    'max_intensity',
    'mean_intensity',
    'min_intensity',
    # 'coords',
)
SUBDOMAIN_PROPERTIES = (
    'label',
    'area',
    'filled_area',
    'major_axis_length',
    'centroid',
    'weighted_centroid',
    'max_intensity',
    'mean_intensity',
    'min_intensity',
    # 'coords',
)
OVERLAP_PROPERTIES = (
    'label',
    'area',
    'filled_area',
    'centroid',
    # 'coords',
)

# Analysis constants
IMAGE_FILE_EXTENSION = "ome.tiff"
DOMAIN_MIN_VOLUME = 200  # Minimum volume for the regions
SUBDOMAIN_MIN_VOLUME = 36  # Minimum volume for the regions
SIGMA = 0.5
PIXEL_SIZE = (.125, .04, .04)  # as ZYX
VOXEL_VOLUME = np.prod(PIXEL_SIZE)

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

with open(os.path.join(INPUT_DIR, 'assay_config.json'), mode="r") as config_file:
    config = json.load(config_file)

assay_id = config["assay_id"]


# Function definitions
def process_channel(channel: np.ndarray, properties: tuple, subdomain_properties: tuple,
                    sigma: float = None, min_volume: int = None,
                    subdomain_min_volume: int = None, binarize: bool = True):
    # Preprocessing
    if sigma is None:
        filtered = channel
    else:
        filtered = gaussian(channel, sigma=sigma, preserve_range=True).astype('uint16')

    # Detecting Domains
    thresholded = filtered > threshold_otsu(filtered)
    domain_labels = label(thresholded)
    domain_labels = clear_border(domain_labels)
    if min_volume is not None:
        domain_labels = domain_labels > 0
        domain_labels = remove_small_objects(domain_labels, connectivity=domain_labels.ndim, min_size=min_volume)
        domain_labels = relabel_sequential(domain_labels.astype('uint8'))[0]
    if binarize:
        domain_labels = domain_labels > 0
        domain_labels = domain_labels.astype('uint8')
    domain_props_dict = regionprops_table(label_image=domain_labels, intensity_image=channel,
                                          properties=properties)
    domain_props_df = pd.DataFrame(domain_props_dict)
    pore_props_3d = regionprops_3D(domain_labels)
    domain_props_df["sphericity"] = 0
    domain_props_df["solidity"] = 0
    for lab in pore_props_3d:
        domain_props_df.loc[domain_props_df.label == lab.label, "sphericity"] = lab.sphericity
        domain_props_df.loc[domain_props_df.label == lab.label, "solidity"] = lab.solidity
    domain_props_df.insert(loc=0, column='roi_type', value='domain')

    # Detecting Subdomains
    subdomain_labels = watershed(np.invert(channel), mask=domain_labels)
    if subdomain_min_volume is not None:
        subdomain_labels = remove_small_objects(subdomain_labels, connectivity=subdomain_labels.ndim,
                                                min_size=min_volume)
        subdomain_labels = relabel_sequential(subdomain_labels, offset=2)[0]
    subdomain_props_dict = regionprops_table(label_image=subdomain_labels, intensity_image=channel,
                                             properties=subdomain_properties)
    subdomain_props_df = pd.DataFrame(subdomain_props_dict)
    subdomain_props_df.insert(loc=0, column='roi_type', value='subdomain')

    # TODO: Add here nr of subdomains

    # Merging domain tables
    props_df = pd.concat([domain_props_df, subdomain_props_df], ignore_index=True)

    # Calculating some measurements
    props_df['volume'] = props_df['area'].apply(lambda a: a * VOXEL_VOLUME)
    props_df['volume_units'] = 'micron^3'

    return props_df, domain_labels, subdomain_labels


def process_overlap(labels, domains_df, overlap_properties):
    if labels.shape[0] == 2 and \
            np.max(labels[0]) == 1 and \
            np.max(labels[1]) == 1:

        overlap_labels = np.all(labels, axis=0).astype('uint8')
        overlap_props_dict = regionprops_table(label_image=overlap_labels,
                                               properties=overlap_properties)
        overlap_props_df = pd.DataFrame(overlap_props_dict)

        # if there is no overlap no rows are created. We nevertheless need to measure distance
        if len(overlap_props_df) == 0:
            overlap_props_df.loc[0, 'area'] = 0

        overlap_props_df.insert(loc=0, column='roi_type', value='overlap')

        overlap_props_df['volume'] = overlap_props_df['area'].apply(lambda a: a * VOXEL_VOLUME)
        overlap_props_df['volume_units'] = 'micron^3'

        # jaccard = (|A inter B| / (|A| + |B| - |A inter B|  ))
        overlap_props_df['overlap_fraction'] = abs(overlap_props_df.at[0,'volume']) / \
                            (abs(domains_df.loc[(domains_df['Channel ID'] == 0) & (domains_df['roi_type'] == 'domain'), 'volume'].values[0]) +
                             abs(domains_df.loc[(domains_df['Channel ID'] == 1) & (domains_df['roi_type'] == 'domain'), 'volume'].values[0]) -
                             abs(overlap_props_df.at[0,'volume']))

        overlap_props_df['distance_x'] = \
            abs(domains_df.loc[
                    (domains_df['Channel ID'] == 0) & (domains_df['roi_type'] == 'domain'), 'weighted_centroid-2'].values[0] - \
                domains_df.loc[
                    (domains_df['Channel ID'] == 1) & (domains_df['roi_type'] == 'domain'), 'weighted_centroid-2'].values[0]) * \
            PIXEL_SIZE[2]
        overlap_props_df['distance_y'] = \
            abs(domains_df.loc[
                    (domains_df['Channel ID'] == 0) & (domains_df['roi_type'] == 'domain'), 'weighted_centroid-1'].values[0] - \
                domains_df.loc[
                    (domains_df['Channel ID'] == 1) & (domains_df['roi_type'] == 'domain'), 'weighted_centroid-1'].values[0]) * \
            PIXEL_SIZE[1]
        overlap_props_df['distance_z'] = \
            abs(domains_df.loc[
                    (domains_df['Channel ID'] == 0) & (domains_df['roi_type'] == 'domain'), 'weighted_centroid-0'].values[0] - \
                domains_df.loc[
                    (domains_df['Channel ID'] == 1) & (domains_df['roi_type'] == 'domain'), 'weighted_centroid-0'].values[0]) * \
            PIXEL_SIZE[0]
        overlap_props_df['distance3d'] = sqrt(overlap_props_df.distance_x ** 2 +
                                              overlap_props_df.distance_y ** 2 +
                                              overlap_props_df.distance_z ** 2)
        overlap_props_df['distance_units'] = 'micron'

        # TODO: implement Matrix overlap

        return overlap_props_df, overlap_labels

    else:
        return None, None


def process_image(image, domain_properties, subdomain_properties, overlap_properties,
                  sigma=None, min_volume=None, subdomain_min_volume=None):
    rois_df = pd.DataFrame()

    domain_labels = np.zeros_like(image, dtype='uint8')
    subdomain_labels = np.zeros_like(image, dtype='uint8')

    # this order (starting by channel number) is not defined by default
    for channel_index, channel in enumerate(image):
        channel_props_df, channel_domain_labels, channel_subdomain_labels = \
            process_channel(channel=channel, properties=domain_properties,
                            subdomain_properties=subdomain_properties,
                            sigma=sigma, min_volume=min_volume,
                            subdomain_min_volume=subdomain_min_volume)

        domain_labels[channel_index] = channel_domain_labels
        subdomain_labels[channel_index] = channel_subdomain_labels

        channel_props_df.insert(loc=0, column='Channel ID', value=channel_index)

        rois_df = pd.concat([rois_df, channel_props_df], ignore_index=True)

    overlap_props_df, overlap_labels = process_overlap(labels=domain_labels,
                                                       domains_df=rois_df,
                                                       overlap_properties=overlap_properties)
    if overlap_props_df is not None:
        rois_df = pd.concat([rois_df, overlap_props_df], ignore_index=True)

    return rois_df, domain_labels, subdomain_labels, overlap_labels


def run():
    files_list = [f for f in os.listdir(INPUT_DIR) if
                  f.endswith(IMAGE_FILE_EXTENSION) and
                  not f.endswith(f"ROIs.{IMAGE_FILE_EXTENSION}")]

    analysis_df = pd.DataFrame()

    for img_file in files_list:
        print(f'Processing image: {img_file}')
        image = imread(os.path.join(INPUT_DIR, img_file))
        if image.ndim == 4:  # More than 1 channel
            image = image.transpose((1, 0, 2, 3))
        elif image.ndim == 3:  # One channel
            image = np.expand_dims(image, 1)

        rois_df, domain_labels, subdomain_labels, overlap_labels = \
            process_image(image=image,
                          domain_properties=DOMAIN_PROPERTIES,
                          subdomain_properties=SUBDOMAIN_PROPERTIES,
                          overlap_properties=OVERLAP_PROPERTIES,
                          sigma=SIGMA,
                          min_volume=DOMAIN_MIN_VOLUME,
                          subdomain_min_volume=SUBDOMAIN_MIN_VOLUME
                          )

        rois_df.insert(loc=0, column='Image Name', value=img_file)

        xtiff.to_tiff(img=domain_labels.transpose((1, 0, 2, 3)),
                      file=os.path.join(OUTPUT_DIR, f'{img_file[:-9]}_domains-ROIs.ome.tiff')
                      )
        xtiff.to_tiff(img=subdomain_labels.transpose((1, 0, 2, 3)),
                      file=os.path.join(OUTPUT_DIR, f'{img_file[:-9]}_subdomains-ROIs.ome.tiff')
                      )
        if overlap_labels is not None:
            xtiff.to_tiff(img=np.expand_dims(overlap_labels, axis=1),
                          file=os.path.join(OUTPUT_DIR, f'{img_file[:-9]}_overlap-ROIs.ome.tiff')
                          )

        analysis_df = pd.concat([analysis_df, rois_df], ignore_index=True)

    analysis_df.to_csv(os.path.join(OUTPUT_DIR, 'analysis_df.csv'))

    metadata_df = pd.read_csv(os.path.join(INPUT_DIR, f"{assay_id}_assays.csv"), header=1)  # TODO:

    merge_df = pd.merge(metadata_df, analysis_df, on="Image Name")
    merge_df.to_csv(os.path.join(OUTPUT_DIR, 'merged_df.csv'))


if __name__ == '__main__':
    run()
    print("done")

# imsave("C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID_merged_matpython\\20190720_RI512_CTCF-AID_AUX-CTL_61b_61a_SIR_2C_ALN_THR_labels_1.ome-tif", img_labels)
# print(img_labels.shape)

# for prop in img_region:
#     print('Label: {} >> Object size: {}'.format(prop.label, prop.area))
# img_region = np.array(regionprops(img_labeled))


#
# n files is the length of the image c1
# if nC == 1
# missubdomainvolume = 36
#
# ######
# alldistmatrixoverlap = [];
# 3D-sim = 1 ; pixel size xy:0.04 | Image size xy:55
# conv WF = 0 ; pixel size xy:0.08 | Image size xy:28
# zsize 0.125
# minimum volume = 200
#
# ##Output summary files
# if nC == 2
# alldistmatrixoverlap = [];
#
# ### Reading, filtering and segmentation of images
#
# filtered image : gaussian filter (imgaussfilt3) with sigma = 0.5
# thresholded image : threshold_otsu (graythresh/imbinarize) #Binarize 2-D grayscale image or 3-D volume by thresholding
#
# ### extraction of segmented objects
#
# labeled image : measure.label (bwconncomp) #Find and count connected components in binary image
# region image : measure.regionprops (regionprops3) # Measure properties of 3-D volumetric image regions
# find image object size > minimumn volume
# mask ismember(labelmatrix)
# mask imclearborder #Suppress light structures connected to image border
#
# if nC == 1
# regionprops(mask, vol, centroid, principalaxislength, surfacearea)
# if nC == 2
# regionprops(mask, centroid)
#
# only keep images with 1 segmented object when centroid is not equal to 1 and then
# jump to the next image file
#
#
#
#
# ### Intermingling between C1 and C2 channels
# if number of channels= 2
# maskmerge by adding the 2 masks, maskoverlap if maskmerge bigger than 1 , projmaskoverlap
#
# Overlap fraction :Jaccard similarity coefficient for image segmentation (mask1,mask2)
# overlapfractionsummary
# --> python: from sklearn.metrics import jaccard_score
#
# ## 3D distance between centroids in um ; which type M?
# xdistance = ( centroid1 of c2 - centroid1 of c1 ) * xypixelsize (0.04 | 0.08)
# y
# z
# distance = sqrt ( xdistance^2 + ydistance^2 + zdistance^2 )
#
# ## Matrix overlap
# statsprojmask2 = regionprops ( 'table', projmask C1 | C2 , 'area', 'centroid' )
# rounding to nearest decimal or integer ( xshift = nx/2 - statsprojmask2.centroid(1) then y for centroid (2)
# shiftmatrix = circshift: shift array circularly
# mergeimc1 = zeros (x,y) --> mergeimc1 = imadd(mergeimc1, double(shiftmatrix))
# double: Convert symbolic values to MATLAB double precision
# imadd: Add two images or add constant to image
#
# dist2D = sqrt((mask2c1.centroid(1)-mask2c2.centroid(1))^2 - (mask2c1.centroid(2)-mask2c2.centroid(2))^2
# alldistmatrixoverlap = [alldistmatrixoverlap; dist2D];
#
#
# ##  Probe structure analysis
#  only for number of channels= 1
# voxelvol = xy * xy * z
#
# #volume in um3
# vol = statsc1.vol(1) * voxelvol
#
# principapaxislength in um
# xdistc1 = statsc1.principalAxisLength(1) * xy
# ydistc1 = statsc1.principalAxisLength(2) * xy
# zdistc1 = statsc1.principalAxisLength(3) * z
# principalAxisLength = sqrt(xdistc1^2 + ydistc1^2 + zdistc1^2)
#
# #sphericity formula; surface area to volume
#
# ##waterrshed segmentation
# image of zeros = im
# mat2gray(im) :Convert matrix to grayscale image
# imA not im
# Inf: Create array of all Inf values
# A = watershed(imA) ; A not mask = 0
#
# connectedcA = measure.labels(A)
# prestatA = regionprops3(connectedcA, 'vol')
# mask idxA if prestatA.vol > minsubdomainvolume
# statsA = regionprops3(maskA, Vol, Centroid, ConvexHull)
# subdomaincentroid = stasA.Centroid
#
# #subdomainvolume = statsA{:,1}.* voxelvol
# table
#
# #nsubdomains = size(statsA,1) #array size
# table
#
# ## Output images
# images with segmentation borders
#
# adjprojC1 = imadjust(mat2gray(projC1),[low out .05;high out 1]) projC1 == channel filtered without max  #imadjust: Adjust image intensity values or colormap
# if nC == 2 --> adjprojC2 then overlayC1C2 = imfuse(adjprojC1,adjprojC2) ##Composite of two images
#
# #bwboundaries: Trace region boundaries in binary image
# bound1 = skimage.segmentation.find_boundaries(projmask2C1, connectivity=1, mode='thick', background=0)
# if nC == 2 --> bound2(projmask2C2) and bound3(projmaskoverlap)
# #num2str: Convert numbers to character array
# of = OF = 'space'num2str(overlapfraction,precision 3) #precision isMaximum number of significant digits
#
#
# figure (position , coord)
# set(gcf, 'color', 'w') #set(H,NameArray,ValueArray)
# ax = gca Current axes or chart
# Axes Position-Related Properties
#
# Set graphics object properties
#
# if nc == 1 --> imshow(adjprojC1)
# colb = y
# else
# imshow(overlayC1C2)
# cold =magenta
#
# for p = 1:length(bound1) ; boundary C1 = bound1{p} ; plot(boundaryC1(:,2), boundaryC1(:,1), 'color', colb, 'LineWidth', 2.25)
# if nC == 2
# for q = 1:length(bound2) ;  boundary C2 = bound1{q}; plot(boundaryC2(:,2), boundaryC1(:,1),'color', 'green', 'LineWidth', 2.25)
# end
# for r = 1:length(bound3) ; boundary C1 = bound1{r} ; plot(boundaryC3(:,2), boundaryC1(:,1), 'color', 'white', 'LineWidth', 3.25)
# end
# text(x,y, of,'Color', 'white','FontSize', 24)
#
# if nC == 2
# strrep: Find and replace substrings
# '_C1_C2_segmented.tif'
# else
# '_' + channel + '_segmented.tif'
#
# Outputimagename = fullfile(outputsubfolder,imname) #Build full file name from parts
# print (outputimagename, '-dtiff' , '-r300')
#
# ##summary files
# if nC==1  # Write table to file
# writetable(volumesummary,fullfile(outputsubfolder,volumesammaryname))
# writetable(principleaxieslegnthsummary,..
# writetable(sphericitysummary,..
# writetable(subdomainvolumesummary,..
# writetable(nsubdomainssummary,..
#
# if nC==2
# writetable(overlapfractionsummary,..
# writetable(distancecentroidsummary,..
#
# #averaged image if nC==2
# shiftoverlap = round(mean(alldistmatrixoverlap)/2)
#
