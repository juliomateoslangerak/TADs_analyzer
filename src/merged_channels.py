from typing import Optional, Sequence
from xml.etree import ElementTree

import os
from time import sleep
import json
import numpy as np
import tifffile
import xtiff
from getpass import getpass
from omero.gateway import BlitzGateway
from meta_data.crop_to_raw import crop_to_raw, crop_to_SIR

OME_TYPES = {
    np.bool_().dtype: 'bool',
    np.int8().dtype: 'int8',
    np.int16().dtype: 'int16',
    np.int32().dtype: 'int32',
    np.uint8().dtype: 'uint8',
    np.uint16().dtype: 'uint16',
    np.uint32().dtype: 'uint32',
    np.float32().dtype: 'float',
    np.float64().dtype: 'double',
}


# The "interleaved" argument was added in version 0.7.2. To not introduce breaking changes, it is a keyword-only
# argument and can as such be handled by ome_xml_kwargs in third-party get_ome_xml implementations that do not (yet)
# support this argument (e.g., imctools).
def get_ome_xml(img: np.ndarray, image_name: Optional[str], channel_names: Optional[Sequence[str]], big_endian: bool,
                pixel_size: Optional[float], pixel_depth: Optional[float], *, interleaved: bool = False,
                **ome_xml_kwargs) -> ElementTree.ElementTree:
    size_t, size_z, size_c, size_y, size_x, size_s = img.shape
    if channel_names is not None:
        assert len(channel_names) == size_c
    if pixel_size is not None:
        assert pixel_size > 0.
    if pixel_depth is not None:
        assert pixel_depth > 0
    ome_namespace = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'
    ome_schema_location = 'http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd'
    ome_element = ElementTree.Element('OME', attrib={
        'xmlns': ome_namespace,
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:schemaLocation': ' '.join((ome_namespace, ome_schema_location))
    })
    instrument_element = ElementTree.SubElement(ome_element, 'Instrument', attrib={
        'ID': 'Instrument:0'
    })
    objective_element = ElementTree.SubElement(instrument_element, 'Objective', attrib={
        'Correction': "Other",
        'ID': 'Objective:10007',
        'Immersion': "Oil",
        'LensNA': "1.4",
        'Manufacturer': "Olympus",
        'Model': "1-U2B836",
        'NominalMagnification': "100.0",
        'WorkingDistance': "0.13",
        'WorkingDistanceUnit': "mm"
    })
    image_element = ElementTree.SubElement(ome_element, 'Image', attrib={
        'ID': 'Image:0',
    })
    acquisition_date_element = ElementTree.SubElement(image_element, "AcquisitionDate")
    acquisition_date_element.text = str(ome_xml_kwargs["AcquisitionDate"])
    description_element = ElementTree.SubElement(image_element, "Description")
    description_element.text = ome_xml_kwargs['Description']
    if image_name is not None:
        image_element.set('Name', image_name)
    pixels_element = ElementTree.SubElement(image_element, 'Pixels', attrib={
        'ID': 'Pixels:0',
        'Type': OME_TYPES[img.dtype],
        'SizeX': str(size_x),
        'SizeY': str(size_y),
        'SizeC': str(size_c),
        'SizeZ': str(size_z),
        'SizeT': str(size_t),
        'DimensionOrder': 'XYCZT',
        'Interleaved': 'true' if interleaved else 'false',
        'BigEndian': 'true' if big_endian else 'false',
    })
    if pixel_size is not None:
        pixels_element.set('PhysicalSizeX', str(pixel_size))
        pixels_element.set('PhysicalSizeXUnit', 'µm')
        pixels_element.set('PhysicalSizeY', str(pixel_size))
        pixels_element.set('PhysicalSizeYUnit', 'µm')
    if pixel_depth is not None:
        pixels_element.set('PhysicalSizeZ', str(pixel_depth))
        pixels_element.set('PhysicalSizeZUnit', 'µm')
    for channel_id in range(size_c):
        channel_element = ElementTree.SubElement(pixels_element, 'Channel', attrib={
            'ID': 'Channel:0:{:d}'.format(channel_id),
            'SamplesPerPixel': str(size_s),
            'EmissionWavelength': str(ome_xml_kwargs['Channel']['EmissionWavelength'][channel_id]),
            'EmissionWavelengthUnit': ome_xml_kwargs['Channel']['EmissionWavelengthUnit'][channel_id],
            'ExcitationWavelength': str(ome_xml_kwargs['Channel']['ExcitationWavelength'][channel_id]),
            'ExcitationWavelengthUnit': ome_xml_kwargs['Channel']['ExcitationWavelengthUnit'][channel_id]
        }
                                                 )
        if channel_names is not None and channel_names[channel_id]:
            channel_element.set('Name', channel_names[channel_id])
    ElementTree.SubElement(pixels_element, 'TiffData')
    return ElementTree.ElementTree(element=ome_element)


# sourcery skip: merge-nested-ifs, raise-specific-error, use-fstring-for-concatenation
CHANNEL_NAME_MAPPINGS = {'683.0': 'ATTO-647', '608.0': 'ATTO-565', '528.0': 'Alexa-488', '435': 'DAPI'}

INPUT_DIRS = [
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/CTCF-AID_AUX",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/CTCF-AID_AUX-CTL",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/Drosophila_DAPI",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/Drosophila_TAD",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC/ESC_1C",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC/ESC_2C",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_DAPI",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA/ESC_TSA_1C",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA/ESC_TSA_2C",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA-CTL/ESC_TSA-CTL_1C",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA-CTL/ESC_TSA-CTL_2C",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ncxNPC",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/NPC/NPC_1C",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/NPC/NPC_2C",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/RAD21-AID_AUX",
    ## "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/RAD21-AID_AUX-CTL"
            ]

try:
    conn = BlitzGateway(username=input('username: '),
                        passwd=getpass('password: '),
                        host="omero.mri.cnrs.fr",
                        port=4064,
                        group="Cavalli Lab",
                        secure=True)
    conn.connect()

    for INPUT_DIR in INPUT_DIRS:
        print(f"Merging {INPUT_DIR}")
        OUTPUT_DIR = f'{INPUT_DIR}'
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)

        with open(os.path.join(INPUT_DIR, 'assay_config.json'), mode="r") as config_file:
            config = json.load(config_file)

        nr_channels = config["nr_channels"]
        files_set = {f[:-7] for f in os.listdir(INPUT_DIR) if f.endswith('.tif')}

        for file_root in files_set:
            sleep(1)
            # if os.path.exists(os.path.join(INPUT_DIR, f"{file_root}.ome.tiff")):
            #     continue
            image = tifffile.imread(os.path.join(INPUT_DIR, f"{file_root}_C1.tif"))

            for ch in range(1, nr_channels):
                new_channel = tifffile.imread(os.path.join(INPUT_DIR, f"{file_root}_C{ch + 1}.tif"))
                image = np.stack((image, new_channel))
            if nr_channels == 1:
                image = np.expand_dims(image, 0)  # adding non existing channel dimension
                if image.ndim == 3:  # There was no z so we add it
                    image = np.expand_dims(image, 0)

            # print(f"Processing file {file_root}")

            raw_image_id = crop_to_raw["_".join(file_root.split("_")[:-1])]
            sir_image_id = crop_to_SIR["_".join(file_root.split("_")[:-1])]
            raw_image = conn.getObject('Image', raw_image_id)
            sir_image = conn.getObject('Image', sir_image_id)

            # raw_image_name = raw_image.getName()
            raw_channels = raw_image.getChannels()

            # Verify channel matching
            if nr_channels < raw_image.getSizeC():
                excitation_wavelengths = [raw_channels[ch].getExcitationWave() for ch in
                                          config["raw-crop_channel_mapping"]]
                emission_wavelengths = [raw_channels[ch].getEmissionWave() for ch in config["raw-crop_channel_mapping"]]
            elif nr_channels == raw_image.getSizeC():
                excitation_wavelengths = [raw_channels[ch].getExcitationWave() for ch in
                                          range(nr_channels)]
                emission_wavelengths = [raw_channels[ch].getEmissionWave() for ch in range(nr_channels)]
            else:
                raise Exception("There were more channels in the crop than in the raw image")

            channel_names = [CHANNEL_NAME_MAPPINGS[str(raw_image.getChannelLabels()[c])] for c in range(nr_channels)]

            metadata = {
                # 'axes': 'CZYX',
                # # OME attributes
                # 'UUID'
                # 'Creator'
                # OME.Image attributes
                # 'Name'
                # OME.Image elements
                'AcquisitionDate': raw_image.getDate().date(),  # Remove time
                # 'AcquisitionDate': raw_image.getDate(),  # Remove time
                # 'AcquisitionDate': '2020-11-01T12:12:12',  # Remove time
                'Description': "3D 3Beam SI",
                # OME.Image.Pixels attributes:
                # 'SignificantBits'
                # 'PhysicalSizeX': 0.04,
                # 'PhysicalSizeXUnit': 'µm',
                # 'PhysicalSizeY': 0.04,
                # 'PhysicalSizeYUnit': 'µm',
                # 'PhysicalSizeZ': 0.125,
                # 'PhysicalSizeZUnit': 'µm',
                # 'TimeIncrement'
                # 'TimeIncrementUnit'
                'Plane': {
                    # 'ExposureTime': [.3] * (56 * nr_channels),
                    # 'ExposureTimeUnit': ['msec'] * (56 * nr_channels),
                    # 'PositionX'
                    # 'PositionXUnit'
                    # 'PositionY'
                    # 'PositionYUnit'
                    # 'PositionZ'
                    # 'PositionZUnit'
                },
                'Channel': {'Name': channel_names,
                            # 'AcquisitionMode'
                            # 'Color'
                            # 'ContrastMethod'
                            'EmissionWavelength': emission_wavelengths,
                            'EmissionWavelengthUnit': ['nm'] * nr_channels,
                            'ExcitationWavelength': excitation_wavelengths,
                            'ExcitationWavelengthUnit': ['nm'] * nr_channels,
                            # 'Fluor'
                            # 'IlluminationType'
                            # 'NDFilter': "",
                            # 'PinholeSize'
                            # 'PinholeSizeUnit'
                            # 'PockelCellSetting'
                            # 'SamplesPerPixel'
                            }
            }
            #
            # with tifffile.TiffWriter(os.path.join(OUTPUT_DIR, file_root + ".ome.tif")) as tif:
            #     tif.write(image, **metadata)

            xtiff.to_tiff(img=image.transpose((1, 0, 2, 3)),
                          file=os.path.join(OUTPUT_DIR, file_root + ".ome.tiff"),
                          image_date=raw_image.getDate().date(),
                          channel_names=channel_names,
                          profile=xtiff.tiff.TiffProfile.OME_TIFF,
                          pixel_size=round(sir_image.getPixelSizeX(), 5),
                          pixel_depth=round(sir_image.getPixelSizeZ(), 5),
                          ome_xml_fun=get_ome_xml,
                          **metadata
                          )

finally:
    conn.close()
    print('Done')
