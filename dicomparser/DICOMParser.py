
import os, pdb
from io import BytesIO
import base64
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from pydicom import dcmread
from pydicom.uid import UID
from pydicom.uid import UID_dictionary
from pathlib import Path

# OPVs
from hvf_extraction_script.hvf_data.hvf_object import Hvf_Object
from hvf_extraction_script.utilities.file_utils import File_Utils

# PDFs
import pymupdf  # PyMuPDF
from PIL import Image

from oct_converter.readers import Dicom


OPHTHALMOLOGY_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.77.1.5.1": "Ophthalmic Photography 8 Bit Image Storage",
    "1.2.840.10008.5.1.4.1.1.77.1.5.2": "Ophthalmic Photography 16 Bit Image Storage",
    "1.2.840.10008.5.1.4.1.1.77.1.5.5": "Wide Field Ophthalmic Photography Stereographic Projection Image Storage",
    "1.2.840.10008.5.1.4.1.1.77.1.5.6": "Wide Field Ophthalmic Photography 3D Coordinates Image Storage",
    "1.2.840.10008.5.1.4.1.1.77.1.5.4": "Ophthalmic Tomography Image Storage",
    "1.2.840.10008.5.1.4.1.1.77.1.5.7": "Ophthalmic Optical Coherence Tomography En Face Image Storage",
    "1.2.840.10008.5.1.4.1.1.77.1.5.8": "Ophthalmic Optical Coherence Tomography B-scan Volume Analysis Storage",
    "1.2.840.10008.5.1.4.1.1.78.7": "Ophthalmic Axial Measurements Storage",
    "1.2.840.10008.5.1.4.1.1.78.8": "Intraocular Lens Calculations Storage",
    "1.2.840.10008.5.1.4.1.1.81.1": "Ophthalmic Thickness Map Storage",
    "1.2.840.10008.5.1.4.1.1.82.1": "Corneal Topography Map Storage",
    "1.2.840.10008.5.1.4.1.1.79.1": "Macular Grid Thickness and Volume Report Storage",
    "1.2.840.10008.5.1.4.1.1.80.1": "Ophthalmic Visual Field Static Perimetry Measurements Storage",
    "1.2.840.10008.5.1.4.1.1.78.1": "Lensometry Measurements Storage",
    "1.2.840.10008.5.1.4.1.1.78.2": "Autorefraction Measurements Storage",
    "1.2.840.10008.5.1.4.1.1.78.3": "Keratometry Measurements Storage",
    "1.2.840.10008.5.1.4.1.1.78.4": "Subjective Refraction Measurements Storage",
    "1.2.840.10008.5.1.4.1.1.78.5": "Visual Acuity Measurements Storage",
    "1.2.840.10008.5.1.4.1.1.78.6": "Spectacle Prescription Report Storage",
    "1.2.840.10008.5.1.4.1.1.7": "Secondary Capture Image Storage",
    "1.2.840.10008.5.1.4.1.1.7.2": "Multi-frame True Color Secondary Capture Image Storage",
    "1.2.840.10008.5.1.4.1.1.104.1": "Encapsulated PDF Storage",
    "1.2.840.10008.5.1.4.1.1.66": "Spatial Registration Storage",
    "1.2.840.10008.5.1.4.1.1.12.77": "Ophthalmic Tomography Image Storage"
    ""
}

class DICOMParser:
    """Base class for parsing DICOM files with a built-in factory method."""
    
    model_parsers = {}

    def __init__(self, dicom_path):
        self.dicom_path = Path(dicom_path)
        self.ds = dcmread(self.dicom_path)
        self.manufacturer = self.ds.get("Manufacturer", "Unknown")
        self.patient_id = self.ds.get("PatientID", "Unknown")
        self.model = self.ds.get("ManufacturerModelName", "Unknown")
        self.modality = self.ds.get("Modality", "Unknown")
        self.study_date = self.ds.get("StudyDate", "Unknown")
        self.sop_class = self.ds.get("SOPClassUID", "Unknown")
        self.sop_instance = self.ds.get("SOPInstanceUID", "Unknown")

    @classmethod
    def register_parser(cls, model_name, parser_class):
        cls.model_parsers[model_name] = parser_class

    @classmethod
    def create_parser(cls, dicom_path):
        ds = dcmread(dicom_path)
        model = ds.get("ManufacturerModelName", "Unknown")
        parser_class = cls.model_parsers.get(model, cls)
        return parser_class(dicom_path)
    
    # Common PDF Parser and Previewer to be replaced if not enough
    def _parse_pdf_pages(self):
        # 'Encapsulated PDF Storage'
        pdf_binary = self.ds.get((0x0042, 0x0011)).value
        pdf_document = pymupdf.open('pdf', pdf_binary)
        png_pages = {}
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            pixmap = page.get_pixmap()
            # print(page.get_text())
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            png_pages[f'page_{page_number + 1}'] = {
                f'page_html_img_base64':f"data:image/png;base64,{img_str}",
                f'page_PIL':image
            }
        return png_pages

    def _preview_pdf_pages(self, output_path, metadata):
        sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}")
        if not os.path.exists(sop_path): os.makedirs(sop_path) # make pdf (png) folder
        for page in metadata['png_pages'].keys():
            metadata['png_pages'][page]['page_PIL'].save(os.path.join(sop_path, f"{page}.png"))   


    def _write_detailed_dicom_header_to_file(self, output_path):
        with open(os.path.join(output_path, f"{self.sop_instance}.txt"), "w", encoding='utf-8') as file:
            file.write(str(self.ds))


    def parse(self):
        raise NotImplementedError("This should be implemented in a subclass.")

    def preview(self, output_path):
        metadata = self.extract_common_metadata()
        with open(os.path.join(output_path, f"{metadata['SOP Instance']}.json"), "w") as file:
            file.write(json.dumps(metadata, indent=4))
        return metadata

    def extract_common_metadata(self):
        """Extract metadata that applies to all DICOMs."""
        return {
            "Manufacturer": self.manufacturer,
            "Patient ID": self.patient_id,
            "Model": self.model,
            "Modality": self.modality,
            "Study Date": self.study_date,
            "SOP Class": self.sop_class,
            'SOP Class Description': OPHTHALMOLOGY_SOP_CLASSES[self.sop_class],
            "SOP Instance": self.sop_instance,
        }

    @staticmethod
    def get_bscan_images_from_pixel_array(pixel_arr):
        bscan_count = pixel_arr.shape[0]
        bscan_images = {}
        for i in range(bscan_count):
            bscan_image = Image.fromarray(pixel_arr[i, :, :])
            bscan_images[f"bscan{i+1}"] = bscan_image

        return bscan_images

    @staticmethod
    def save_bscan_images(meta, output_pth):
        sop_path = os.path.join(output_pth, f"{meta['SOP Instance']}")
        if not os.path.exists(sop_path): os.makedirs(sop_path) # make pdf (png) folder
        for bscan in meta['bscan_images'].keys():
            meta['bscan_images'][bscan].save(os.path.join(sop_path, f"{bscan}.png"))


### Begin Subclasses by manufacturermodelname ###


class ATLAS_9000(DICOMParser):
    """Parser for {
        'manufacturer': 'Carl Zeiss Meditec',
        'manufacturermodelname': 'ATLAS 9000',
        'modality': 'OPM'
        'sopclassuiddescription': None
        }
    """

    def parse(self):
        metadata = self.extract_common_metadata()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':
            # 'Encapsulated PDF Storage'
            metadata['png_pages'] = self._parse_pdf_pages()

        return metadata

    def preview(self, output_path, write_dicom_header=False):
        if write_dicom_header:
            self._write_detailed_dicom_header_to_file(output_path)
        metadata = self.parse()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':
            # 'Encapsulated PDF Storage'
            self._preview_pdf_pages(output_path, metadata)


# The String is from the ManufacturerModelName field in the DICOM file
DICOMParser.register_parser("ATLAS 9000", ATLAS_9000)




class CIRRUS_HD_OCT4000(DICOMParser):
    """Parser for {
        'manufacturer': 'Carl Zeiss Meditec',
        'manufacturermodelname': 'CIRRUS HD-OCT 4000',
        'modality': 'OPT',
        'sopclassuiddescription': 'Encapsulated PDF Storage' or 'Spatial Registration Storage']
        }
    """

    def parse(self):
        metadata = self.extract_common_metadata()
        # Series Description
        metadata["Series Description"] = self.ds.get("SeriesDescription", "Unknown")
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':
            # 'Encapsulated PDF Storage'
            metadata['png_pages'] = self._parse_pdf_pages()
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.66':
            # Spatial Registration Storage
            if metadata["Series Description"] == 'Macular Thickness':
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # DeviceSerialNumber
                metadata["DeviceSerialNumber"] = self.ds.get("DeviceSerialNumber", "Unknown")
                # ReferencedInstanceSequence
                ReferencedInstanceSequence = []
                for InstanceSequence in self.ds.get("ReferencedInstanceSequence", "Unknown"):
                    IS = {
                        "ReferencedSOPClassUID": InstanceSequence.ReferencedSOPClassUID,
                        "ReferencedSOPInstance_UID": InstanceSequence.ReferencedSOPInstanceUID
                    }
                    ReferencedInstanceSequence.append(IS)
                metadata["ReferencedInstanceSequence"] = ReferencedInstanceSequence
                # Acquisition Context Sequence
                AcquisitionContextSequence = []
                for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                    CS = {
                        "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    AcquisitionContextSequence.append(CS)
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
                metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])
                metadata['(0x0409, 0x1001)'] = f"{self.ds[(0x0409,0x1001)].VR}: Array of {len(self.ds[(0x0409,0x1001)].value)} elements"
                metadata['(0x0409, 0x1002)'] = f"{self.ds[(0x0409,0x1002)].VR}: Array of {len(self.ds[(0x0409,0x1002)].value)} elements"
                metadata['(0x0409, 0x1003)'] = f"{self.ds[(0x0409,0x1003)].VR}: Array of {len(self.ds[(0x0409,0x1003)].value)} elements"
            
            elif metadata["Series Description"] == 'Macular Cube 512x128':
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # DeviceSerialNumber
                metadata["DeviceSerialNumber"] = self.ds.get("DeviceSerialNumber", "Unknown")
                # Acquisition Context Sequence
                AcquisitionContextSequence = []
                for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                    CS = {
                        "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    AcquisitionContextSequence.append(CS)
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
                metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])

                slices = []
                for slice in self.ds[(0x0407, 0x10a0)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a0)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a0)][0][(0x0407, 0x100e)].value, "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a1)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x100e)].value,
                                                "images":slices,
                                                "(0x0407, 0x1015)":f"{self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1015)].VR}: Array of {len(self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1015)].value)} elements",
                                                "(0x0407, 0x1016)":f"{self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1016)].VR}: Array of {len(self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1016)].value)} elements"}
                slices = []
                for slice in self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a2)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a3)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a6)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a7)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10b5)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x101c)].value],
                                                "images":slices}

            else:
                # Series Description
                metadata["Series Description"] = self.ds.get("SeriesDescription", "Unknown")
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # PositionReferenceIndicator
                metadata["PositionReferenceIndicator"] = self.ds.get("PositionReferenceIndicator", "Unknown")
                # DeviceSerialNumber
                metadata["DeviceSerialNumber"] = self.ds.get("DeviceSerialNumber", "Unknown")
                # FrameOfReferenceUID
                metadata["FrameOfReferenceUID"] = self.ds.get("FrameOfReferenceUID", "Unknown")
                # SynchronizationFrameOfReferenceUID
                metadata["SynchronizationFrameOfReferenceUID"] = self.ds.get("SynchronizationFrameOfReferenceUID", "Unknown")
            
        return metadata

    def preview(self, output_path, write_dicom_header=False):
        if write_dicom_header:
            self._write_detailed_dicom_header_to_file(output_path)
        metadata = self.parse()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':
            # 'Encapsulated PDF Storage'
            self._preview_pdf_pages(output_path, metadata)
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.66':
            # Spatial Registration Storage
            with open(os.path.join(output_path, f"{metadata['SOP Instance']}.json"), "w") as file:
                sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}.json")
                file.write(json.dumps(metadata, indent=4))


# The String is from the ManufacturerModelName field in the DICOM file
DICOMParser.register_parser("CIRRUS HD-OCT 4000", CIRRUS_HD_OCT4000)


class CIRRUS_HD_OCT_5000(DICOMParser):
    """Parser for {
        'manufacturer': 'Carl Zeiss Meditec',
        'manufacturermodelname': 'CIRRUS HD-OCT 5000',
        'modality': 'OP', 'OPT'

        'sopclassuiddescription': 'Ophthalmic Photography 8 Bit Image Storage',
          or 'Encapsulated PDF Storage',
             'Opthalmic Tomography Image Storage',
             'Spatial Registration Storage',
        }
    """

    def parse(self):
        metadata = self.extract_common_metadata()
        # Series Description
        metadata["Series Description"] = self.ds.get("SeriesDescription", "Unknown")
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.77.1.5.1':
            # 'Ophthalmic Photography 8 Bit Image Storage'
            try:
                pixel_array = self.ds.pixel_array
            except Exception as e:
                print("pixel array issue")
                print(repr(e))
            
            image = Image.fromarray(pixel_array)
            metadata['image_PIL'] = image
            # Possible Image Kind
            metadata["Image Type"] = self.ds.get("ChannelDescriptionCodeSequence", "Unknown")[0].CodeMeaning
            # Laterality
            metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
            # Bits Allocated
            metadata["Bits Allocated"] = self.ds.get("BitsAllocated", "Unknown")
            # Photometric Interpretation
            metadata["Photometric Interpretation"] = self.ds.get("PhotometricInterpretation", "Unknown")
            # Pixel Spacing
            metadata["Pixel Spacing"] = self.ds.get("PixelSpacing", "Unknown")
            # Private Tags
            metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':       
            # 'Encapsulated PDF Storage'     
            metadata['png_pages'] = self._parse_pdf_pages()
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.77.1.5.4':
            # 'Ophthalmic Tomography Image Storage'
            if metadata["Series Description"] == 'RASTER_SINGLE':
                # RASTER_SINGLE
                try:
                    pixel_array = self.ds.pixel_array
                except Exception as e:
                    print("pixel array issue")
                    print(repr(e))
                image = Image.fromarray(pixel_array)
                metadata['image_PIL'] = image
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # Bits Allocated
                metadata["Bits Allocated"] = self.ds.get("BitsAllocated", "Unknown")
                # Photometric Interpretation
                metadata["Photometric Interpretation"] = self.ds.get("PhotometricInterpretation", "Unknown")
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
            else:
                try:
                    pixel_array = self.ds.pixel_array
                    # pixel_array = np.transpose(pixel_array, (0, 2, 1))  # Now shape is (128, 512, 1024)
                except Exception as e:
                    print("pixel array issue")
                    print(repr(e))
                bscan_count = pixel_array.shape[0]
                bscan_images = {}
                for i in range(bscan_count):
                    bscan_image = Image.fromarray(pixel_array[i, :, :])
                    bscan_images[f"bscan{i+1}"] = bscan_image
                metadata['bscan_images'] = bscan_images
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.66':
            # Spatial Registration Storage
            ## Series Description
            if metadata["Series Description"] == 'Macular Thickness':
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # DeviceSerialNumber
                metadata["DeviceSerialNumber"] = self.ds.get("DeviceSerialNumber", "Unknown")
                # ReferencedInstanceSequence
                ReferencedInstanceSequence = []
                for InstanceSequence in self.ds.get("ReferencedInstanceSequence", "Unknown"):
                    IS = {
                        "ReferencedSOPClassUID": InstanceSequence.ReferencedSOPClassUID,
                        "ReferencedSOPInstance_UID": InstanceSequence.ReferencedSOPInstanceUID
                    }
                    ReferencedInstanceSequence.append(IS)
                metadata["ReferencedInstanceSequence"] = ReferencedInstanceSequence
                # Acquisition Context Sequence
                AcquisitionContextSequence = []
                for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                    CS = {
                        "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    AcquisitionContextSequence.append(CS)
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
                metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])
                metadata['(0x0409, 0x1001)'] = f"{self.ds[(0x0409,0x1001)].VR}: Array of {len(self.ds[(0x0409,0x1001)].value)} elements"
                metadata['(0x0409, 0x1002)'] = f"{self.ds[(0x0409,0x1002)].VR}: Array of {len(self.ds[(0x0409,0x1002)].value)} elements"
                metadata['(0x0409, 0x1003)'] = f"{self.ds[(0x0409,0x1003)].VR}: Array of {len(self.ds[(0x0409,0x1003)].value)} elements"
                
            elif metadata["Series Description"] == 'Macular Cube 512x128':
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # DeviceSerialNumber
                metadata["DeviceSerialNumber"] = self.ds.get("DeviceSerialNumber", "Unknown")
                # Acquisition Context Sequence
                AcquisitionContextSequence = []
                for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                    CS = {
                        "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    AcquisitionContextSequence.append(CS)
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
                metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])

                slices = []
                for slice in self.ds[(0x0407, 0x10a0)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a0)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a0)][0][(0x0407, 0x100e)].value, "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a1)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x100e)].value,
                                                "images":slices,
                                                "(0x0407, 0x1015)":f"{self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1015)].VR}: Array of {len(self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1015)].value)} elements",
                                                "(0x0407, 0x1016)":f"{self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1016)].VR}: Array of {len(self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1016)].value)} elements"}
                slices = []
                for slice in self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a2)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a3)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a6)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a7)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10b5)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x101c)].value],
                                                "images":slices}

            elif metadata["Series Description"] == 'Glaucoma OU Analysis':
                # self._write_detailed_dicom_header_to_file("/scratch90/QTIM/Active/23-0284/dashboard/Data/forum_all_image_classification/CIRRUS HD-OCT 5000/OPT/Spatial Registration Storage/Glaucoma OU Analysis")
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # DeviceSerialNumber
                metadata["DeviceSerialNumber"] = self.ds.get("DeviceSerialNumber", "Unknown")
                # ReferencedInstanceSequence
                ReferencedInstanceSequence = []
                for InstanceSequence in self.ds.get("ReferencedInstanceSequence", "Unknown"):
                    IS = {
                        "ReferencedSOPClassUID": InstanceSequence.ReferencedSOPClassUID,
                        "ReferencedSOPInstance_UID": InstanceSequence.ReferencedSOPInstanceUID,
                        "CodeMeaning": InstanceSequence.PurposeOfReferenceCodeSequence[0].CodeMeaning
                    }
                    ReferencedInstanceSequence.append(IS)
                metadata["ReferencedInstanceSequence"] = ReferencedInstanceSequence
                # Acquisition Context Sequence
                AcquisitionContextSequence = []
                for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                    CS = {
                        "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    AcquisitionContextSequence.append(CS)
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
                metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])

                metadata['(0x0409, 0x1001)'] = f"{self.ds[(0x0409, 0x1001)].VR}: Array of {len(self.ds[(0x0409, 0x1001)].value)} elements"
                metadata['(0x0409, 0x1002)'] = f"{self.ds[(0x0409, 0x1002)].VR}: Array of {len(self.ds[(0x0409, 0x1002)].value)} elements"
                metadata['(0x0409, 0x1003)'] = f"{self.ds[(0x0409, 0x1003)].VR}: Array of {len(self.ds[(0x0409, 0x1003)].value)} elements"
                metadata['(0x0409, 0x1004)'] = f"{self.ds[(0x0409, 0x1004)].VR}: Array of {len(self.ds[(0x0409, 0x1004)].value)} elements"
                metadata['(0x0409, 0x1005)'] = f"{self.ds[(0x0409, 0x1005)].VR}: Array of {len(self.ds[(0x0409, 0x1005)].value)} elements"
                metadata['(0x0409, 0x1006)'] = f"{self.ds[(0x0409, 0x1006)].VR}: Array of {len(self.ds[(0x0409, 0x1006)].value)} elements"
                metadata['(0x0409, 0x1007)'] = f"{self.ds[(0x0409, 0x1007)].VR}: Array of {len(self.ds[(0x0409, 0x1007)].value)} elements"

                metadata['(0x0409, 0x10d2)'] = f"{self.ds[(0x0409, 0x10d2)].VR}: Array of {len(self.ds[(0x0409, 0x10d2)].value)} elements"
                metadata['(0x0409, 0x10d3)'] = f"{self.ds[(0x0409, 0x10d3)].VR}: Array of {len(self.ds[(0x0409, 0x10d3)].value)} elements"
                metadata['(0x0409, 0x10d4)'] = f"{self.ds[(0x0409, 0x10d4)].VR}: Array of {len(self.ds[(0x0409, 0x10d4)].value)} elements"
                metadata['(0x0409, 0x10d5)'] = f"{self.ds[(0x0409, 0x10d5)].VR}: Array of {len(self.ds[(0x0409, 0x10d5)].value)} elements"
                metadata['(0x0409, 0x10d6)'] = f"{self.ds[(0x0409, 0x10d6)].VR}: Array of {len(self.ds[(0x0409, 0x10d6)].value)} elements"
                metadata['(0x0409, 0x10d7)'] = f"{self.ds[(0x0409, 0x10d7)].VR}: Array of {len(self.ds[(0x0409, 0x10d7)].value)} elements"
                metadata['(0x0409, 0x10d8)'] = f"{self.ds[(0x0409, 0x10d8)].VR}: Array of {len(self.ds[(0x0409, 0x10d8)].value)} elements"
                metadata['(0x0409, 0x10d9)'] = f"{self.ds[(0x0409, 0x10d9)].VR}: Array of {len(self.ds[(0x0409, 0x10d9)].value)} elements"
                metadata['(0x0409, 0x10da)'] = f"{self.ds[(0x0409, 0x10da)].VR}: Array of {len(self.ds[(0x0409, 0x10da)].value)} elements"
                metadata['(0x0409, 0x10db)'] = f"{self.ds[(0x0409, 0x10db)].VR}: Array of {len(self.ds[(0x0409, 0x10db)].value)} elements"
                metadata['(0x0409, 0x10dc)'] = f"{self.ds[(0x0409, 0x10dc)].VR}: Array of {len(self.ds[(0x0409, 0x10dc)].value)} elements"
                metadata['(0x0409, 0x10dd)'] = f"{self.ds[(0x0409, 0x10dd)].VR}: Array of {len(self.ds[(0x0409, 0x10dd)].value)} elements"

                metadata['(0x0409, 0x10ef)'] = f"{self.ds[(0x0409, 0x10ef)].VR}: Array of {len(self.ds[(0x0409, 0x10ef)].value)} elements"
            
            elif metadata["Series Description"] == 'Optic Disc Cube 200x200':
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # DeviceSerialNumber
                metadata["DeviceSerialNumber"] = self.ds.get("DeviceSerialNumber", "Unknown")
                # Acquisition Context Sequence
                AcquisitionContextSequence = []
                for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                    CS = {
                        "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    AcquisitionContextSequence.append(CS)
                
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
                metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])

                slices = []
                for slice in self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a1)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x100e)].value,
                                                "images":slices,
                                                "(0x0407, 0x1015)":f"{self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1015)].VR}: Array of {len(self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1015)].value)} elements",
                                                "(0x0407, 0x1016)":f"{self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1016)].VR}: Array of {len(self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1016)].value)} elements"}
                slices = []
                for slice in self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a2)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a3)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a6)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a7)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10b5)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
            
            elif metadata["Series Description"] == 'RASTER_21_LINES' or metadata["Series Description"] == 'HD 5 Line Raster':
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
                metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])
                # Acquisition Context Sequence
                AcquisitionContextSequence = []
                for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                    CS = {
                        "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    AcquisitionContextSequence.append(CS)
                slices = []
                for slice in self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a3)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a5)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a5)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a5)][0][(0x0407, 0x100e)].value,
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a6)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10b5)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x101c)].value],
                                                "images":slices}

            elif metadata["Series Description"] == '5 Line Raster':
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
                metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])
                # Acquisition Context Sequence
                AcquisitionContextSequence = []
                for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                    CS = {
                        "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    AcquisitionContextSequence.append(CS)
                slices = []
                for slice in self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a3)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a4)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a4)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a4)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a4)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a6)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10b5)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
            elif metadata["Series Description"] == 'Guided Progression Analysis':
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # DeviceSerialNumber
                metadata["DeviceSerialNumber"] = self.ds.get("DeviceSerialNumber", "Unknown")
                # Acquisition Context Sequence
                AcquisitionContextSequence = []
                for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                    CS = {
                        "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    AcquisitionContextSequence.append(CS)
                # ReferencedInstanceSequence
                ReferencedInstanceSequence = []
                for InstanceSequence in self.ds.get("ReferencedInstanceSequence", "Unknown"):
                    IS = {
                        "ReferencedSOPClassUID": InstanceSequence.ReferencedSOPClassUID,
                        "ReferencedSOPInstance_UID": InstanceSequence.ReferencedSOPInstanceUID,
                        "CodeMeaning": InstanceSequence.PurposeOfReferenceCodeSequence[0].CodeMeaning
                    }
                    ReferencedInstanceSequence.append(IS)
                metadata["ReferencedInstanceSequence"] = ReferencedInstanceSequence
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
                metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])

                metadata['(0x0409, 0x1001)'] = f"{self.ds[(0x0409, 0x1001)].VR}: Array of {len(self.ds[(0x0409, 0x1001)].value)} elements"
                metadata['(0x0409, 0x1002)'] = f"{self.ds[(0x0409, 0x1002)].VR}: Array of {len(self.ds[(0x0409, 0x1002)].value)} elements"
                metadata['(0x0409, 0x1003)'] = f"{self.ds[(0x0409, 0x1003)].VR}: Array of {len(self.ds[(0x0409, 0x1003)].value)} elements"
                metadata['(0x0409, 0x1004)'] = f"{self.ds[(0x0409, 0x1004)].VR}: Array of {len(self.ds[(0x0409, 0x1004)].value)} elements"
                metadata['(0x0409, 0x1005)'] = f"{self.ds[(0x0409, 0x1005)].VR}: Array of {len(self.ds[(0x0409, 0x1005)].value)} elements"
                metadata['(0x0409, 0x1006)'] = f"{self.ds[(0x0409, 0x1006)].VR}: Array of {len(self.ds[(0x0409, 0x1006)].value)} elements"
                metadata['(0x0409, 0x1007)'] = f"{self.ds[(0x0409, 0x1007)].VR}: Array of {len(self.ds[(0x0409, 0x1007)].value)} elements"

                metadata['(0x0409, 0x10d2)'] = f"{self.ds[(0x0409, 0x10d2)].VR}: Array of {len(self.ds[(0x0409, 0x10d2)].value)} elements"
                metadata['(0x0409, 0x10d3)'] = f"{self.ds[(0x0409, 0x10d3)].VR}: Array of {len(self.ds[(0x0409, 0x10d3)].value)} elements"
                metadata['(0x0409, 0x10d4)'] = f"{self.ds[(0x0409, 0x10d4)].VR}: Array of {len(self.ds[(0x0409, 0x10d4)].value)} elements"
                metadata['(0x0409, 0x10d5)'] = f"{self.ds[(0x0409, 0x10d5)].VR}: Array of {len(self.ds[(0x0409, 0x10d5)].value)} elements"
                metadata['(0x0409, 0x10d6)'] = f"{self.ds[(0x0409, 0x10d6)].VR}: Array of {len(self.ds[(0x0409, 0x10d6)].value)} elements"
                metadata['(0x0409, 0x10d7)'] = f"{self.ds[(0x0409, 0x10d7)].VR}: Array of {len(self.ds[(0x0409, 0x10d7)].value)} elements"
                metadata['(0x0409, 0x10d8)'] = f"{self.ds[(0x0409, 0x10d8)].VR}: Array of {len(self.ds[(0x0409, 0x10d8)].value)} elements"
                metadata['(0x0409, 0x10d9)'] = f"{self.ds[(0x0409, 0x10d9)].VR}: Array of {len(self.ds[(0x0409, 0x10d9)].value)} elements"
                metadata['(0x0409, 0x10da)'] = f"{self.ds[(0x0409, 0x10da)].VR}: Array of {len(self.ds[(0x0409, 0x10da)].value)} elements"
                metadata['(0x0409, 0x10db)'] = f"{self.ds[(0x0409, 0x10db)].VR}: Array of {len(self.ds[(0x0409, 0x10db)].value)} elements"
                metadata['(0x0409, 0x10dc)'] = f"{self.ds[(0x0409, 0x10dc)].VR}: Array of {len(self.ds[(0x0409, 0x10dc)].value)} elements"
                metadata['(0x0409, 0x10dd)'] = f"{self.ds[(0x0409, 0x10dd)].VR}: Array of {len(self.ds[(0x0409, 0x10dd)].value)} elements"

                metadata['(0x0409, 0x10ef)'] = f"{self.ds[(0x0409, 0x10ef)].VR}: Array of {len(self.ds[(0x0409, 0x10ef)].value)} elements"


        return metadata

    def preview(self, output_path, write_dicom_header=False):
        if write_dicom_header:
            self._write_detailed_dicom_header_to_file(output_path)
        metadata = self.parse()
        # 'Ophthalmic Photography 8 Bit Image Storage'
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.77.1.5.1':
            sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}")
            metadata['image_PIL'].save(os.path.join(output_path, sop_path+".png"))  # To save the image to a file (e.g., PNG format)
        
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':
            # 'Encapsulated PDF Storage'
            self._preview_pdf_pages(output_path, metadata)
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.77.1.5.4':
            # 'Ophthalmic Tomography Image Storage'
            if metadata["Series Description"] == 'RASTER_SINGLE':
                sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}")
                metadata['image_PIL'].save(os.path.join(output_path, sop_path+".png"))  # To save the image to a file (e.g., PNG format)
            else:
                ## Bscans
                sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}")
                if not os.path.exists(sop_path): os.makedirs(sop_path) # make pdf (png) folder
                for bscan in metadata['bscan_images'].keys():
                    metadata['bscan_images'][bscan].save(os.path.join(sop_path, f"{bscan}.png"))
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.66':
            # Spatial Registration Storage
            # if not os.path.exists(sop_path): os.makedirs(sop_path)
            with open(os.path.join(output_path, f"{metadata['SOP Instance']}.json"), "w") as file:
                sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}.json")
                file.write(json.dumps(metadata, indent=4))


# The String is from the ManufacturerModelName field in the DICOM file
DICOMParser.register_parser("CIRRUS HD-OCT 5000", CIRRUS_HD_OCT_5000)



class CIRRUS_HD_OCT_6000(DICOMParser):
    """Parser for {
        'manufacturer': 'Carl Zeiss Meditec',
        'manufacturermodelname': 'CIRRUS HD-OCT 6000',
        'modality': 'OP', 'OPT'

        'sopclassuiddescription': 'Ophthalmic Photography 8 Bit Image Storage',
          or 'Ophthalmic Tomography Image Storage',
             'Encapsulated PDF Storage',
             'Spatial Registration Storage'
        }
    """

    def parse(self):
        metadata = self.extract_common_metadata()
        # Series Description
        metadata["Series Description"] = self.ds.get("SeriesDescription", "Unknown") # will have SFA\GPA
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.77.1.5.1':
            # 'Ophthalmic Photography 8 Bit Image Storage'
            try:
                pixel_array = self.ds.pixel_array
            except Exception as e:
                print("pixel array issue")
                print(repr(e))
            image = Image.fromarray(pixel_array)
            metadata['image_PIL'] = image
            # Laterality
            metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
            # Bits Allocated
            metadata["Bits Allocated"] = self.ds.get("BitsAllocated", "Unknown")
            # Photometric Interpretation
            metadata["Photometric Interpretation"] = self.ds.get("PhotometricInterpretation", "Unknown")
            # Pixel Spacing
            metadata["Pixel Spacing"] = self.ds.get("PixelSpacing", "Unknown")
            # Private Tags
            metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':       
            # 'Encapsulated PDF Storage'     
            metadata['png_pages'] = self._parse_pdf_pages()
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.77.1.5.4':
            # 'Ophthalmic Tomography Image Storage'
            try:
                pixel_array = self.ds.pixel_array
                # pixel_array = np.transpose(pixel_array, (0, 2, 1))  # Now shape is (128, 512, 1024)

            except Exception as e:
                print("pixel array issue")
                print(repr(e))
            bscan_count = pixel_array.shape[0]
            bscan_images = {}
            for i in range(bscan_count):
                bscan_image = Image.fromarray(pixel_array[i, :, :])

                bscan_images[f"bscan{i+1}"] = bscan_image
            metadata['bscan_images'] = bscan_images
            en_face_image = Image.fromarray(np.max(pixel_array, axis=1))  # Collapse the depth axis
            metadata['en_face_image'] = en_face_image
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.66':
            # Spatial Registration Storage
            ## Series Description
            if metadata["Series Description"] == 'Macular Thickness':
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # DeviceSerialNumber
                metadata["DeviceSerialNumber"] = self.ds.get("DeviceSerialNumber", "Unknown")
                # ReferencedInstanceSequence
                ReferencedInstanceSequence = []
                for InstanceSequence in self.ds.get("ReferencedInstanceSequence", "Unknown"):
                    IS = {
                        "ReferencedSOPClassUID": InstanceSequence.ReferencedSOPClassUID,
                        "ReferencedSOPInstance_UID": InstanceSequence.ReferencedSOPInstanceUID
                    }
                    ReferencedInstanceSequence.append(IS)
                metadata["ReferencedInstanceSequence"] = ReferencedInstanceSequence
                # Acquisition Context Sequence
                AcquisitionContextSequence = []
                for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                    CS = {
                        "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    AcquisitionContextSequence.append(CS)
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
                metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])
                metadata['(0x0409, 0x1001)'] = f"{self.ds[(0x0409,0x1001)].VR}: Array of {len(self.ds[(0x0409,0x1001)].value)} elements"
                metadata['(0x0409, 0x1002)'] = f"{self.ds[(0x0409,0x1002)].VR}: Array of {len(self.ds[(0x0409,0x1002)].value)} elements"
                metadata['(0x0409, 0x1003)'] = f"{self.ds[(0x0409,0x1003)].VR}: Array of {len(self.ds[(0x0409,0x1003)].value)} elements"
                
            elif metadata["Series Description"] == 'Macular Cube 512x128':
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # DeviceSerialNumber
                metadata["DeviceSerialNumber"] = self.ds.get("DeviceSerialNumber", "Unknown")
                # Acquisition Context Sequence
                AcquisitionContextSequence = []
                for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                    CS = {
                        "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    AcquisitionContextSequence.append(CS)
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
                metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])

                slices = []
                for slice in self.ds[(0x0407, 0x10a0)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a0)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a0)][0][(0x0407, 0x100e)].value, "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a1)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x100e)].value,
                                                "images":slices,
                                                "(0x0407, 0x1015)":f"{self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1015)].VR}: Array of {len(self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1015)].value)} elements",
                                                "(0x0407, 0x1016)":f"{self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1016)].VR}: Array of {len(self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1016)].value)} elements"}
                slices = []
                for slice in self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a2)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a3)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a6)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a7)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10b5)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
            
            elif metadata["Series Description"] == 'Glaucoma OU Analysis':
                # self._write_detailed_dicom_header_to_file("/scratch90/QTIM/Active/23-0284/dashboard/Data/forum_all_image_classification/CIRRUS HD-OCT 6000/OPT/Spatial Registration Storage/Glaucoma OU Analysis")
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # DeviceSerialNumber
                metadata["DeviceSerialNumber"] = self.ds.get("DeviceSerialNumber", "Unknown")
                # ReferencedInstanceSequence
                ReferencedInstanceSequence = []
                for InstanceSequence in self.ds.get("ReferencedInstanceSequence", "Unknown"):
                    IS = {
                        "ReferencedSOPClassUID": InstanceSequence.ReferencedSOPClassUID,
                        "ReferencedSOPInstance_UID": InstanceSequence.ReferencedSOPInstanceUID,
                        "CodeMeaning": InstanceSequence.PurposeOfReferenceCodeSequence[0].CodeMeaning
                    }
                    ReferencedInstanceSequence.append(IS)
                metadata["ReferencedInstanceSequence"] = ReferencedInstanceSequence
                # Acquisition Context Sequence
                AcquisitionContextSequence = []
                for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                    CS = {
                        "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    AcquisitionContextSequence.append(CS)
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
                metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])

                metadata['(0x0409, 0x1001)'] = f"{self.ds[(0x0409, 0x1001)].VR}: Array of {len(self.ds[(0x0409, 0x1001)].value)} elements"
                metadata['(0x0409, 0x1002)'] = f"{self.ds[(0x0409, 0x1002)].VR}: Array of {len(self.ds[(0x0409, 0x1002)].value)} elements"
                metadata['(0x0409, 0x1003)'] = f"{self.ds[(0x0409, 0x1003)].VR}: Array of {len(self.ds[(0x0409, 0x1003)].value)} elements"
                metadata['(0x0409, 0x1004)'] = f"{self.ds[(0x0409, 0x1004)].VR}: Array of {len(self.ds[(0x0409, 0x1004)].value)} elements"
                metadata['(0x0409, 0x1005)'] = f"{self.ds[(0x0409, 0x1005)].VR}: Array of {len(self.ds[(0x0409, 0x1005)].value)} elements"
                metadata['(0x0409, 0x1006)'] = f"{self.ds[(0x0409, 0x1006)].VR}: Array of {len(self.ds[(0x0409, 0x1006)].value)} elements"
                metadata['(0x0409, 0x1007)'] = f"{self.ds[(0x0409, 0x1007)].VR}: Array of {len(self.ds[(0x0409, 0x1007)].value)} elements"

                metadata['(0x0409, 0x10d2)'] = f"{self.ds[(0x0409, 0x10d2)].VR}: Array of {len(self.ds[(0x0409, 0x10d2)].value)} elements"
                metadata['(0x0409, 0x10d3)'] = f"{self.ds[(0x0409, 0x10d3)].VR}: Array of {len(self.ds[(0x0409, 0x10d3)].value)} elements"
                metadata['(0x0409, 0x10d4)'] = f"{self.ds[(0x0409, 0x10d4)].VR}: Array of {len(self.ds[(0x0409, 0x10d4)].value)} elements"
                metadata['(0x0409, 0x10d5)'] = f"{self.ds[(0x0409, 0x10d5)].VR}: Array of {len(self.ds[(0x0409, 0x10d5)].value)} elements"
                metadata['(0x0409, 0x10d6)'] = f"{self.ds[(0x0409, 0x10d6)].VR}: Array of {len(self.ds[(0x0409, 0x10d6)].value)} elements"
                metadata['(0x0409, 0x10d7)'] = f"{self.ds[(0x0409, 0x10d7)].VR}: Array of {len(self.ds[(0x0409, 0x10d7)].value)} elements"
                metadata['(0x0409, 0x10d8)'] = f"{self.ds[(0x0409, 0x10d8)].VR}: Array of {len(self.ds[(0x0409, 0x10d8)].value)} elements"
                metadata['(0x0409, 0x10d9)'] = f"{self.ds[(0x0409, 0x10d9)].VR}: Array of {len(self.ds[(0x0409, 0x10d9)].value)} elements"
                metadata['(0x0409, 0x10da)'] = f"{self.ds[(0x0409, 0x10da)].VR}: Array of {len(self.ds[(0x0409, 0x10da)].value)} elements"
                metadata['(0x0409, 0x10db)'] = f"{self.ds[(0x0409, 0x10db)].VR}: Array of {len(self.ds[(0x0409, 0x10db)].value)} elements"
                metadata['(0x0409, 0x10dc)'] = f"{self.ds[(0x0409, 0x10dc)].VR}: Array of {len(self.ds[(0x0409, 0x10dc)].value)} elements"
                metadata['(0x0409, 0x10dd)'] = f"{self.ds[(0x0409, 0x10dd)].VR}: Array of {len(self.ds[(0x0409, 0x10dd)].value)} elements"

                metadata['(0x0409, 0x10ef)'] = f"{self.ds[(0x0409, 0x10ef)].VR}: Array of {len(self.ds[(0x0409, 0x10ef)].value)} elements"
                
            elif metadata["Series Description"] == 'Optic Disc Cube 200x200':
                # self._write_detailed_dicom_header_to_file('/scratch90/QTIM/Active/23-0284/dashboard/Data/forum_all_image_classification/CIRRUS HD-OCT 6000/OPT/Spatial Registration Storage/Optic Disc Cube 200x200')
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # DeviceSerialNumber
                metadata["DeviceSerialNumber"] = self.ds.get("DeviceSerialNumber", "Unknown")
                # Acquisition Context Sequence
                AcquisitionContextSequence = []
                for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                    CS = {
                        "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    AcquisitionContextSequence.append(CS)
                
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
                metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])

                slices = []
                for slice in self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a1)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x100e)].value,
                                                "images":slices,
                                                "(0x0407, 0x1015)":f"{self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1015)].VR}: Array of {len(self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1015)].value)} elements",
                                                "(0x0407, 0x1016)":f"{self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1016)].VR}: Array of {len(self.ds[(0x0407, 0x10a1)][0][(0x0407, 0x1016)].value)} elements"}
                slices = []
                for slice in self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a2)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a2)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a3)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a3)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a6)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a6)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10a7)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10a7)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
                slices = []
                for slice in self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x1005)]:
                    slice = f"{slice[(0x0407, 0x1006)].VR}: Array of {len(slice[(0x0407, 0x1006)].value)} elements"
                    slices.append(slice)
                metadata['(0x0407, 0x10b5)'] = {"(0x0407, 0x100e)":self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x100e)].value,
                                                "(0x0407, 0x101c)":OPHTHALMOLOGY_SOP_CLASSES[self.ds[(0x0407, 0x10b5)][0][(0x0407, 0x101c)].value],
                                                "images":slices}
            elif metadata["Series Description"] == 'Guided Progression Analysis':
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # DeviceSerialNumber
                metadata["DeviceSerialNumber"] = self.ds.get("DeviceSerialNumber", "Unknown")
                # Acquisition Context Sequence
                AcquisitionContextSequence = []
                for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                    CS = {
                        "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    AcquisitionContextSequence.append(CS)
                # ReferencedInstanceSequence
                ReferencedInstanceSequence = []
                for InstanceSequence in self.ds.get("ReferencedInstanceSequence", "Unknown"):
                    IS = {
                        "ReferencedSOPClassUID": InstanceSequence.ReferencedSOPClassUID,
                        "ReferencedSOPInstance_UID": InstanceSequence.ReferencedSOPInstanceUID,
                        "CodeMeaning": InstanceSequence.PurposeOfReferenceCodeSequence[0].CodeMeaning
                    }
                    ReferencedInstanceSequence.append(IS)
                metadata["ReferencedInstanceSequence"] = ReferencedInstanceSequence
                # Private Tags
                metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
                metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])

                metadata['(0x0409, 0x1001)'] = f"{self.ds[(0x0409, 0x1001)].VR}: Array of {len(self.ds[(0x0409, 0x1001)].value)} elements"
                metadata['(0x0409, 0x1002)'] = f"{self.ds[(0x0409, 0x1002)].VR}: Array of {len(self.ds[(0x0409, 0x1002)].value)} elements"
                metadata['(0x0409, 0x1003)'] = f"{self.ds[(0x0409, 0x1003)].VR}: Array of {len(self.ds[(0x0409, 0x1003)].value)} elements"
                metadata['(0x0409, 0x1004)'] = f"{self.ds[(0x0409, 0x1004)].VR}: Array of {len(self.ds[(0x0409, 0x1004)].value)} elements"
                metadata['(0x0409, 0x1005)'] = f"{self.ds[(0x0409, 0x1005)].VR}: Array of {len(self.ds[(0x0409, 0x1005)].value)} elements"
                metadata['(0x0409, 0x1006)'] = f"{self.ds[(0x0409, 0x1006)].VR}: Array of {len(self.ds[(0x0409, 0x1006)].value)} elements"
                metadata['(0x0409, 0x1007)'] = f"{self.ds[(0x0409, 0x1007)].VR}: Array of {len(self.ds[(0x0409, 0x1007)].value)} elements"

                metadata['(0x0409, 0x10d2)'] = f"{self.ds[(0x0409, 0x10d2)].VR}: Array of {len(self.ds[(0x0409, 0x10d2)].value)} elements"
                metadata['(0x0409, 0x10d3)'] = f"{self.ds[(0x0409, 0x10d3)].VR}: Array of {len(self.ds[(0x0409, 0x10d3)].value)} elements"
                metadata['(0x0409, 0x10d4)'] = f"{self.ds[(0x0409, 0x10d4)].VR}: Array of {len(self.ds[(0x0409, 0x10d4)].value)} elements"
                metadata['(0x0409, 0x10d5)'] = f"{self.ds[(0x0409, 0x10d5)].VR}: Array of {len(self.ds[(0x0409, 0x10d5)].value)} elements"
                metadata['(0x0409, 0x10d6)'] = f"{self.ds[(0x0409, 0x10d6)].VR}: Array of {len(self.ds[(0x0409, 0x10d6)].value)} elements"
                metadata['(0x0409, 0x10d7)'] = f"{self.ds[(0x0409, 0x10d7)].VR}: Array of {len(self.ds[(0x0409, 0x10d7)].value)} elements"
                metadata['(0x0409, 0x10d8)'] = f"{self.ds[(0x0409, 0x10d8)].VR}: Array of {len(self.ds[(0x0409, 0x10d8)].value)} elements"
                metadata['(0x0409, 0x10d9)'] = f"{self.ds[(0x0409, 0x10d9)].VR}: Array of {len(self.ds[(0x0409, 0x10d9)].value)} elements"
                metadata['(0x0409, 0x10da)'] = f"{self.ds[(0x0409, 0x10da)].VR}: Array of {len(self.ds[(0x0409, 0x10da)].value)} elements"
                metadata['(0x0409, 0x10db)'] = f"{self.ds[(0x0409, 0x10db)].VR}: Array of {len(self.ds[(0x0409, 0x10db)].value)} elements"
                metadata['(0x0409, 0x10dc)'] = f"{self.ds[(0x0409, 0x10dc)].VR}: Array of {len(self.ds[(0x0409, 0x10dc)].value)} elements"
                metadata['(0x0409, 0x10dd)'] = f"{self.ds[(0x0409, 0x10dd)].VR}: Array of {len(self.ds[(0x0409, 0x10dd)].value)} elements"

                metadata['(0x0409, 0x10ef)'] = f"{self.ds[(0x0409, 0x10ef)].VR}: Array of {len(self.ds[(0x0409, 0x10ef)].value)} elements"


        return metadata

    def preview(self, output_path, write_dicom_header=False):
        if write_dicom_header:
            self._write_detailed_dicom_header_to_file(output_path)
        metadata = self.parse()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.77.1.5.1':
            sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}")
            metadata['image_PIL'].save(os.path.join(output_path, sop_path+".png"))  # To save the image to a file (e.g., PNG format)
        
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':
            # 'Encapsulated PDF Storage'
            self._preview_pdf_pages(output_path, metadata)
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.77.1.5.4':
            # 'Ophthalmic Tomography Image Storage'
            ## Bscans
            sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}")
            if not os.path.exists(sop_path): os.makedirs(sop_path) # make pdf (png) folder
            for bscan in metadata['bscan_images'].keys():
                metadata['bscan_images'][bscan].save(os.path.join(sop_path, f"{bscan}.png"))
            ## En Face
            metadata['en_face_image'].save(os.path.join(sop_path, f"en_face_from_max_operation_across_bscans.png"))
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.66':
            # Spatial Registration Storage
            with open(os.path.join(output_path, f"{metadata['SOP Instance']}.json"), "w") as file:
                sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}.json")
                file.write(json.dumps(metadata, indent=4))

            


# The String is from the ManufacturerModelName field in the DICOM file
DICOMParser.register_parser("CIRRUS HD-OCT 6000", CIRRUS_HD_OCT_6000)


class CLARUS_700(DICOMParser):
    """Parser for {
        'manufacturer': 'Carl Zeiss Meditec',
        'manufacturermodelname': 'CLARUS 700',
        'modality': 'OP',

        'sopclassuiddescription': 'Ophthalmic Photography 8 Bit Image Storage',
        }
    """

    def parse(self):
        metadata = self.extract_common_metadata()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.77.1.5.1':
            # 'Ophthalmic Photography 8 Bit Image Storage'
            try:
                pixel_array = self.ds.pixel_array
            except Exception as e:
                print("pixel array issue")
                print(repr(e))
            arr = pixel_array.astype(np.float32)
            arr[..., 0] = arr[..., 0] + 1.402 * (arr[..., 2] - 128)
            arr[..., 1] = arr[..., 0] - 0.344136 * (arr[..., 1] - 128) - 0.714136 * (arr[..., 2] - 128)
            arr[..., 2] = arr[..., 0] + 1.772 * (arr[..., 1] - 128)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr)
            metadata['image_PIL'] = image
            # Bits Allocated
            metadata["Bits Allocated"] = self.ds.get("BitsAllocated", "Unknown")
            # Photometric Interpretation
            metadata["Photometric Interpretation"] = f"RGB from {self.ds.get('PhotometricInterpretation', 'Unknown')}" 

        return metadata

    def preview(self, output_path, write_dicom_header=False):
        if write_dicom_header:
            self._write_detailed_dicom_header_to_file(output_path)
        metadata = self.parse()
        # metadata.keys()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.77.1.5.1':
            # 'Ophthalmic Photography 8 Bit Image Storage'
            sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}")
            metadata['image_PIL'].save(os.path.join(output_path, sop_path+".png"))  # To save the image to a file (e.g., PNG format)
              


# The String is from the ManufacturerModelName field in the DICOM file
DICOMParser.register_parser("CLARUS 700", CLARUS_700)









# FORUM Cataract Workplace CALLISTOeye Plugin










class FORUMGlaucomaWorkplaceParser(DICOMParser):
    """Parser for {
        'manufacturer': 'Carl Zeiss Meditec',
        'manufacturermodelname': 'FORUM Glaucoma Workplace',
        'modality': 'OPV'

        'sopclassuiddescription': 'Encapsulated PDF Storage',
           or 'Ophthalmic Visual Field Static Perimetry Measurements Storage',
        }
    """

    def parse(self):
        metadata = self.extract_common_metadata()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.80.1':
            # 'Ophthalmic Visual Field Static Perimetry Measurements Storage'
            hvf_dicom = File_Utils.read_dicom_from_file(self.dicom_path);
            hvf_obj = Hvf_Object.get_hvf_object_from_dicom(hvf_dicom);
            metadata['HVF Object'] = hvf_obj.serialize_to_json()


        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':            
            metadata['png_pages'] = self._parse_pdf_pages()
        
        # Series Description
        metadata["Series Description"] = self.ds.get("SeriesDescription", "Unknown") # will have SFA\GPA

        return metadata

    def preview(self, output_path, write_dicom_header=False):
        if write_dicom_header:
            self._write_detailed_dicom_header_to_file(output_path)
        metadata = self.parse()
        metadata.keys()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.80.1':
            # with open(os.path.join(output_path, f"{metadata['SOP Instance']}.json"), "w") as file:
                # file.write(metadata['HVF Object'])
            sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}.json")
            File_Utils.write_string_to_file(metadata['HVF Object'], sop_path)
            
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':
            # 'Encapsulated PDF Storage'
            sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}")
            if not os.path.exists(sop_path): os.makedirs(sop_path) # make pdf (png) folder
            for page in metadata['png_pages'].keys():
                metadata['png_pages'][page]['page_PIL'].save(os.path.join(sop_path, f"{page}.png"))



# The String is from the ManufacturerModelName field in the DICOM file
DICOMParser.register_parser("FORUM Glaucoma Workplace", FORUMGlaucomaWorkplaceParser)




class HFA_3(DICOMParser):
    """Parser for {
        'manufacturer': 'Carl Zeiss Meditec',
        'manufacturermodelname': 'HFA 3',
        'modality': 'OPT',
        'sopclassuiddescription': 'Encapsulated PDF Storage' or 'Spatial Registration Storage']
        }
    """

    def parse(self):
        metadata = self.extract_common_metadata()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.66':
            # Spatial Registration Storage
            # Laterality
            metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
            # Acquisition Context Sequence
            AcquisitionContextSequence = []
            for ContextSequence in self.ds.get("AcquisitionContextSequence", "Unknown"):
                CS = {
                    "CodeMeaning": ContextSequence.ConceptNameCodeSequence[0].CodeMeaning,
                }
                AcquisitionContextSequence.append(CS)
            # Private Tags
            metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201, 0x1000)]])
            metadata['(0x0301, 0x1008)'] = f"{self.ds[(0x0301, 0x1008)].VR}: Array of {len(self.ds[(0x0301, 0x1008)].value)} elements"


        return metadata

    def preview(self, output_path, write_dicom_header=False):
        if write_dicom_header:
            self._write_detailed_dicom_header_to_file(output_path)
        metadata = self.parse()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.66':
            # Spatial Registration Storage
            with open(os.path.join(output_path, f"{metadata['SOP Instance']}.json"), "w") as file:
                sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}.json")
                file.write(json.dumps(metadata, indent=4))
                

# The String is from the ManufacturerModelName field in the DICOM file
DICOMParser.register_parser("HFA 3", HFA_3)


class Humphrey_Field_Analyzer_3(DICOMParser):
    """Parser for {
        'manufacturer': 'Carl Zeiss Meditec',
        'manufacturermodelname': 'Humphrey Field Analyzer 3',
        'modality': 'OP',

        'sopclassuiddescription': 'Ophthalmic Photography 8 Bit Image Storage',
        }
    """

    def parse(self, attempt_to_extract_dicom_tags_not_pixel_datas=False):
        metadata = self.extract_common_metadata()
        # self._write_detailed_dicom_header_to_file("/scratch90/QTIM/Active/23-0284/dashboard/Data/forum_all_image_classification/Humphrey Field Analyzer 3/OP/Ophthalmic Photography 8 Bit Image Storage/Tag not found")
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.77.1.5.1':
            if not attempt_to_extract_dicom_tags_not_pixel_datas:
                # 'Ophthalmic Photography 8 Bit Image Storage'
                try:
                    pixel_array = self.ds.pixel_array
                except Exception as e:
                    print("pixel array issue")
                    print(repr(e))
                bscan_count = pixel_array.shape[0]
                bscan_images = {}
                for i in range(bscan_count):
                    bscan_image = Image.fromarray(pixel_array[i, :, :])
                    bscan_images[f"bscan{i+1}"] = bscan_image
                metadata['bscan_images'] = bscan_images
            elif attempt_to_extract_dicom_tags_not_pixel_datas:
                # BB - I wrote this elif for the purpose of extracting the dicom tags that are not pixel data
                # Dictionary to accumulate sum and count for each (x, y) coordinate
                coord_value_map = defaultdict(lambda: {
                    "raw_sum": 0, "abs_sum": 0, "pattern_sum": 0, "abs_perc_sum": 0, "pattern_perc_sum": 0,
                    "count": 0
                })
                # Iterate over perimetry test points
                for frame_data in self.ds[(0x0303, 0x1010)].value:
                    def get_value(tag, default=np.nan):
                        """Safely get a DICOM tag value, returning NaN if missing."""
                        elem = frame_data.get(tag)
                        return elem.value if elem is not None else default

                    x = get_value((0x0303, 0x1013))  # X Coordinate
                    y = get_value((0x0303, 0x1014))  # Y Coordinate
                    raw_value = get_value((0x0303, 0x1017))  # Raw Threshold Sensitivity (dB)
                    abs_value = get_value((0x0303, 0x101d))  # Absolute Value (if available)
                    pattern_value = get_value((0x0303, 0x101e))  # Pattern Deviation
                    abs_perc = get_value((0x0303, 0x101a))  # Absolute Percentile
                    pattern_perc = get_value((0x0303, 0x101c))  # Pattern Percentile

                    # Accumulate sum and count for each (x, y) coordinate
                    coord_value_map[(x, y)]["raw_sum"] += raw_value
                    coord_value_map[(x, y)]["abs_sum"] += abs_value
                    coord_value_map[(x, y)]["pattern_sum"] += pattern_value
                    coord_value_map[(x, y)]["abs_perc_sum"] += abs_perc
                    coord_value_map[(x, y)]["pattern_perc_sum"] += pattern_perc
                    coord_value_map[(x, y)]["count"] += 1  # Track number of occurrences

                # Compute averaged values
                unique_x_coords, unique_y_coords = [], []
                unique_raw_values, unique_abs_values = [], []
                unique_pattern_values, unique_abs_percentile, unique_pattern_percentile = [], [], []

                for (x, y), data in coord_value_map.items():
                    count = data["count"]

                    unique_x_coords.append(x)
                    unique_y_coords.append(y)
                    unique_raw_values.append(data["raw_sum"]) # / count)
                    unique_abs_values.append(data["abs_sum"]) # / count)
                    unique_pattern_values.append(data["pattern_sum"]) # / count)
                    unique_abs_percentile.append(data["abs_perc_sum"] / count)
                    unique_pattern_percentile.append(data["pattern_perc_sum"] / count)

                # Convert to numpy arrays
                unique_x_coords = np.array(unique_x_coords)
                unique_y_coords = np.array(unique_y_coords)
                unique_raw_values = np.nan_to_num(np.array(unique_raw_values, dtype=np.float64), nan=0)
                unique_abs_values = np.nan_to_num(np.array(unique_abs_values, dtype=np.float64), nan=0)
                unique_pattern_values = np.nan_to_num(np.array(unique_pattern_values, dtype=np.float64), nan=0)
                unique_abs_percentile = np.nan_to_num(np.array(unique_abs_percentile, dtype=np.float64), nan=0)
                unique_pattern_percentile = np.nan_to_num(np.array(unique_pattern_percentile, dtype=np.float64), nan=0)

                # Plot setup: 3 rows, 2 columns (first row spans both columns)
                fig, axes = plt.subplots(3, 2, figsize=(12, 16), gridspec_kw={'height_ratios': [1.5, 1, 1]})
                fig.subplots_adjust(hspace=0.4, wspace=0.3)

                # Function to plot text annotations
                def plot_text(ax, values, title):
                    ax.scatter(unique_x_coords, unique_y_coords, color="gray", s=1, alpha=0.1)  # Faint reference points
                    for x, y, val in zip(unique_x_coords, unique_y_coords, values):
                        ax.text(x, y, f"{int(val)}", fontsize=10, ha='center', va='bottom', fontweight='bold')
                    ax.set_xlabel("X Coordinate (Visual Field)")
                    ax.set_ylabel("Y Coordinate (Visual Field)")
                    ax.set_title(title)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.grid(True, linestyle='--', alpha=0.5)

                # First row (full width)
                plot_text(axes[0, 0], unique_raw_values, "Raw Threshold Sensitivity (dB)")
                axes[0, 1].axis("off")  # Disable the second subplot in the first row

                # Second row
                plot_text(axes[1, 0], unique_abs_values, "Absolute Values")
                plot_text(axes[1, 1], unique_pattern_values, "Pattern Deviation")

                # Third row
                plot_text(axes[2, 0], unique_abs_percentile, "Absolute Percentile")
                plot_text(axes[2, 1], unique_pattern_percentile, "Pattern Percentile")

                # plt.savefig(".....09.png")
                # Convert plot to a PIL Image
                # buf = BytesIO()
                # plt.savefig(buf, format="PNG", bbox_inches='tight', pad_inches=0.1)
                # plt.close(fig)
                # buf.seek(0)
                # image = Image.open(buf)  # Convert buffer into a PIL image
                # Extract perimetry data from private tags
                # 'Ophthalmic Photography 8 Bit Image Storage'
                # try:
                #     pixel_array = self.ds.pixel_array
                # except Exception as e:
                #     print("pixel array issue")
                #     print(repr(e))
                # image = Image.fromarray(pixel_array[1, :, :])  # First frame of 252...this might be even more raw
                # metadata['image_PIL'] = image
                self.ds.get("SeriesDescription", "Unknown")
                # Number of Frames
                metadata["Number of Frames"] = self.ds.get("NumberOfFrames", "Unknown")
                # Laterality
                metadata["Laterality"] = self.ds.get("Laterality", "Unknown")
                # Bits Allocated
                metadata["Bits Allocated"] = self.ds.get("BitsAllocated", "Unknown")
                # Photometric Interpretation
                metadata["Photometric Interpretation"] = self.ds.get("PhotometricInterpretation", "Unknown")


        return metadata

    def preview(self, output_path, attempt_to_extract_dicom_tags_not_pixel_datas=False, write_dicom_header=False):
        if write_dicom_header:
            self._write_detailed_dicom_header_to_file(output_path)
        metadata = self.parse()
        # metadata.keys()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.77.1.5.1':
            if not attempt_to_extract_dicom_tags_not_pixel_datas:
                # 'Ophthalmic Tomography Image Storage'
                ## Bscans
                sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}")
                if not os.path.exists(sop_path): os.makedirs(sop_path) # make pdf (png) folder
                for bscan in metadata['bscan_images'].keys():
                    metadata['bscan_images'][bscan].save(os.path.join(sop_path, f"{bscan}.png"))
            elif attempt_to_extract_dicom_tags_not_pixel_datas:
                # A PNG of HVF plots derived from tags...extremely experimental...not sure if it will work
                sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}")
                metadata['image_PIL'].save(os.path.join(output_path, sop_path+".png"))  # To save the image to a file (e.g., PNG format)
        
            


# The String is from the ManufacturerModelName field in the DICOM file
DICOMParser.register_parser("Humphrey Field Analyzer 3", Humphrey_Field_Analyzer_3)


class IOLMaster_700(DICOMParser):
    """Parser for {
        'manufacturer': 'Carl Zeiss Meditec',
        'manufacturermodelname': 'IOLMaster 700',
        'modality': 'OP', ['IOL', 'KER', 'OAM'] - these still need to be developed
           

        'sopclassuiddescription': 'Ophthalmic Photography 8 Bit Image Storage',
           or ['Keratometry Measurements Storage',
               'Multi-frame True Color Secondary Capture Image Storage',
               'Encapsulated PDF Storage',
               'Ophthalmic Axial Measurements Storage',
               'Intraocular Lens Calculations Storage'] - these still need to be developed
        }
    """

    def parse(self):
        metadata = self.extract_common_metadata()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.77.1.5.1':
            # 'Ophthalmic Photography 8 Bit Image Storage'
            try:
                pixel_array = self.ds.pixel_array
            except Exception as e:
                print("pixel array issue")
                print(repr(e))
            image = Image.fromarray(pixel_array)
            metadata['image_PIL'] = image
            # Bits Allocated
            metadata["Bits Allocated"] = self.ds.get("BitsAllocated", "Unknown")
            # Photometric Interpretation
            metadata["Photometric Interpretation"] = self.ds.get("PhotometricInterpretation", "Unknown")
            # Pixel Spacing
            metadata["Pixel Spacing"] = self.ds.get("PixelSpacing", "Unknown")

        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.7.2':
            # Multi-frame True Color Secondary Capture Image Storage"
            try:
                pixel_array = self.ds.pixel_array
            except Exception as e:
                print("pixel array issue")
                print(repr(e))
            bscan_count = pixel_array.shape[0]
            bscan_images = {}
            for i in range(bscan_count):
                bscan_image = Image.fromarray(pixel_array[i, :, :])

                bscan_images[f"bscan{i+1}"] = bscan_image
            metadata['bscan_images'] = bscan_images

        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':
            # 'Encapsulated PDF Storage'
            metadata['png_pages'] = self._parse_pdf_pages()

        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.78.8':
            # 'Intraocular Lens Calculations Storage'
            # Private Tags
            metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
            # ReferencedInstanceSequence
            IntraocularLensCalculationsLeftEyeSequence = []
            for InstanceSequence in self.ds.get("IntraocularLensCalculationsLeftEyeSequence", "Unknown"):
                IOLPowerSequence = []
                for Sequence in InstanceSequence.IOLPowerSequence:
                    SequenceDict = {
                        "Pre-Selected for Implantation":Sequence.PreSelectedForImplantation,
                        "IOL Power":Sequence.IOLPower,
                        "Predicted Refractive Error":Sequence.PredictedRefractiveError,
                        "Implant Part Number":Sequence.ImplantPartNumber,
                    }
                    IOLPowerSequence.append(SequenceDict)
                LensConstantSequence = []
                for Sequence in InstanceSequence.LensConstantSequence:
                    SequenceDict = {
                        "Code Value": Sequence.ConceptNameCodeSequence[0].CodeValue,
                        "Code Meaning": Sequence.ConceptNameCodeSequence[0].CodeMeaning,
                    }
                    LensConstantSequence.append(SequenceDict)
                LensThicknessSequence = []
                for Sequence in InstanceSequence.LensThicknessSequence:
                    Sequence = InstanceSequence.LensThicknessSequence[0]
                    SequenceDict = {
                        "Lens Thickness": Sequence.LensThickness, 
                        "Code Value": Sequence.SourceOfLensThicknessDataCodeSequence[0].CodeValue,
                        "Code Meaning": Sequence.SourceOfLensThicknessDataCodeSequence[0].CodeMeaning,
                    }
                    LensThicknessSequence.append(SequenceDict)
                AnteriorChamberDepthSequence = []
                for Sequence in InstanceSequence.AnteriorChamberDepthSequence:
                    SequenceDict = {
                        "Anterior Chamber Depth": Sequence.AnteriorChamberDepth,
                        "Source of Anterior Chamber Depth Data Code Sequence": {
                            "Code Value": Sequence.SourceOfAnteriorChamberDepthDataCodeSequence[0].CodeValue,
                            "Code Meaning": Sequence.SourceOfAnteriorChamberDepthDataCodeSequence[0].CodeMeaning,
                        }
                    }
                    AnteriorChamberDepthSequence.append(SequenceDict)
                CornealSizeSequence = []
                for Sequence in InstanceSequence.CornealSizeSequence:
                    SequenceDict = {
                        "Corneal Size": Sequence.CornealSize,
                        "Source of Corneal Size Data Code Sequence": {
                            "Code Value": Sequence.SourceOfCornealSizeDataCodeSequence[0].CodeValue,
                            "Code Meaning": Sequence.SourceOfCornealSizeDataCodeSequence[0].CodeMeaning,
                        }
                    }
                    CornealSizeSequence.append(SequenceDict)
                SteepKeratometricAxisSequence = []
                for Sequence in InstanceSequence.SteepKeratometricAxisSequence:
                    SequenceDict = {
                        "Radius of Curvature": Sequence.RadiusOfCurvature,
                        "Keratometric Power": Sequence.KeratometricPower,
                        "Keratometric Axis": Sequence.KeratometricAxis,
                    }
                    SteepKeratometricAxisSequence.append(SequenceDict)
                FlatKeratometricAxisSequence = []
                for Sequence in InstanceSequence.FlatKeratometricAxisSequence:
                    SequenceDict = {
                        "Radius of Curvature": Sequence.RadiusOfCurvature,
                        "Keratometric Power": Sequence.KeratometricPower,
                        "Keratometric Axis": Sequence.KeratometricAxis,
                    }
                    FlatKeratometricAxisSequence.append(SequenceDict)
                
                CorneaMeasurementsSequence = []
                for Sequence in InstanceSequence.CorneaMeasurementsSequence:
                    SteepCornealAxisSequence = []
                    for SteepSequence in Sequence.SteepCornealAxisSequence:
                        SteepSequenceDict = {
                            "Radius of Curvature": SteepSequence.RadiusOfCurvature,
                            "Corneal Power": SteepSequence.CornealPower,
                            "Corneal Axis": SteepSequence.CornealAxis,
                        }
                        SteepCornealAxisSequence.append(SteepSequenceDict)

                    FlatCornealAxisSequence = []
                    for FlatSequence in Sequence.FlatCornealAxisSequence:
                        FlatSequenceDict = {
                            "Radius of Curvature": FlatSequence.RadiusOfCurvature,
                            "Corneal Power": FlatSequence.CornealPower,
                            "Corneal Axis": FlatSequence.CornealAxis,
                        }
                        FlatCornealAxisSequence.append(FlatSequenceDict)

                    CorneaMeasurementMethodCodeSequence = []
                    for MethodSequence in Sequence.CorneaMeasurementMethodCodeSequence:
                        MethodSequenceDict = {
                            "Code Value": MethodSequence.CodeValue,
                            "Code Meaning": MethodSequence.CodeMeaning,
                        }
                        CorneaMeasurementMethodCodeSequence.append(MethodSequenceDict)

                    SequenceDict = {
                        "Keratometer Index": Sequence.KeratometerIndex,
                        "Source of Cornea Measurement Data Code Sequence": {
                            "Code Value": Sequence.SourceOfCorneaMeasurementDataCodeSequence[0].CodeValue,
                            "Code Meaning": Sequence.SourceOfCorneaMeasurementDataCodeSequence[0].CodeMeaning,
                        },
                        "Steep Corneal Axis Sequence": SteepCornealAxisSequence,
                        "Flat Corneal Axis Sequence": FlatCornealAxisSequence,
                        "Cornea Measurement Method Code Sequence": CorneaMeasurementMethodCodeSequence,
                    }
                    CorneaMeasurementsSequence.append(SequenceDict)
                IS = {
                    "Ophthalmic Axial Length Sequence": {
                        "OphthalmicAxialLength": InstanceSequence.OphthalmicAxialLengthSequence[0].OphthalmicAxialLength,
                        "Source of Ophthalmic Axial Length Code Sequence": InstanceSequence.OphthalmicAxialLengthSequence[0].SourceOfOphthalmicAxialLengthCodeSequence[0].CodeMeaning,
                        "Ophthalmic Axial Length Selection Method Code Sequence": InstanceSequence.OphthalmicAxialLengthSequence[0].OphthalmicAxialLengthSelectionMethodCodeSequence[0].CodeMeaning,
                    },
                    "IOL Formula Code Sequence": {
                        "Code Value": InstanceSequence.IOLFormulaCodeSequence[0].CodeValue,
                        "Code Meaning": InstanceSequence.IOLFormulaCodeSequence[0].CodeMeaning,
                    },
                    "Keratometer Index": InstanceSequence.KeratometerIndex,
                    "Target Refraction": InstanceSequence.TargetRefraction,
                    "Refractive Procedure Occurred": InstanceSequence.RefractiveProcedureOccurred,
                    "Surgically Induced Astigmatism Sequence": {
                        "Cylinder Axis": InstanceSequence.SurgicallyInducedAstigmatismSequence[0].CylinderAxis,
                        "Cylinder Power": InstanceSequence.SurgicallyInducedAstigmatismSequence[0].CylinderPower,
                    },
                    "Type of Optical Correction": InstanceSequence.TypeOfOpticalCorrection,
                    "IOL Power Sequence": IOLPowerSequence,
                    "Lens Constant Sequence": LensConstantSequence,
                    "IOL Manufacturer": InstanceSequence.IOLManufacturer,
                    "Implant Name": InstanceSequence.ImplantName,
                    "Keratometry Measurement Type Code Sequence": {
                        "Code Value": InstanceSequence.KeratometryMeasurementTypeCodeSequence[0].CodeValue,
                        "Code Meaning": InstanceSequence.KeratometryMeasurementTypeCodeSequence[0].CodeMeaning,
                        },
                    "IOL Power For Exact Emmetropia": InstanceSequence.IOLPowerForExactEmmetropia,
                    "IOL Power For Exact Target Refraction": InstanceSequence.IOLPowerForExactTargetRefraction,
                    "CornealSizeSequence" : CornealSizeSequence,
                    "AnteriorChamberDepthSequence" : AnteriorChamberDepthSequence,
                    "SteepKeratometricAxisSequence" : SteepKeratometricAxisSequence,
                    "FlatKeratometricAxisSequence" : FlatKeratometricAxisSequence,
                    "CorneaMeasurementsSequence" : CorneaMeasurementsSequence,

                }
                IntraocularLensCalculationsLeftEyeSequence.append(IS)
            metadata["IntraocularLensCalculationsLeftEyeSequence"] = IntraocularLensCalculationsLeftEyeSequence
            metadata["Measurement Laterality"] = self.ds.get("MeasurementLaterality", "Unknown")
            # ReferencedInstanceSequence
            ReferencedRefractiveMeasurementsSequence = []
            for InstanceSequence in self.ds.get("ReferencedRefractiveMeasurementsSequence", "Unknown"):
                IS = {
                    "ReferencedSOPClassUID": OPHTHALMOLOGY_SOP_CLASSES[InstanceSequence.ReferencedSOPClassUID],
                    "ReferencedSOPInstance_UID": InstanceSequence.ReferencedSOPInstanceUID
                }
                ReferencedRefractiveMeasurementsSequence.append(IS)
            metadata["ReferencedRefractiveMeasurementsSequence"] = ReferencedRefractiveMeasurementsSequence
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.78.7':
            # 'Ophthalmic Axial Measurements Right Eye Sequence'
            # Ophthalmic Axial Length Measurements
            OphthalmicAxialMeasurementsRightEyeSequence = []
            for InstanceSequence in self.ds.get("OphthalmicAxialMeasurementsRightEyeSequence", "Unknown"):
                # Pupil Dilated
                PupilDilated = InstanceSequence.PupilDilated#######################
                # Lens Status Code Sequence
                LensStatusCodeSequence = []#######################
                for Sequence in InstanceSequence.LensStatusCodeSequence:
                    SequenceDict = {
                        "Code Value": Sequence.CodeValue,
                        "Coding Scheme Designator": Sequence.CodingSchemeDesignator,
                        "Code Meaning": Sequence.CodeMeaning,
                    }
                    LensStatusCodeSequence.append(SequenceDict)

                # Vitreous Status Code Sequence
                VitreousStatusCodeSequence = []#######################
                for Sequence in InstanceSequence.VitreousStatusCodeSequence:
                    SequenceDict = {
                        "Code Value": Sequence.CodeValue,
                        "Coding Scheme Designator": Sequence.CodingSchemeDesignator,
                        "Code Meaning": Sequence.CodeMeaning,
                    }
                    VitreousStatusCodeSequence.append(SequenceDict)

                # Ophthalmic Axial Length Measurements Sequence
                OphthalmicAxialLengthMeasurementsSequence = []#######################
                Measurement1 = InstanceSequence.OphthalmicAxialLengthMeasurementsSequence[0]
                Measurement2 = InstanceSequence.OphthalmicAxialLengthMeasurementsSequence[1]
                for Length in Measurement1.OphthalmicAxialLengthMeasurementsTotalLengthSequence:
                    # Length = Measurement1.OphthalmicAxialLengthMeasurementsTotalLengthSequence[0]
                    LengthDict = {
                        "Ophthalmic Axial Length": Length.OphthalmicAxialLength,
                        "Ophthalmic Axial Length Measurement ": Length[(0x0022, 0x1140)].value,
                        "Ophthalmic Axial Length Data Source Code Sequence": {
                            "Code Value": Length[(0x0022, 0x1225)][0][(0x0022, 0x1150)][0].CodeValue,
                            "Code Meaning": Length[(0x0022, 0x1225)][0][(0x0022, 0x1150)][0].CodeMeaning,
                        },
                        "Ophthalmic Axial Length Data Source":  Length[(0x0022, 0x1225)][0][(0x0022, 0x1159)].value,
                        "Referenced Ophthalmic Axial Length Measurement QC Image Sequence": {
                            "Referenced SOP Class UID": OPHTHALMOLOGY_SOP_CLASSES[Length[(0x0022, 0x1330)][0][(0x0008, 0x1150)].value],
                            "Referenced SOP Instance UID": Length[(0x0022, 0x1330)][0][(0x0008, 0x1155)].value,
                            "Referenced Frame Number": Length[(0x0022, 0x1330)][0][(0x0008, 0x1160)].value,
                        },
                    }
                OphthalmicAxialLengthMeasurementsSequence.append(LengthDict)

                # Loop through the Ophthalmic Axial Length Measurements Segmental Length Sequence
                for Segment in Measurement2.OphthalmicAxialLengthMeasurementsSegmentalLengthSequence:
                    # Segment = Measurement2.OphthalmicAxialLengthMeasurementsSegmentalLengthSequence[0]#
                    SegmentDict = {
                        "Ophthalmic Axial Length": Segment.OphthalmicAxialLength,
                        "Ophthalmic Axial Length Measurement": Segment[(0x0022, 0x1140)].value,
                        "Ophthalmic Axial Length Measurements Segment Name Code Sequence": {
                            "Code Value": Segment.OphthalmicAxialLengthMeasurementsSegmentNameCodeSequence[0].CodeValue,
                            "Coding Scheme Designator": Segment.OphthalmicAxialLengthMeasurementsSegmentNameCodeSequence[0].CodingSchemeDesignator,
                            "Code Meaning": Segment.OphthalmicAxialLengthMeasurementsSegmentNameCodeSequence[0].CodeMeaning,
                        },
                        "Ophthalmic Axial Length Data Source": Segment[(0x0022, 0x1225)][0][(0x0022, 0x1159)].value,
                    }

                    # Loop through Optical Ophthalmic Axial Length Measurements Sequence
                    for OpticalMeasurement in Segment.OpticalOphthalmicAxialLengthMeasurementsSequence:
                        OpticalMeasurement = Segment.OpticalOphthalmicAxialLengthMeasurementsSequence[0]
                        OpticalMeasurementDict = {
                            "Ophthalmic Axial Length Data Source Code Sequence": {
                                "Code Value": OpticalMeasurement.OphthalmicAxialLengthDataSourceCodeSequence[0].CodeValue,
                                "Coding Scheme Designator": OpticalMeasurement.OphthalmicAxialLengthDataSourceCodeSequence[0].CodingSchemeDesignator,
                                "Code Meaning": OpticalMeasurement.OphthalmicAxialLengthDataSourceCodeSequence[0].CodeMeaning,
                            },
                            "Ophthalmic Axial Length Data Source LO": OpticalMeasurement[(0x0022, 0x1159)].value
                        }
                        SegmentDict["Optical Ophthalmic Axial Length Measurements Sequence"] = OpticalMeasurementDict

                    # Append the segment data to the main list
                OphthalmicAxialLengthMeasurementsSequence.append(SegmentDict)

                # Optical Selected Ophthalmic Axial Length Sequence
                OpticalSelectedOphthalmicAxialLengthSequence = []#######################
                for LengthSequence in InstanceSequence.OpticalSelectedOphthalmicAxialLengthSequence:
                    LengthSequence = InstanceSequence.OpticalSelectedOphthalmicAxialLengthSequence[0]
                    LengthSequenceDict = {
                        "Ophthalmic Axial Length Measurement": LengthSequence[(0x0022, 0x1010)].value,
                        "Ophthalmic Axial Length": LengthSequence[(0x0022, 0x1260)][0][(0x0022, 0x1019)].value
                    }
                    OphthalmicAxialLengthQualityMetricSequence = []
                    for QMS in LengthSequence[(0x0022, 0x1260)][0][(0x0022, 0x1262)]:
                        QMS = LengthSequence[(0x0022, 0x1260)][0][(0x0022, 0x1262)][0]
                        QMSDict = {
                            "CodeValue": QMS[(0x0040, 0x08ea)][0].CodeValue,
                            "CodeMeaning": QMS[(0x0040, 0x08ea)][0].CodeMeaning,
                        }
                        OphthalmicAxialLengthQualityMetricSequence.append(QMSDict)
                    LengthSequenceDict['Ophthalmic Axial Length Quality Metric Sequence'] = OphthalmicAxialLengthQualityMetricSequence
                    OpticalSelectedOphthalmicAxialLengthSequence.append(LengthSequenceDict)   
                  
                # Finalizing the instance sequence
                IS = {
                    "Pupil Dilated": PupilDilated,
                    "Lens Status Code Sequence": LensStatusCodeSequence,
                    "Vitreous Status Code Sequence": VitreousStatusCodeSequence,
                    "Ophthalmic Axial Length Measurements Sequence": OphthalmicAxialLengthMeasurementsSequence,
                    "Optical Selected Ophthalmic Axial Length Sequence": OpticalSelectedOphthalmicAxialLengthSequence,
                }

                OphthalmicAxialMeasurementsRightEyeSequence.append(IS)
            #--------------------------
            OphthalmicAxialMeasurementsLeftEyeSequence = []
            for InstanceSequence in self.ds.get("OphthalmicAxialMeasurementsLeftEyeSequence", "Unknown"):
                # Pupil Dilated
                PupilDilated = InstanceSequence.PupilDilated#######################
                # Lens Status Code Sequence
                LensStatusCodeSequence = []#######################
                for Sequence in InstanceSequence.LensStatusCodeSequence:
                    SequenceDict = {
                        "Code Value": Sequence.CodeValue,
                        "Coding Scheme Designator": Sequence.CodingSchemeDesignator,
                        "Code Meaning": Sequence.CodeMeaning,
                    }
                    LensStatusCodeSequence.append(SequenceDict)

                # Vitreous Status Code Sequence
                VitreousStatusCodeSequence = []#######################
                for Sequence in InstanceSequence.VitreousStatusCodeSequence:
                    SequenceDict = {
                        "Code Value": Sequence.CodeValue,
                        "Coding Scheme Designator": Sequence.CodingSchemeDesignator,
                        "Code Meaning": Sequence.CodeMeaning,
                    }
                    VitreousStatusCodeSequence.append(SequenceDict)

                # Ophthalmic Axial Length Measurements Sequence
                OphthalmicAxialLengthMeasurementsSequence = []#######################
                Measurement1 = InstanceSequence.OphthalmicAxialLengthMeasurementsSequence[0]
                Measurement2 = InstanceSequence.OphthalmicAxialLengthMeasurementsSequence[1]
                for Length in Measurement1.OphthalmicAxialLengthMeasurementsTotalLengthSequence:
                    # Length = Measurement1.OphthalmicAxialLengthMeasurementsTotalLengthSequence[0]
                    LengthDict = {
                        "Ophthalmic Axial Length": Length.OphthalmicAxialLength,
                        "Ophthalmic Axial Length Measurement ": Length[(0x0022, 0x1140)].value,
                        "Ophthalmic Axial Length Data Source Code Sequence": {
                            "Code Value": Length[(0x0022, 0x1225)][0][(0x0022, 0x1150)][0].CodeValue,
                            "Code Meaning": Length[(0x0022, 0x1225)][0][(0x0022, 0x1150)][0].CodeMeaning,
                        },
                        "Ophthalmic Axial Length Data Source":  Length[(0x0022, 0x1225)][0][(0x0022, 0x1159)].value,
                        "Referenced Ophthalmic Axial Length Measurement QC Image Sequence": {
                            "Referenced SOP Class UID": OPHTHALMOLOGY_SOP_CLASSES[Length[(0x0022, 0x1330)][0][(0x0008, 0x1150)].value],
                            "Referenced SOP Instance UID": Length[(0x0022, 0x1330)][0][(0x0008, 0x1155)].value,
                            "Referenced Frame Number": Length[(0x0022, 0x1330)][0][(0x0008, 0x1160)].value,
                        },
                    }
                OphthalmicAxialLengthMeasurementsSequence.append(LengthDict)

                # Loop through the Ophthalmic Axial Length Measurements Segmental Length Sequence
                for Segment in Measurement2.OphthalmicAxialLengthMeasurementsSegmentalLengthSequence:
                    # Segment = Measurement2.OphthalmicAxialLengthMeasurementsSegmentalLengthSequence[0]#
                    SegmentDict = {
                        "Ophthalmic Axial Length": Segment.OphthalmicAxialLength,
                        "Ophthalmic Axial Length Measurement": Segment[(0x0022, 0x1140)].value,
                        "Ophthalmic Axial Length Measurements Segment Name Code Sequence": {
                            "Code Value": Segment.OphthalmicAxialLengthMeasurementsSegmentNameCodeSequence[0].CodeValue,
                            "Coding Scheme Designator": Segment.OphthalmicAxialLengthMeasurementsSegmentNameCodeSequence[0].CodingSchemeDesignator,
                            "Code Meaning": Segment.OphthalmicAxialLengthMeasurementsSegmentNameCodeSequence[0].CodeMeaning,
                        },
                        "Ophthalmic Axial Length Data Source": Segment[(0x0022, 0x1225)][0][(0x0022, 0x1159)].value,
                    }

                    # Loop through Optical Ophthalmic Axial Length Measurements Sequence
                    for OpticalMeasurement in Segment.OpticalOphthalmicAxialLengthMeasurementsSequence:
                        OpticalMeasurement = Segment.OpticalOphthalmicAxialLengthMeasurementsSequence[0]
                        OpticalMeasurementDict = {
                            "Ophthalmic Axial Length Data Source Code Sequence": {
                                "Code Value": OpticalMeasurement.OphthalmicAxialLengthDataSourceCodeSequence[0].CodeValue,
                                "Coding Scheme Designator": OpticalMeasurement.OphthalmicAxialLengthDataSourceCodeSequence[0].CodingSchemeDesignator,
                                "Code Meaning": OpticalMeasurement.OphthalmicAxialLengthDataSourceCodeSequence[0].CodeMeaning,
                            },
                            "Ophthalmic Axial Length Data Source LO": OpticalMeasurement[(0x0022, 0x1159)].value
                        }
                        SegmentDict["Optical Ophthalmic Axial Length Measurements Sequence"] = OpticalMeasurementDict

                    # Append the segment data to the main list
                OphthalmicAxialLengthMeasurementsSequence.append(SegmentDict)

                # Optical Selected Ophthalmic Axial Length Sequence
                OpticalSelectedOphthalmicAxialLengthSequence = []#######################
                for LengthSequence in InstanceSequence.OpticalSelectedOphthalmicAxialLengthSequence:
                    LengthSequence = InstanceSequence.OpticalSelectedOphthalmicAxialLengthSequence[0]
                    LengthSequenceDict = {
                        "Ophthalmic Axial Length Measurement": LengthSequence[(0x0022, 0x1010)].value,
                        "Ophthalmic Axial Length": LengthSequence[(0x0022, 0x1260)][0][(0x0022, 0x1019)].value
                    }
                    OphthalmicAxialLengthQualityMetricSequence = []
                    for QMS in LengthSequence[(0x0022, 0x1260)][0][(0x0022, 0x1262)]:
                        QMS = LengthSequence[(0x0022, 0x1260)][0][(0x0022, 0x1262)][0]
                        QMSDict = {
                            "CodeValue": QMS[(0x0040, 0x08ea)][0].CodeValue,
                            "CodeMeaning": QMS[(0x0040, 0x08ea)][0].CodeMeaning,
                        }
                        OphthalmicAxialLengthQualityMetricSequence.append(QMSDict)
                    LengthSequenceDict['Ophthalmic Axial Length Quality Metric Sequence'] = OphthalmicAxialLengthQualityMetricSequence
                    OpticalSelectedOphthalmicAxialLengthSequence.append(LengthSequenceDict)   
                  
                # Finalizing the instance sequence
                IS = {
                    "Pupil Dilated": PupilDilated,
                    "Lens Status Code Sequence": LensStatusCodeSequence,
                    "Vitreous Status Code Sequence": VitreousStatusCodeSequence,
                    "Ophthalmic Axial Length Measurements Sequence": OphthalmicAxialLengthMeasurementsSequence,
                    "Optical Selected Ophthalmic Axial Length Sequence": OpticalSelectedOphthalmicAxialLengthSequence,
                }

                OphthalmicAxialMeasurementsLeftEyeSequence.append(IS)


            metadata["OphthalmicAxialMeasurementsRightEyeSequence"] = OphthalmicAxialMeasurementsRightEyeSequence
            metadata["OphthalmicAxialMeasurementsLeftEyeSequence"] = OphthalmicAxialMeasurementsLeftEyeSequence
            metadata["Measurement Laterality"] = self.ds.get("MeasurementLaterality", "Unknown")
            
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.78.3':
            # 'Keratometry Measurements Storage'
            # Private Tags
            metadata['(0x2201, 0x1000)'] = ''.join([i for i in self.ds[(0x2201,0x1000)]])
            metadata['(0x2201, 0x1002)'] = ''.join([i for i in self.ds[(0x2201,0x1002)]])
            metadata["Keratometry Right Eye Sequence"] = {
                "Steep Keratometric Axis Sequence": {
                    "RadiusOfCurvature": self.ds.get("KeratometryRightEyeSequence", "Unknown")[0].SteepKeratometricAxisSequence[0].RadiusOfCurvature,
                    "KeratometricPower ": self.ds.get("KeratometryRightEyeSequence", "Unknown")[0].SteepKeratometricAxisSequence[0].KeratometricPower,
                    "KeratometricAxis": self.ds.get("KeratometryRightEyeSequence", "Unknown")[0].SteepKeratometricAxisSequence[0].KeratometricAxis,

                },
                "Flat Keratometric Axis Sequence": {
                    "RadiusOfCurvature": self.ds.get("KeratometryRightEyeSequence", "Unknown")[0].SteepKeratometricAxisSequence[0].RadiusOfCurvature,
                    "KeratometricPower ": self.ds.get("KeratometryRightEyeSequence", "Unknown")[0].SteepKeratometricAxisSequence[0].KeratometricPower,
                    "KeratometricAxis": self.ds.get("KeratometryRightEyeSequence", "Unknown")[0].SteepKeratometricAxisSequence[0].KeratometricAxis,
                },
            }
            metadata["Keratometry Left Eye Sequence"] = {
                "Steep Keratometric Axis Sequence": {
                    "RadiusOfCurvature": self.ds.get("KeratometryLeftEyeSequence", "Unknown")[0].SteepKeratometricAxisSequence[0].RadiusOfCurvature,
                    "KeratometricPower ": self.ds.get("KeratometryLeftEyeSequence", "Unknown")[0].SteepKeratometricAxisSequence[0].KeratometricPower,
                    "KeratometricAxis": self.ds.get("KeratometryLeftEyeSequence", "Unknown")[0].SteepKeratometricAxisSequence[0].KeratometricAxis,

                },
                "Flat Keratometric Axis Sequence": {
                    "RadiusOfCurvature": self.ds.get("KeratometryLeftEyeSequence", "Unknown")[0].SteepKeratometricAxisSequence[0].RadiusOfCurvature,
                    "KeratometricPower ": self.ds.get("KeratometryLeftEyeSequence", "Unknown")[0].SteepKeratometricAxisSequence[0].KeratometricPower,
                    "KeratometricAxis": self.ds.get("KeratometryLeftEyeSequence", "Unknown")[0].SteepKeratometricAxisSequence[0].KeratometricAxis,
                },
            }
            metadata['(0x1201, 0x1001)'] = {
                "(0x1201, 0x1003)": self.ds[((0x1201, 0x1001))][0][((0x1201, 0x1003))][0][((0x1201, 0x1005))].value,
                "(0x1201, 0x1004)": self.ds[((0x1201, 0x1001))][0][((0x1201, 0x1004))][0][((0x1201, 0x1005))].value,
                "(0x1201, 0x1006)": self.ds[((0x1201, 0x1001))][0][((0x1201, 0x1006))].value,
                "(0x1201, 0x1007)": self.ds[((0x1201, 0x1001))][0][((0x1201, 0x1007))].value,
                "(0x1201, 0x101d)": OPHTHALMOLOGY_SOP_CLASSES[self.ds[((0x1201, 0x1001))][0][((0x1201, 0x101d))][0][((0x1201, 0x101e))].value],

            }
            metadata['(0x1201, 0x1002)'] = {
                "(0x1201, 0x1003)": self.ds[((0x1201, 0x1002))][0][((0x1201, 0x1003))][0][((0x1201, 0x1005))].value,
                "(0x1201, 0x1004)": self.ds[((0x1201, 0x1002))][0][((0x1201, 0x1004))][0][((0x1201, 0x1005))].value,
                "(0x1201, 0x1006)": self.ds[((0x1201, 0x1002))][0][((0x1201, 0x1006))].value,
                "(0x1201, 0x1007)": self.ds[((0x1201, 0x1002))][0][((0x1201, 0x1007))].value,
                "(0x1201, 0x101d)": OPHTHALMOLOGY_SOP_CLASSES[self.ds[((0x1201, 0x1002))][0][((0x1201, 0x101d))][0][((0x1201, 0x101e))].value],

            }
            metadata['(0x1201, 0x1008)'] = {
                "(0x1201, 0x1007)": self.ds[((0x1201, 0x1008))][0][((0x1201, 0x1007))].value,
                "(0x1201, 0x100a)": {
                    "(0x1201, 0x1005)": self.ds[((0x1201, 0x1008))][0][((0x1201, 0x100a))][0][((0x1201, 0x1005))].value,
                    "(0x1201, 0x100c)": self.ds[((0x1201, 0x1008))][0][((0x1201, 0x100a))][0][((0x1201, 0x100c))].value,
                    "(0x1201, 0x100d)": self.ds[((0x1201, 0x1008))][0][((0x1201, 0x100a))][0][((0x1201, 0x100d))].value,
                    "(0x1201, 0x100e)": self.ds[((0x1201, 0x1008))][0][((0x1201, 0x100a))][0][((0x1201, 0x100e))].value,
                },
                "(0x1201, 0x100b)": {
                    "(0x1201, 0x1005)": self.ds[((0x1201, 0x1008))][0][((0x1201, 0x100b))][0][((0x1201, 0x1005))].value,
                    "(0x1201, 0x100c)": self.ds[((0x1201, 0x1008))][0][((0x1201, 0x100b))][0][((0x1201, 0x100c))].value,
                    "(0x1201, 0x100d)": self.ds[((0x1201, 0x1008))][0][((0x1201, 0x100b))][0][((0x1201, 0x100d))].value,
                    "(0x1201, 0x100e)": self.ds[((0x1201, 0x1008))][0][((0x1201, 0x100b))][0][((0x1201, 0x100e))].value,
                },
                "(0x1201, 0x101b)": self.ds[((0x1201, 0x1008))][0][((0x1201, 0x101b))].value,
                "(0x1201, 0x101c)": self.ds[((0x1201, 0x1008))][0][((0x1201, 0x101c))].value,
            }
            metadata['(0x1201, 0x1009)'] = {
                "(0x1201, 0x1007)": self.ds[((0x1201, 0x1009))][0][((0x1201, 0x1007))].value,
                "(0x1201, 0x100a)": {
                    "(0x1201, 0x1005)": self.ds[((0x1201, 0x1009))][0][((0x1201, 0x100a))][0][((0x1201, 0x1005))].value,
                    "(0x1201, 0x100c)": self.ds[((0x1201, 0x1009))][0][((0x1201, 0x100a))][0][((0x1201, 0x100c))].value,
                    "(0x1201, 0x100d)": self.ds[((0x1201, 0x1009))][0][((0x1201, 0x100a))][0][((0x1201, 0x100d))].value,
                    "(0x1201, 0x100e)": self.ds[((0x1201, 0x1009))][0][((0x1201, 0x100a))][0][((0x1201, 0x100e))].value,
                },
                "(0x1201, 0x100b)": {
                    "(0x1201, 0x1005)": self.ds[((0x1201, 0x1009))][0][((0x1201, 0x100b))][0][((0x1201, 0x1005))].value,
                    "(0x1201, 0x100c)": self.ds[((0x1201, 0x1009))][0][((0x1201, 0x100b))][0][((0x1201, 0x100c))].value,
                    "(0x1201, 0x100d)": self.ds[((0x1201, 0x1009))][0][((0x1201, 0x100b))][0][((0x1201, 0x100d))].value,
                    "(0x1201, 0x100e)": self.ds[((0x1201, 0x1009))][0][((0x1201, 0x100b))][0][((0x1201, 0x100e))].value,
                },
                "(0x1201, 0x101b)": self.ds[((0x1201, 0x1009))][0][((0x1201, 0x101b))].value,
                "(0x1201, 0x101c)": self.ds[((0x1201, 0x1009))][0][((0x1201, 0x101c))].value,
            }
            metadata['(0x1201, 0x100f)'] = {
                "(0x1201, 0x1006)": self.ds[((0x1201, 0x100f))][0][((0x1201, 0x1006))].value,
                "(0x1201, 0x1011)": {
                    "(0x1201, 0x1013)": self.ds[((0x1201, 0x100f))][0][((0x1201, 0x1011))][0][((0x1201, 0x1013))].value,
                    "(0x1201, 0x1014)": self.ds[((0x1201, 0x100f))][0][((0x1201, 0x1011))][0][((0x1201, 0x1014))].value,
                    "(0x1201, 0x1015)": self.ds[((0x1201, 0x100f))][0][((0x1201, 0x1011))][0][((0x1201, 0x1015))].value,
                    "(0x1201, 0x1016)": self.ds[((0x1201, 0x100f))][0][((0x1201, 0x1011))][0][((0x1201, 0x1016))].value,
                },
                "(0x1201, 0x1012)": {
                    "(0x1201, 0x1013)": self.ds[((0x1201, 0x100f))][0][((0x1201, 0x1012))][0][((0x1201, 0x1013))].value,
                    "(0x1201, 0x1014)": self.ds[((0x1201, 0x100f))][0][((0x1201, 0x1012))][0][((0x1201, 0x1014))].value,
                    "(0x1201, 0x1015)": self.ds[((0x1201, 0x100f))][0][((0x1201, 0x1012))][0][((0x1201, 0x1015))].value,
                    "(0x1201, 0x1016)": self.ds[((0x1201, 0x100f))][0][((0x1201, 0x1012))][0][((0x1201, 0x1016))].value,
                },
                "(0x1201, 0x1017)": self.ds[((0x1201, 0x100f))][0][((0x1201, 0x1017))].value,
            }
            metadata['(0x1201, 0x1010)'] = {
                "(0x1201, 0x1006)": self.ds[((0x1201, 0x1010))][0][((0x1201, 0x1006))].value,
                "(0x1201, 0x1011)": {
                    "(0x1201, 0x1013)": self.ds[((0x1201, 0x1010))][0][((0x1201, 0x1011))][0][((0x1201, 0x1013))].value,
                    "(0x1201, 0x1014)": self.ds[((0x1201, 0x1010))][0][((0x1201, 0x1011))][0][((0x1201, 0x1014))].value,
                    "(0x1201, 0x1015)": self.ds[((0x1201, 0x1010))][0][((0x1201, 0x1011))][0][((0x1201, 0x1015))].value,
                    "(0x1201, 0x1016)": self.ds[((0x1201, 0x1010))][0][((0x1201, 0x1011))][0][((0x1201, 0x1016))].value,
                },
                "(0x1201, 0x1012)": {
                    "(0x1201, 0x1013)": self.ds[((0x1201, 0x1010))][0][((0x1201, 0x1012))][0][((0x1201, 0x1013))].value,
                    "(0x1201, 0x1014)": self.ds[((0x1201, 0x1010))][0][((0x1201, 0x1012))][0][((0x1201, 0x1014))].value,
                    "(0x1201, 0x1015)": self.ds[((0x1201, 0x1010))][0][((0x1201, 0x1012))][0][((0x1201, 0x1015))].value,
                    "(0x1201, 0x1016)": self.ds[((0x1201, 0x1010))][0][((0x1201, 0x1012))][0][((0x1201, 0x1016))].value,
                },
                "(0x1201, 0x1017)": self.ds[((0x1201, 0x1010))][0][((0x1201, 0x1017))].value,
            }
            metadata['(0x1201, 0x1018)'] = f"{self.ds[(0x1201, 0x1018)][0][(0x1201, 0x101a)].VR}: Array of {len(self.ds[(0x1201, 0x1018)][0][(0x1201, 0x101a)].value)} elements"
            metadata['(0x1201, 0x1019)'] = f"{self.ds[(0x1201, 0x1019)][0][(0x1201, 0x101a)].VR}: Array of {len(self.ds[(0x1201, 0x1019)][0][(0x1201, 0x101a)].value)} elements"

            metadata['(0x1203, 0x1001)'] = self.ds[(0x1203, 0x1001)][0][(0x1203, 0x100a)][0][(0x1203, 0x100b)].value
            metadata['(0x1203, 0x1002)'] = self.ds[(0x1203, 0x1002)][0][(0x1203, 0x100a)][0][(0x1203, 0x100b)].value

        return metadata

    def preview(self, output_path, write_dicom_header=False):
        if write_dicom_header:
            self._write_detailed_dicom_header_to_file(output_path)
        metadata = self.parse()
        # metadata.keys()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.77.1.5.1':
            sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}")
            metadata['image_PIL'].save(os.path.join(output_path, sop_path+".png"))  # To save the image to a file (e.g., PNG format)
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.7.2':
            # Multi-frame True Color Secondary Capture Image Storage"
            ## Bscans
            sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}")
            if not os.path.exists(sop_path): os.makedirs(sop_path) # make pdf (png) folder
            for bscan in metadata['bscan_images'].keys():
                metadata['bscan_images'][bscan].save(os.path.join(sop_path, f"{bscan}.png"))
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':
            # 'Encapsulated PDF Storage'
            self._preview_pdf_pages(output_path, metadata)
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.78.8':
            # 'Intraocular Lens Calculations Storage'
            with open(os.path.join(output_path, f"{metadata['SOP Instance']}.json"), "w") as file:
                sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}.json")
                file.write(json.dumps(metadata, indent=4))
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.78.7':
            # 'Ophthalmic Axial Measurements Right Eye Sequence'
            with open(os.path.join(output_path, f"{metadata['SOP Instance']}.json"), "w") as file:
                sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}.json")
                file.write(json.dumps(metadata, indent=4))
        elif metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.78.3':
            # 'Keratometry Measurements Storage'
            with open(os.path.join(output_path, f"{metadata['SOP Instance']}.json"), "w") as file:
                sop_path = os.path.join(output_path, f"{metadata['SOP Instance']}.json")
                file.write(json.dumps(metadata, indent=4))

# The String is from the ManufacturerModelName field in the DICOM file
DICOMParser.register_parser("IOLMaster 700", IOLMaster_700)



class PLEX_ELITE_PE9000(DICOMParser):
    """Parser for {
        'manufacturer': 'Carl Zeiss Meditec',
        'manufacturermodelname': 'PLEX ELITE PE9000',
        'modality': 'OPT',
        'sopclassuiddescription': 'Encapsulated PDF Storage']
        }
    """

    def parse(self):
        metadata = self.extract_common_metadata()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':
            # 'Encapsulated PDF Storage'
            metadata['png_pages'] = self._parse_pdf_pages()
            
        return metadata

    def preview(self, output_path, write_dicom_header=False):
        if write_dicom_header:
            self._write_detailed_dicom_header_to_file(output_path)
        metadata = self.parse()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':
            # 'Encapsulated PDF Storage'
            self._preview_pdf_pages(output_path, metadata)


# The String is from the ManufacturerModelName field in the DICOM file
DICOMParser.register_parser("PLEX ELITE PE9000", PLEX_ELITE_PE9000)



class Retina_Workplace(DICOMParser):
    """Parser for {
        'manufacturer': 'Carl Zeiss Meditec',
        'manufacturermodelname': 'Retina Workplace',
        'modality': 'OPT',
        'sopclassuiddescription': 'Encapsulated PDF Storage']
        }
    """

    def parse(self):
        metadata = self.extract_common_metadata()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':
            # 'Encapsulated PDF Storage'
            metadata['png_pages'] = self._parse_pdf_pages()
            
        return metadata

    def preview(self, output_path, write_dicom_header=False):
        if write_dicom_header:
            self._write_detailed_dicom_header_to_file(output_path)
        metadata = self.parse()
        if metadata['SOP Class'] == '1.2.840.10008.5.1.4.1.1.104.1':
            # 'Encapsulated PDF Storage'
            self._preview_pdf_pages(output_path, metadata)


# The String is from the ManufacturerModelName field in the DICOM file
DICOMParser.register_parser("Retina Workplace", Retina_Workplace)


class TopconIMAGEnetOCTParser(DICOMParser):
    def parse(self):
        metadata = self.extract_common_metadata()
        # Get dicom into oct_converter format
        file = Dicom(self.dicom_path)
        # Extract OCT Volume
        oct_volume = (
            file.read_oct_volume()
        )  # returns an OCT volume with additional metadata if available
        # oct_volume.volume.shape is (n_slices, h, w)
        # Get B Scan Images
        bscan_imgs = self.get_bscan_images_from_pixel_array(oct_volume.volume)
        # Set metadata
        metadata['bscan_images'] = bscan_imgs

        return metadata

    def preview(self, output_path, write_dicom_header=False):
        if write_dicom_header:
            self._write_detailed_dicom_header_to_file(output_path)
        metadata = self.parse()
        # TODO: add logic to determine what to do
        self.save_bscan_images(meta=metadata, output_pth=output_path)

DICOMParser.register_parser("3DOCT-1Maestro2", TopconIMAGEnetOCTParser)
