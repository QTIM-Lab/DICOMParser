import pytest
import numpy as np
from PIL import Image
from dicomparser.DICOMParser import DICOMParser

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("pixel_arr, expected_keys, expected_shapes", [
    # Test case 1: Single B-scan, 2x2 image
    (np.array([[[1, 2],
                [3, 4]]], dtype=np.uint8),
     ["bscan1"],
     [(2, 2)]),

    # Test case 2: Two B-scans, 3x3 images
    (np.array([[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],
               [[9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]]], dtype=np.uint8),
     ["bscan1", "bscan2"],
     [(3, 3), (3, 3)]),

    # Test case 3: Empty input (0 B-scans)
    (np.empty((0, 2, 2), dtype=np.uint8),
     [],
     []),
])
def test_get_bscan_images_from_pixel_array(pixel_arr, expected_keys, expected_shapes):
    """
    Test the conversion of a 3D numpy array to a dictionary of PIL Images.
    """
    dicom_path = '/persist/QTIM/Active/23-0284/dashboard/Data/topcon_samples_from_steve_03_27_2025/OCT/100100_10142024_151925_OP_R_001.dcm'
    dicom_parser = DICOMParser(dicom_path=dicom_path)
    # Act
    result = dicom_parser.get_bscan_images_from_pixel_array(pixel_arr)

    # Assert
    assert list(result.keys()) == expected_keys
    assert all(isinstance(result[k], Image.Image) for k in expected_keys)
    assert [result[k].size for k in expected_keys] == [(w, h) for h, w in expected_shapes]
