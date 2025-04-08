import pytest
import numpy as np
import os
from PIL import Image
from dicomparser.DICOMParser import DICOMParser

@pytest.fixture
def dicom_parser():
    dicom_path = '/persist/QTIM/Active/23-0284/dashboard/Data/topcon_samples_from_steve_03_27_2025/OCT/100100_10142024_151925_OP_R_001.dcm'
    return DICOMParser(dicom_path=dicom_path)

@pytest.mark.parametrize("bscan_count, img_shape", [
    (1, (10, 10)),
    (3, (5, 5)),
    (19, (8, 8))
])
def test_save_bscan_images(dicom_parser, tmp_path, bscan_count, img_shape):
    """
    Test that B-scan images are correctly saved to disk and match the original image content.
    """
    # Create dummy pixel data and images
    bscan_images = {}
    original_arrays = []

    for i in range(bscan_count):
        arr = np.random.randint(0, 255, img_shape, dtype=np.uint8)
        original_arrays.append(arr)
        bscan_images[f"bscan{i+1}"] = Image.fromarray(arr)

    meta = {
        "SOP Instance": "TEST123",
        "bscan_images": bscan_images
    }

    # Use temporary directory for output
    dicom_parser.save_bscan_images(meta, tmp_path)

    # Check that all images were saved correctly
    saved_dir = os.path.join(tmp_path, "TEST123")
    assert os.path.isdir(saved_dir)

    for i in range(bscan_count):
        fname = f"bscan{i+1}.png"
        fpath = os.path.join(saved_dir, fname)
        assert os.path.exists(fpath)

        # Open saved image and compare pixel values
        saved_img = Image.open(fpath).convert("L")  # Ensure grayscale
        saved_arr = np.array(saved_img)
        np.testing.assert_array_equal(saved_arr, original_arrays[i])
