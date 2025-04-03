# DICOMParser

* For CU Ophthalmology Images
* Convert PACS images to PNG or other human readable format

## DICOMParser
```bash
cd DICOMParser
pyenv activate dicom_parser
poetry install --no-root
```

Parse:
```python
from dicomparser import DICOMParser
dicom_file = '.dcm'
parser = DICOMParser.create_parser(dicom_file) # Factory method selects subclass
```
```python
>>> parser
<DICOMParser.CIRRUS_HD_OCT_6000 object at 0x7a0527163c50>
```
```
metadata = parser.parse()
```
```python
>>> metadata.keys()
dict_keys(['Manufacturer', 'Patient ID', 'Model', 'Modality', 'Study Date', 'SOP Class', 'SOP Class Description', 'SOP Instance', 'Series Description', 'png_pages'])
```

Preview:
```
parser.preview('path_to_output_preview')
parser.preview('path_to_output_preview', write_dicom_header=True) # If in addition to preview best guess, you want entire DICOM at `path_to_output_preview`
```

## DICOMParser scripts

### preview.py

```sh
python preview.py \
    --input_file file.dcm \
    --output_folder path/to/output

```