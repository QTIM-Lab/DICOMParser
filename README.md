# DICOMParser

* For CU Ophthalmology Images
* Convert PACS images to PNG or other human readable format

## DICOMParser
```bash
cd DICOMParser
pyenv activate dicom_parser
poetry install --no-root
```

```python
from DICOMParser import DICOMParser

dicom_file = '.dcm'
parser = DICOMParser.create_parser(dicom_file) # Factory method selects subclass
parser.preview('path_to_output_preview')
parser.preview('path_to_output_preview', write_dicom_header=True) # If in addition to preview best guess, you want entire DICOM at `path_to_output_preview`
```