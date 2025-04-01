from DICOMParser import DICOMParser
import argparse


def parse_args():
    """Parse command line arguments for input file and output folder."""
    parser = argparse.ArgumentParser(description='Process input file and output folder.')

    parser.add_argument('--input_file', '-i', required=True, 
                        help='Path to the input file')
    # Add required arguments
    parser.add_argument('--output_folder', '-o', required=True, 
                        help='Path to the output folder')

    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Print the folder and file names
    print(f"Input file: {args.input_file}")
    print(f"Output folder: {args.output_folder}")
    dicom_file = args.input_file
    output_folder = args.output_folder
    parser = DICOMParser.create_parser(dicom_file) # Factory method selects subclass
    parser.preview(output_folder)
    print('Done')

if __name__ == "__main__":
    main()
