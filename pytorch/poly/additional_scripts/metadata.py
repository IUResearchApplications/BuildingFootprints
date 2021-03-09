import exiftool
import glob
import sys
import os

def run_metadata(raw_labels, predictions):
    # Make sure the file paths exist
    if not os.path.isdir(raw_labels):
        sys.exit("Error: Main file path to the raw labels does not exist")

    if not os.path.isdir(predictions):
        sys.exit("Error: Main file path to the predictions does not exist")

    # Grab the file paths to all of the images in the directories
    raw_labels_fp = glob.glob(os.path.join(raw_labels, '*.tif'))
    predictions_fp = glob.glob(os.path.join(predictions, '*.tif'))

    # Sort them to make sure the metadata is reattached to the correct images
    raw_labels_fp.sort()
    predictions_fp.sort()

    # Find the number of predictions
    num_img = len(predictions_fp)

    print ("Reattaching metadata...")
    # Set up exiftool
    with exiftool.ExifTool() as et:
        for i in range(num_img):
            # Options and file names used need to be in bytes
            # -TagsFromFile is used to copy metadata from one file to another
            # -overwrite_original overwrites the original file rather than create a new one, so it
            # will output file.tif rather than file.tif and file.original_tif
            # raw_labels_fp[i] is where the metadata is being copied from
            # -exif:all means only the exif tags will be copied
            # --GDALMetadata will prevent the program from copying that exif tag
            # predictions_fp[i] is where the metadata is being copied two
            et.execute(b'-overwrite_original', b'-TagsFromFile', raw_labels_fp[i].encode('utf-8'),
                           b'-exif:all', b'--GDALMetadata', predictions_fp[i].encode('utf-8'))
    print ("Done.")

def main():
    # File path to the directory with all of the test images that still have the metadata attached
    raw_labels = '/file/path/to/raw_test_images'

    # File path to the directory with all of the predictions
    predictions = '/file/path/to/predictions'

    run_metadata(raw_labels, predictions)

if __name__ == '__main__':
    main()

