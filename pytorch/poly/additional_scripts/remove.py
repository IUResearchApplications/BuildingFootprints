import glob
import os
import numpy as np
import sys
from PIL import Image

def create_dir(dir_name):
    # If the directory does not exist then create it
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name) 

def remove_metadata(input_path, output, new_dir, rgb):

    # Set up the directories to store the stripped images in
    if new_dir:
        create_dir(output)

    if not os.path.isdir(input_path):
        sys.exit("Error: Main file path to input directory does not exist")

    if not os.path.isdir(output):
        sys.exit("Error: Main file path to output directory does not exist")

    input_files = glob.glob(os.path.join(input_path, '*.tif'))

    # Strip the metadata off of all the images and save them
    print ("Stripping metadata...")
    for i in input_files:
        bs_name = os.path.basename(i)
        if rgb:
            img_array = np.array(Image.open(i))[:,:,0:3]
        else:
            img_array = np.array(Image.open(i))

        img = Image.fromarray(img_array)
        img.save(os.path.join(output, bs_name))
        img.close()

    print ("Done.")

def main():
    # Path to the directory where the files with the metadata attached are
    input_path = '/file/path/to/metadata'

    # Path to the directory where the stripped images should be saved
    output = '/file/path/to/stripped_images'

    # Indicate if you want to create a new directory if one hasn't been been created yet to save
    # the stripped images to.
    new_dir = False

    # Indicate if you are removing metadata from RGB images or binary images
    # True for RGB, false for binary
    rgb = True

    remove_metadata(input_path, output, new_dir, rgb)

if __name__ == '__main__':
    main()
    
