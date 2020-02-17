import sys
import os
import glob
import subprocess
from setup import setup_run

def smooth_lines(geojson_path, file_name, smooth_path):
    # Set up the file path and image name for the output
    bs_name = os.path.basename(file_name)
    smooth_file = os.path.join(smooth_path, bs_name)

    # Smooth the lines (automatically saves smooth_file as a new file)
    subprocess.call(['ogr2ogr', '-f', 'GeoJSON', '-overwrite', smooth_file, file_name,
                     '-simplify', '2'])

def run_smooth(original_path):
    # Set up required file paths to the images
    poly_fp = glob.glob(os.path.join(original_path, '*.geojson'))

    # Set up the main file path to where the smooth GeoJSONs will be saved
    smooth_path = os.path.join(original_path, 'smooth_geojson')

    # If the directory to save the smoothed GeoJSONs to does not exist then create it
    if not os.path.isdir(smooth_path):
        os.mkdir(smooth_path)
        print ("Created folder 'smooth_geojson'")

    print ('Smoothing the GeoJSONs...')
    for geojson_fp in poly_fp:
        # Smooth the polygon's lines
        smooth_lines(original_path, geojson_fp, smooth_path)

    print ('Done.')

    return smooth_path

def main():
    original_path = setup_run('smooth')    

    run_smooth(original_path)

if __name__ == '__main__':
    main()
