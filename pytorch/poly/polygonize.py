import sys
import subprocess
import os
import ogr
import glob
from setup import setup_run

def call_gdal_polygonize(input_file, output_file):
    # If the file already exists, delete it first or additional polygons will be saved to the 
    # files if this is ran more than once.
    if os.path.isfile(output_file):
        os.remove(output_file)

    # Call gdal_polygonize.py
    subprocess.call(['gdal_polygonize.py', input_file, '-b', '1', '-q', '-f','GeoJSON',
                    output_file])

    # Open the image with OGR
    src = ogr.Open(output_file)

    # If the GeoTIFF has no shapes to polygonize then gdal_polygonize outputs a GeoJSON in an
    # incorrect format, so delete it.
    layer = src.GetLayer(0)

    # The GeoJSON that needs to be deleted will have no features
    count = layer.GetFeatureCount()

    if count == 0:
        os.remove(output_file)
        print ('Removed ' + os.path.basename(output_file))

def run_polygonize(main_path):
    # Set up required file paths to the images
    poly_fp = glob.glob(os.path.join(main_path, '*.tif'))

    # Set up the main file path to where the original GeoJSONs will be saved
    geojson_path = os.path.join(main_path, 'original_geojson')

    # If the directory to save the GeoJSONs to does not exist then create it
    if not os.path.isdir(geojson_path):
        os.mkdir(geojson_path)
        print ("Created folder 'original_geojson'")    

    print ('Polygonizing the predictions...')
    for tif_fp in poly_fp:
        # Switch the file extension from .tif to .geojson
        file_name = os.path.splitext(os.path.basename(tif_fp))[0]
        geojson_name = os.path.join(geojson_path, file_name + '.geojson')

        # Set up the file path and name of the new GeoJSON file
        json_fp = os.path.join(main_path, geojson_name)

        # Polygonize the predictions
        call_gdal_polygonize(tif_fp, json_fp)

    print ('Done.')

    return geojson_path

def main():
    main_path = setup_run('polygonize')

    run_polygonize(main_path)

if __name__ == '__main__':
    main()
