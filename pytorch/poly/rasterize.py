import sys
import subprocess
import os
import gdal
import glob
from setup import setup_run

def call_gdal_rasterize(input_file, main_path, tif_name, overwrite_tif):
    # Set up the path to the GeoTIFF
    tif_file = os.path.join(main_path, tif_name)

    # Open the image with GDAL
    src = gdal.Open(tif_file)

    # Find the corner coordinates of the image
    ul_x, x_res, x_skew, ul_y, y_skew, y_res  = src.GetGeoTransform()

    lr_x = ul_x + (src.RasterXSize * x_res)
    lr_y = ul_y + (src.RasterYSize * y_res)

    # Read the raster band as a separate variable
    band = src.GetRasterBand(1)

    # Data type of the values
    band_type = gdal.GetDataTypeName(band.DataType)

    # Grab the no data value
    no_data = band.GetNoDataValue()

    # Set up the output file path and image name
    if overwrite_tif:
        output_file = tif_file

    else:
        output_path = os.path.join(main_path, 'smooth_predictions')  
        output_file = os.path.join(output_path, tif_name)
        # If the directory to save the new GeoTIFFs to does not exist then create it   
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
            print ("Created folder 'smooth_predictions'")

    # Call gdal_rasterize
    subprocess.call(['gdal_rasterize', '-a_nodata', str(no_data), '-ot', str(band_type), '-te',
                    str(ul_x), str(lr_y), str(lr_x), str(ul_y), '-tr', str(x_res), str(y_res),
                    '-burn', '255', '-q', input_file, output_file])

def run_rasterize(main_path, smooth_path, overwrite_tif):
    # Set up required file paths to the images
    smooth_fp = glob.glob(os.path.join(smooth_path, '*.geojson'))
    ext_length = len(os.path.splitext(smooth_fp[0]))

    print ('Rasterizing the GeoJSONs...')
    for geojson in smooth_fp:
        # Switch the file extension from .geojson to .tif
        file_name = os.path.splitext(os.path.basename(geojson))[0]
        tif_name = file_name + '.tif'

        # Rasterize the GeoJSONs
        call_gdal_rasterize(geojson, main_path, tif_name, overwrite_tif)

    print ('Done.')

def main():
    main_path, smooth_path, overwrite_tif = setup_run('rasterize')
    run_rasterize(main_path, smooth_path, overwrite_tif)

if __name__ == '__main__':
    main()
