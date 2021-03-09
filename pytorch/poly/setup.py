import sys
import os

def setup_run(status):
    """
    This function sets up where the predictions are loaded in from and where polygonization,
    smoothing, and rasterizing are saved to. Set up to work only with GeoTIFFs.
    """
    # File path to the predictions.
    main_path = "/file/path/to/predictions/"

    # Indicate if you want to replace the original GeoTIFFs or save the new GeoTIFFs to a
    # different directory.
    overwrite_tif = False

    if status == 'polygonize':
        # Make sure the file path exist
        if not os.path.isdir(main_path):
            sys.exit("Error: Main file path to directory 'predictions' does not exist")

        return main_path

    elif status == 'smooth':
        original_path = os.path.join(main_path, 'original_geojson')
        # Make sure the file path exist
        if not os.path.isdir(original_path):
            sys.exit("Error: Main file path to directory 'original_geojson' does not exist")

        return original_path

    else:
        smooth_path = os.path.join(main_path, 'original_geojson', 'smooth_geojson')
        # Make sure the file path exist
        if not os.path.isdir(smooth_path):
            sys.exit("Error: Main file path to directory 'smooth_geojson' does not exist")

        return main_path, smooth_path, overwrite_tif
