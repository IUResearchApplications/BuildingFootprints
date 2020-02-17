# Polygonization #
## Required ##
The predictions can be polygonized. There are a couple of additional programs needed:
* [GDAL](https://gdal.org/) (version 2.2.4)
* [ExifTool](https://sno.phy.queensu.ca/~phil/exiftool/) (version 11.70 or higher)

## On Indiana University's Carbonate Computer Cluster ##
The python scripts in this directory were run on Indiana University's Carbonate computer cluster in Research Desktop (RED). RED has GDAL already installed but needs ExifTool. If you are running this in RED, then there are a couple of steps you need to do in order to run the Python scripts.

To run `metadata.py` you need an environment that has Python and PyExifTool installed in it. In order to use PyExifTool you will need to download [ExifTool](https://sno.phy.queensu.ca/~phil/exiftool/) by following the instructions on the site. Once it is downloaded you will need to export the path to ExifTool to your PATH in your terminal with `export PATH=$PATH:/Path/to/ExifTool/Image-ExifTool-11.70/`. Your version might be different. You will need to do this for every new terminal session if you want to run `metadata.py`.

To run `polygonize.py`, `smooth.py`, and `rasterize.py` type in the command line:
1. `module unload python`
2. `module load anaconda/python2.7`
3. `source activate /N/soft/rhel7/qgis/qgis_conda_env`

Now your environment should be set up to run `polygonize.py`, `smooth.py`, and `rasterize.py`.

## On Personal/Work Desktop ##
If you are not running this on Research Desktop (RED) then there are different steps you will need to take in order for the scripts to run. 

1. Install GDAL
2. Download [ExifTool](https://sno.phy.queensu.ca/~phil/exiftool/) by following the instructions on the site.
3. Export the path to ExifTool to your PATH in your terminal with `export PATH=$PATH:/Path/to/ExifTool/Image-ExifTool-11.70/`. Your version might be different. You will need to do this for every new terminal you open if you want to run `metadata.py`.

Once you have done all of this you should be able to run `polygonize.py`, `smooth.py`, and `rasterize.py`.

To run `metadata.py` you need an environment that has Python and PyExifTool installed on it.

## Setup and Run ##
The predictions should be GeoTIFFs.

The metadata needs to be reattached to the predictions. Specify where the images with the metadata are and where you saved your predictions in `metadata.py` then run it. If you do not reattach the metadata, then `rasterize.py` will not work.

In `setup.py` you will need to specify where your predictions are saved. If you want to rasterize your GeoJSONs afterwards then you will also need to specify if you want to overwrite the original GeoTIFF files. Once this is set up run `polygonization.py`. This will create the directory `original_geojson` inside your `predictions` directory. This is where the new polygonized GeoJSONs will be saved. Now run `smooth.py` to smooth the lines in the GeoJSON files. A new directory called `smooth_geojon` will be created inside your `original_geojson` directory. If you want to change your GeoJSONs back into GeoTIFFs then run `rasterize.py`. The burn value for rasterizing is set to 255. If you chose not to overwrite the original GeoTIFFs then a new directory called `smooth_predictions` will be created in your `predictions` directory. This is where the new GeoTIFFs will be saved.

If you want to strip the metadata from the images, then you can specify where the images are and where they will be saved in `remove.py` and run it.

**Note:** The amount of GeoJSON files might be smaller than the amount of original GeoTIFF files. If a prediction has no shapes in it then `polygonize.py` will output a GeoJSON file in an incorrect format, so it is deleted. Thus, the missing files are ones that had nothing in them to polygonize.

## File Structure ##
The structure of the file system looks like this after you have ran all of the scripts:
* predictions
    * original_geojson
        * smooth_geojson
    * smooth_predictions (If original GeoTIFFs are not overwritten)

