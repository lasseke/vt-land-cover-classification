"""Helper functions for the Google Earth Engine Python API"""

import json
import re
from typing import List, Optional
import pandas as pd
import ee
import folium

ee.Initialize()


def get_norway_geometry() -> ee.Geometry.Polygon:
    '''
    Returns handdrawn shape of Norway (hard coded, coarse coordinates).

    Return type: ee.Geometry.Polygon
    '''

    # Simplified, handdrawn geometry for Norway bounds
    nor_geom = ee.Geometry.Polygon(
        coords=[
            [
                [12.516597591701263, 58.56335863250716],
                [13.571285091701263, 60.82427563929901],
                [13.834956966701263, 62.695044801132276],
                [15.065425716701263, 64.94283928358745],
                [17.877925716701263, 67.5595597918735],
                [21.217769466701263, 68.51273038442407],
                [25.612300716701263, 68.1558055834418],
                [27.545894466701263, 69.45811599514172],
                [28.600581966701263, 68.57702625686804],
                [31.500972591701263, 69.27227591533666],
                [32.81933196670126, 70.48143127597439],
                [28.688472591701263, 71.34352616981468],
                [24.381831966701263, 71.2872131257632],
                [18.932613216701263, 70.45204469620343],
                [15.241206966701263, 69.8553141649792],
                [13.219722591701263, 68.92738774387972],
                [9.616206966701263, 64.71856439012235],
                [4.870113216701264, 62.49276796105288],
                [4.254878841701264, 61.66978494130909],
                [3.903316341701264, 59.1091310990978],
                [6.540035091701264, 57.63449789078464]
            ]
        ],
        proj='EPSG:4326',
        maxError=0.001,
        geodesic=False
    )

    return nor_geom


def mask_clouds_sentinel2(image: ee.Image) -> ee.Image:
    '''
    Mask clouds from Sentinel-2 imagery using the quality band QA60 and scales
    surface reflectance between 0 and 1.

    Input:
    image: ee.Image from which cloudy pixels should be removed.
    '''

    qa60 = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively. Operate bitwise (<<).
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa60.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa60.bitwiseAnd(cirrus_bit_mask).eq(0)
    )

    # Return the masked and scaled data, without the QA bands.
    return image.updateMask(mask).divide(10000).select("B.*").copyProperties(
        image, ["system:time_start"]
    )


def mask_clouds_landsat7(image: ee.Image) -> ee.Image:
    '''
    Mask clouds from Landsat-7 imagery using the quality band 'pixel_qa'.

    Input:
    image: ee.Image from which cloudy pixels should be removed.
    '''

    qa = image.select('pixel_qa')

    # If the cloud bit (5) is set and the cloud confidence (7) is high
    # or the cloud shadow bit is set (3), then it's a bad pixel.
    cloud = qa.bitwiseAnd(1 << 5).And(
        qa.bitwiseAnd(1 << 7)).Or(qa.bitwiseAnd(1 << 3))

    # Remove edge pixels that don't occur in all bands
    mask2 = image.mask().reduce(ee.Reducer.min())

    return image.updateMask(cloud.Not()).updateMask(mask2)


def apply_landsat_scale_factors(image: ee.Image) -> ee.Image:
    '''
    Apply scaling to the SR values as specified in
    https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C01_T1_SR#bands.

    Input:
    image: Landsat 7 ee.Image to rescale.
    '''

    img_sr_bands = image.select(
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B7'],
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']
    ).multiply(0.0001)

    img_thermal_band = image.select(['B6'], ['B6']).multiply(0.1)

    return img_sr_bands.addBands(img_thermal_band)


def get_lower_and_upper_quantiles(
    img_collection: ee.ImageCollection, lower: int, upper: int
):
    '''
    Create outlier images based on upper and lower quartile values.

    lower: lower quartile threshold value.
    upper: upper quartile threshold value.
    '''

    # Calculate 95th and 5th percentiles to remove outliers
    return img_collection.reduce(ee.Reducer.percentile([lower])), \
        img_collection.reduce(ee.Reducer.percentile([upper]))


def get_image_spectral_index(
    image: ee.Image, ind_name: str, sensor_name: str
) -> ee.Image:
    '''
    Mapping function to calculate spectral indices for a single GEE image.
    '''

    # Read in dictionary of defined spectral indices
    with open(
        '../data/dict/spectral_indices.json', encoding='utf-8'
    ) as json_file:
        indices_dict = json.load(json_file)

    # Get corresponding formula from dict, throw exception if undefined
    try:
        cur_formula = indices_dict[ind_name].get("formula").get(sensor_name)
    except KeyError:
        print(
            "The requested entry is not defined in 'spectral_indices.json'."
        )
        raise

    # Retrieve all band strings (needed for GEE function arguments)
    # All unique (-> set) bands in formula, names starting with B*
    cur_bands_str_list = set(re.findall(r'\bB\w+', cur_formula))
    # Convert to string and store in list
    cur_bands_str_list = [str(x) for x in cur_bands_str_list]
    # Construct required dict
    cur_expression_dict = {band: image.select(
        band) for band in cur_bands_str_list}

    return image.addBands(
        image.expression(cur_formula, cur_expression_dict).rename(
            ind_name).float()
    )


# DEFINE FUNCTIONS COMPATIBLE WITH GEE MAPPING FOR CALCULATING INDICES
# SENTINEL2

def map_ndvi_sentinel2(image: ee.Image) -> ee.Image:
    """Map Sentinel 2 NDVI"""

    return get_image_spectral_index(
        image, ind_name="NDVI", sensor_name="Sentinel2"
    )


def map_gndvi_sentinel2(image: ee.Image) -> ee.Image:
    """Map Sentinel 2 GNDVI"""

    return get_image_spectral_index(
        image, ind_name="GNDVI", sensor_name="Sentinel2"
    )


def map_evi_sentinel2(image: ee.Image) -> ee.Image:
    """Map Sentinel 2 EVI"""

    return get_image_spectral_index(
        image, ind_name="EVI", sensor_name="Sentinel2"
    )


def map_savi_sentinel2(image: ee.Image) -> ee.Image:
    """Map Sentinel 2 SAVI"""

    return get_image_spectral_index(
        image, ind_name="SAVI", sensor_name="Sentinel2"
    )


def map_ndmi_sentinel2(image: ee.Image) -> ee.Image:
    """Map Sentinel 2 NDMI"""

    return get_image_spectral_index(
        image, ind_name="NDMI", sensor_name="Sentinel2"
    )


# DEFINE FUNCTIONS COMPATIBLE WITH GEE MAPPING FOR CALCULATING INDICES
# Landsat 7

def map_ndvi_landsat7(image: ee.Image) -> ee.Image:
    """Map Landsat 7 NDVI"""

    return get_image_spectral_index(
        image, ind_name="NDVI", sensor_name="Landsat7"
    )


def map_gndvi_landsat7(image: ee.Image) -> ee.Image:
    """Map Landsat 7 GNDVI"""

    return get_image_spectral_index(
        image, ind_name="GNDVI", sensor_name="Landsat7"
    )


def map_evi_landsat7(image: ee.Image) -> ee.Image:
    """Map Landsat 7 EVI"""

    return get_image_spectral_index(
        image, ind_name="EVI", sensor_name="Landsat7"
    )


def map_savi_landsat7(image: ee.Image) -> ee.Image:
    """Map Landsat 7 SAVI"""

    return get_image_spectral_index(
        image, ind_name="SAVI", sensor_name="Landsat7"
    )


def map_ndmi_landsat7(image: ee.Image) -> ee.Image:
    """Map Landsat 7 NDMI"""

    return get_image_spectral_index(
        image, ind_name="NDMI", sensor_name="Landsat7"
    )


# CLIPPING

def _clip_to_geometry(image: ee.Image, geometry: ee.Geometry):
    '''
    Mask to the extent of a given geometry.
    '''

    mask = ee.Image.constant(1).clip(geometry).mask()

    return image.updateMask(mask)


def clip_to_nor_geometry(image):
    '''
    Mask to the extent of the handdrawn Norway shape.
    '''

    return _clip_to_geometry(image, get_norway_geometry())


# Interactive maps using folium

def add_ee_layer(self, ee_object, vis_params, name):
    '''
    Display GEE image on folium WMS map.
    '''

    try:
        if isinstance(ee_object, ee.image.Image):

            map_id_dict = ee.Image(ee_object).getMapId(vis_params)

            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True
            ).add_to(self)

    except Exception as excep:
        print(f"Could not display {name}, reason: {excep}")


# Extract points from an ee.Image

class VTFeatureCollection(object):
    """Class for extracting points from an ee.FeatureCollection"""

    def __init__(
        self,
        user_name: str,
        extract_columns: List[str],
        vt_asset_name: str = 'vtprespoints'
    ):

        self.extract_columns = extract_columns
        self.path_to_fc_asset = f'users/{user_name}/{vt_asset_name}'

        # Build feature collection
        self.update_feat_collection()

    def update_feat_collection(self):
        """Update feature collection json with current class parameters"""

        # initialize
        vt_fc_raw = ee.FeatureCollection(self.path_to_fc_asset)

        # Extract relevant columns, see Strand (2013) and Bryn et al. (2018)
        self.feat_collection = vt_fc_raw.select(self.extract_columns)
        self.n_points = self.feat_collection.size().getInfo()

        print(
            f"FeatureCollection initialized, number of rows: {self.n_points}"
        )

    # Extract raster

    def extract_from_eeimg(
        self, img: ee.Image, scale: int, tile_scale: int
    ):
        """Send request to extract values from ee.Image at point locations"""

        # Extract values from ee.Image
        sampled_fc = img.sampleRegions(
            collection=self.feat_collection,
            properties=list(
                # Use given variable column names
                self.feat_collection.first().getInfo().get('properties').keys()
            ),
            scale=scale,  # Resampled spatial scale, in meters
            tileScale=tile_scale  # Default: 1, Scaling tiles to save memory
        )

        print("Starting extraction. This process can take 5-10 minutes.")

        # Warning, takes time! Example timing output:
        # 5min 22s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
        sampled_list = sampled_fc.toList(self.n_points).getInfo()

        return sampled_list

    def set_path(self, new_path: str) -> None:
        """Set relative path to GEE asset"""
        self.path_to_fc_asset = new_path
        self.update_feat_collection()

    def get_path(self) -> str:
        """Get relative path to GEE asset"""
        return self.path_to_fc_asset

    def set_extract_columns(self, new_columns: List[str]) -> None:
        """Change column names to extract"""
        self.extract_columns = new_columns
        self.update_feat_collection()

    def get_extract_columns(self) -> List[str]:
        """Get column names to extract"""
        return self.extract_columns

    def get_feat_collection(self) -> ee.FeatureCollection:
        """Get FeatureCollection"""
        return self.feat_collection


def df_from_sampled_list(
    sampled_list, drop_cols: Optional[List[str]] = None,
    save_as: str = None, save_path: str = "../data/raw/"
) -> pd.DataFrame:
    """
    Create a pandas.DataFrame from a sampled list created by
    VTFeatureCollection.extract_from_eeimg()
    """

    # Convert list to pandas DataFrame
    feature_mat = pd.json_normalize(
        sampled_list,
        meta=['properties'],
        meta_prefix=False,
        sep="."
    ).drop(
        labels=drop_cols if drop_cols else [],  # Drop unneeded columns?
        axis=1
    ).convert_dtypes()  # Infer data types

    # Remove prefix from column names
    feature_mat.columns = [
        col.replace("properties.", "") for col in feature_mat.columns
    ]

    # Check for duplicated entries
    duplicated = feature_mat.duplicated(subset=['POINT_X', 'POINT_Y'])
    print(f"Duplicate entries in raw df: {sum(duplicated)}.")

    if sum(duplicated) != 0:

        print("Removing duplicate entries...")

        # Remove duplicate rows, caused by artifact in original data
        feature_mat.drop(
            feature_mat[feature_mat.layer == "2e_p"].index, inplace=True
        )
        feature_mat.drop(
            feature_mat[feature_mat.layer == "2f_p"].index, inplace=True
        )

        duplicated = feature_mat.duplicated(
            subset=['POINT_X', 'POINT_Y']
        )

        if sum(duplicated) == 0:
            print("Finished!")
        else:
            raise RuntimeError(
                f"Something went wrong. Found {sum(duplicated)} duplicate "
                + "entries, should be 0. Check the datasets."
            )

    if save_as is not None:
        feature_mat.to_csv(save_path+save_as, index=False)

    return feature_mat
