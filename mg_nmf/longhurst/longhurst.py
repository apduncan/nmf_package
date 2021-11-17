"""
Adapted from code by Sara Collins. MIT. 3/18/2015, available at
https://github.com/thechisholmlab/Longhurst-Province-Finder.

Find the Longhurst province for latitude/longitude coordinates.
"""

from typing import Dict, List, Any, Tuple, Generator
import xml.etree.ElementTree as ETree
import pandas as pd

XML_NS: Dict[str, str] = {
    'region': 'geo.vliz.be/MarineRegions',
    'gml': 'http://www.opengis.net/gml'
}

def parse_longhurst_xml(file: str = 'longhurst.xml') -> Dict[str, Dict[str, Any]]:
    """Parse and XML file containing the definitions of Longhurst provinces.
    :param file: XML file containing Longhurst province definitions
    :type file: str

    :return: Dictionary with key fid, value is a dictionary of with all properties of that province
    :rtype: Dict[str, Dict[str, Any]]
    """

    tree: ETree.ElementTree = ETree.parse(file)
    root: ETree.Element = tree.getroot()
    provinces: Dict[str, Dict[str, Any]] = {}
    region: ETree.Element
    for region in root.findall('.//region:longhurst', XML_NS):
        # Get province code, name, and bounding box from file
        prov_code: str = region.find('.//region:provcode', XML_NS).text
        prov_name: str = region.find('.//region:provdescr', XML_NS).text
        fid: str = region.attrib['fid']
        box: str = region.find('.//gml:coordinates', XML_NS).text

        # Parse bounding box coordinates
        box_split: List[str] = box.split(' ')
        x1, y1 = tuple(float(x) for x in box_split[0].split(','))
        x2, y2 = tuple(float(y) for y in box_split[1].split(','))

        # Find the element giving the full definition of the region's geometry
        geometry: ETree.Element = region.find('.//region:the_geom', XML_NS)

        provinces[fid] = dict(prov_code=prov_code, prov_name=prov_name, fid=fid, x1=x1, x2=x2, y1=y1, y2=y2,
                              geom=geometry)
    return provinces

PROVINCES: Dict[str, Dict[str, Any]] = parse_longhurst_xml()

def _candidates(lat: float, lon: float) -> List[Dict[str, Any]]:
    """Identify possible provinces for coordinates, based on the bounding boxes which enclose the coordinates.
    :param lat: Latitude
    :type lat: float
    :param lon: Longitude
    :type lon: float

    :return: Candidate provinces for these coordinates
    :rtype: List[Dict[str, Any]]
    """

    return [p for p in PROVINCES.values() if (p['y1'] <= lat <= p['y2']) and (p['x1'] <= lon <= p['x2'])]

def _crossing_test(lat: float, lon: float, region: Dict[str, Any]) -> bool:
    """Determine if coordinates are within a given region, using crossing test.

    :param lat: Latitude
    :type lat: float
    :param lon: Longitude
    :type lon: float
    :param region: Properties of the region, as extracted from XML
    :type region: Dict[str, Any]

    :rtype: bool
    :return: Coordinates in region, true if they are
    """
    geom: ETree.Element = region['geom']
    crossings: int = 0

    for g in geom:
        c: ETree.Element = g.findall('.//gml:coordinates', XML_NS)

        for i in c:
            # Get pairs of coordinates
            coord_pairs: List[Tuple[float, float]] = [tuple(float(z) for z in x.split(',')) for x in i.text.split(' ')]
            # Use pairs of coordinates at p and p+1 to perform Crossings Test
            for p in range(len(coord_pairs)-1):
                pa, pb = coord_pairs[p], coord_pairs[p+1]
                pass_lat: bool = (pa[1] >= lat >= pb[1]) or (pa[1] <= lat <= pb[1])
                pass_lon: bool = lon <= pb[0]
                if pass_lat and pass_lon:
                    crossings += 1
    if crossings % 2 == 1:
        return True
    else:
        return False


def get_province(lat: float, lon: float) -> List[Tuple[str, Dict[str, Any]]]:
    """Find province which contains the given coordinates. Where could be multiple, returns all candidates in
    a list.

    :param lat: Latitude
    :type lat: float
    :param lon: Longitude
    :type lon: float

    :return: List of longhurst provinces containing this coordinate. Multiple returned where ambiguous. Returned as a
                list of tuples, in form (province id, dictionary of province properties)
    :type: List[Tuple[str, Dict[str, Any]]
    """

    # Get candidate provinces based on bounding boxes
    candidates: List[Dict[str, Any]] = _candidates(lat, lon)

    # Perform crossing tests for each candidate province
    provinces: List[Dict[str, Any]] = [x for x in candidates if _crossing_test(lat, lon, x)]

    return provinces


def append_province_df(df: pd.DataFrame, lat_col: str, lon_col: str, province_col: str = 'longhurst_province') \
        -> pd.DataFrame:
    """Convenience function to add the province code onto a dataframe containing lat and lon"""
    def row_to_prov(series: pd.Series) -> str:
        lat, lon = float(series[lat_col]), float(series[lon_col])
        provinces: List[Dict[str, Any]] = get_province(lat, lon)
        prov_str: str = '' if len(provinces) == 0 else ','.join(x['prov_code'] for x in provinces)
        return prov_str
    df[province_col] = df.apply(row_to_prov, axis=1)
    return df

def province_search(find: str, key: str = 'prov_code') -> Dict[str, Any]:
    """Get all details of a povince, from one property (probably fid or code)

    :param find:
    :type find:
    :param key:
    :type key:

    :return: Properties of the region
    """

    return next(x for x in PROVINCES.values() if x[key] == find)
