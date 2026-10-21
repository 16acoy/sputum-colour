from PIL import Image
from PIL.ExifTags import TAGS
import re
import pandas as pd
import cv2
import pillow_heif
import numpy as np
import json


def imread_compatible(path):
    # TODO - Record original image version before decoding between formats
    # TODO - Update exif metadata extraction to take openCV images instead?
    img = cv2.imread(str(path))
    img_grey = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is not None:
        return img, img_grey

    # Fall back to HEIC/HEIF decoding
    try:
        heif_file = pillow_heif.read_heif(str(path))
        img_rgb = np.asarray(heif_file)
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cv2.cvtColor(
            img_rgb, cv2.COLOR_RGB2GRAY
        )

    except Exception as e:
        print(f"Could not decode {path}: {e}")
        return None, None


# Get patient ID from filename if present
def get_patient_num(filename):
    if filename.lower().startswith("pt"):
        pt = filename.split(" ")[0][2:]  # Extract the number after 'pt'
        return pt

    return None


# Get lighting location from filename if present
def get_lighting_location(filename):
    # find 'location*' or 'location *' in filename
    location_match = re.search(r"location\s*(\d+)", filename, re.IGNORECASE)
    if location_match:
        return int(location_match.group(1))

    # account for typo - 'lcoation'
    location_match = re.search(r"lcoation\s*(\d+)", filename, re.IGNORECASE)
    if location_match:
        return int(location_match.group(1))

    return 1  # If no location recorded, assume location 1 (as per Rachel Gyte)


# Extract metadata from image file
def extract_metadata(filename, dir_path, site, photographer, lighting_location):
    metadata_dict = {}
    # Define desired keys
    keys = [
        "filename",
        "site",
        "photographer",
        "DateTime",
        "patient_id",
        "lighting_location",
        "Model",
        "YCbCrPositioning",
        "XResolution",
        "YResolution",
    ]

    # Add parameter metadata
    metadata_dict["filename"] = filename
    metadata_dict["site"] = site
    metadata_dict["photographer"] = photographer
    metadata_dict["lighting_location"] = lighting_location
    metadata_dict["patient_id"] = get_patient_num(filename)

    img = Image.open(f"{dir_path}/{filename}")
    exifdata = img.getexif()

    # Looping through all the tags present in exifdata
    for tagid in exifdata:
        tagname = TAGS.get(tagid, tagid)
        value = exifdata.get(tagid)
        if tagname in keys:
            metadata_dict[tagname] = value

    # Rename DateTime to date, Model to phone_model
    if "DateTime" in metadata_dict:
        metadata_dict["date"] = metadata_dict.pop("DateTime")
        # Convert to datetime object
        metadata_dict["date"] = pd.to_datetime(
            metadata_dict["date"], format="%Y:%m:%d %H:%M:%S", errors="coerce"
        )
    if "Model" in metadata_dict:
        metadata_dict["phone_model"] = metadata_dict.pop("Model")

    # Fill any empty dates
    if "date" not in metadata_dict or pd.isnull(metadata_dict["date"]):
        metadata_dict["date"] = None

    return metadata_dict


# Load label annotation data from JSON files
def load_label_json_file(path_labels):
    data = json.load(open(path_labels))
    try:
        df = pd.DataFrame(data["_via_img_metadata"])
    except:
        df = pd.DataFrame(data)
    vertices_df = df.transpose().reset_index()[["filename", "regions"]]

    return vertices_df
