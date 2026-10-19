
from PIL import Image
from PIL.ExifTags import TAGS
import re
import pandas as pd



# Get patient ID from filename if present
def get_patient_num(filename):
    if filename.lower().startswith('pt'):
        pt = filename.split(' ')[0][2:]  # Extract the number after 'pt'
        return pt

    return None

# Get lighting location from filename if present
def get_lighting_location(filename):
    # find 'location*' or 'location *' in filename
    location_match = re.search(r'location\s*(\d+)', filename, re.IGNORECASE)
    if location_match:
        return int(location_match.group(1))

    # account for typo - 'lcoation'
    location_match = re.search(r'lcoation\s*(\d+)', filename, re.IGNORECASE)
    if location_match:
        return int(location_match.group(1))

    return 1 # If no location recorded, assume location 1 (as per Rachel Gyte)

# Extract metadata from image file
def extract_metadata(filename, dir_path, site, photographer, lighting_location):
    metadata_dict = {}
    # Define desired keys
    keys = ['filename', 'site', 'photographer', 'DateTime', 'patient_id', 'lighting_location', 'Model', 'YCbCrPositioning', 'XResolution', 'YResolution']

    # Add parameter metadata
    metadata_dict['filename'] = filename
    metadata_dict['site'] = site
    metadata_dict['photographer'] = photographer
    metadata_dict['lighting_location'] = lighting_location
    metadata_dict['patient_id'] = get_patient_num(filename)

    img = Image.open(f"{dir_path}/{filename}")
    exifdata = img.getexif()

    # Looping through all the tags present in exifdata
    for tagid in exifdata:

        tagname = TAGS.get(tagid, tagid)
        value = exifdata.get(tagid)
        if tagname in keys:
            metadata_dict[tagname] = value

    # Rename DateTime to date, Model to phone_model
    if 'DateTime' in metadata_dict:
        metadata_dict['date'] = metadata_dict.pop('DateTime')
        # Convert to datetime object
        metadata_dict['date'] = pd.to_datetime(metadata_dict['date'], format='%Y:%m:%d %H:%M:%S', errors='coerce')
    if 'Model' in metadata_dict:
        metadata_dict['phone_model'] = metadata_dict.pop('Model')

    # Fill any empty dates
    if 'date' not in metadata_dict or pd.isnull(metadata_dict['date']):
        metadata_dict['date'] = None

    return metadata_dict

