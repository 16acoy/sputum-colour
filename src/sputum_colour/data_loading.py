from PIL import Image
from PIL.ExifTags import TAGS
import re
import pandas as pd
import cv2
import pillow_heif
import numpy as np
import json
import os
import matplotlib.pyplot as plt


def imread_compatible(path, printit=False):
    img = cv2.imread(str(path))
    img_grey = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if printit == True:
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title("Loaded colour Image")

    if img is not None:
        print(f"{path} successfully loaded as a cv2 image.")
        return img, img_grey

    # Fall back to HEIC/HEIF decoding
    try:
        heif_file = pillow_heif.read_heif(str(path))
        img_rgb = np.asarray(heif_file)
        cv_image = (
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY),
        )
        print(f"{path} successfully loaded as a HEIF image and converted to cv2.")
        return cv_image

    except Exception as e:
        print(f"Could not decode {path}: {e}")
        return None, None


def resize_image_to_ref(image_in, h_target, w_target):
    is_color = image_in.ndim == 3
    if is_color:
        im_pil = Image.fromarray(cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB))
    else:
        im_pil = Image.fromarray(image_in)

    width, height = im_pil.size
    target_aspect_ratio = w_target / h_target
    source_aspect_ratio = width / height

    if source_aspect_ratio > target_aspect_ratio:
        crop_width = round(height * target_aspect_ratio)
        left = (width - crop_width) // 2
        crop_box = (left, 0, left + crop_width, height)
    elif source_aspect_ratio < target_aspect_ratio:
        crop_height = round(width / target_aspect_ratio)
        top = (height - crop_height) // 2
        crop_box = (0, top, width, top + crop_height)
    else:
        crop_box = (0, 0, width, height)

    resized_image = im_pil.crop(crop_box).resize((w_target, h_target))
    numpy_image = np.array(resized_image)
    if is_color:
        return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return numpy_image


# Get patient ID from filename if present, or from defined CSVs otherwise. 'unknown' as fallback
def get_patient_num(filename, site):
    if site == "mcr_clinic":
        if filename.lower().startswith("pt"):
            patient_id = filename.split(" ")[0][2:]  # Extract the number after 'pt'
            return patient_id
        else:
            return "unknown"
    elif site == "mcr_lab":
        # Use csv of known patient IDs
        paths_df = pd.read_csv("../files/paths.csv")
        patient_ids_path = paths_df[paths_df["path_type"] == "jm_patient_ids"].iloc[0][
            "path"
        ]
        df = pd.read_csv(patient_ids_path)
        patient_id = df[df["filename"] == filename]["patient_id"].values[0]
        if pd.isna(patient_id):
            # No known patient ID for this sample
            patient_id = "unknown"

        return patient_id

    elif site == "qub_lab":
        # Get first 4 chars
        patient_id = filename[0:4]
        return patient_id

    return "unknown"


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
def extract_metadata(
    file_num, filename, dir_path, site, photographer, lighting_location
):
    metadata_dict = {}
    # Define desired keys
    keys = [
        "file_num",
        "filename",
        "site",
        "photographer",
        "DateTime",
        "patient_id",
        "lighting_location",
        "Model",
        "YCbCrPositioning",
    ]

    # Add parameter metadata
    metadata_dict["file_num"] = file_num
    metadata_dict["filename"] = filename
    metadata_dict["site"] = site
    metadata_dict["photographer"] = photographer
    metadata_dict["lighting_location"] = lighting_location
    metadata_dict["patient_id"] = get_patient_num(filename, site)

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
    else:
        # No 'date taken' field known for image
        if site == "qub_lab":
            parts = filename.split("_")
            part_index = len(parts) - 2
            date_from_filename = os.path.splitext(parts[part_index])[0]
            print(date_from_filename)
            metadata_dict["date"] = pd.to_datetime(
                date_from_filename, format="%d%m%Y", errors="coerce"
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


def get_point_array(x_points, y_points):
    point_array = []
    for n in range(len(x_points)):
        x = x_points[n]
        y = y_points[n]
        point_array.append([x, y])
    point_array = np.array(point_array)

    return point_array


def get_region_mask(target_image, vertices_df_row, num_regions=1):
    # Retrieve polyline points and store in numpy array of (x,y) coords
    masks = []
    print(target_image.shape)
    h_target, w_target, channels = target_image.shape

    for n in range(num_regions):
        print("doing region" + str(n))

        shape_type = vertices_df_row["regions"][n]["shape_attributes"]
        if shape_type["name"] == "point":
            x_point = shape_type["cx"]
            y_point = shape_type["cy"]
            masks.append((x_point, y_point))
        else:
            x_points = vertices_df_row["regions"][n]["shape_attributes"]["all_points_x"]
            y_points = vertices_df_row["regions"][n]["shape_attributes"]["all_points_y"]

            # Display min and max values to help with identifying exact test tube edges in algorithm
            # print('Max x value: ' + str(max(x_points)))
            # print('Min x value: ' + str(min(x_points)))
            # print('Max y value: ' + str(max(y_points)))
            # print('Min y value: ' + str(min(y_points)))

            point_array = get_point_array(x_points, y_points)

            # Convert polylines to mask using cv2
            mask = np.zeros((h_target, w_target))
            cv2.fillPoly(mask, [point_array], color=1)
            masks.append(mask)

        # cv2.imshow("Triangle", mask)
        # cv2.waitKey(0)

        # Closing all open windows
        # cv2.destroyAllWindows()

    return masks


######################
# MAIN FUNCTIONS
######################


def data_collection():
    paths_df = pd.read_csv("../files/paths.csv")
    target_image_path = paths_df[paths_df["path_type"] == "target_image_raw"].iloc[0][
        "path"
    ]
    print(target_image_path)
    dst_image, dst_image_grey = imread_compatible(target_image_path)
    # Get resolution of image
    h_target, w_target, channels = dst_image.shape

    # Get all images used for ROI extraction
    rg_photos_path = paths_df[paths_df["path_type"] == "rg_photos"].iloc[0]["path"]
    jm_photos_path = paths_df[paths_df["path_type"] == "jm_photos"].iloc[0]["path"]
    qub_photos_path = paths_df[paths_df["path_type"] == "qub_photos"].iloc[0]["path"]

    # Access images, metadata + reindex image files for downstream storage/processing

    all_paths = [rg_photos_path, jm_photos_path, qub_photos_path]
    all_sites = ["mcr_clinic", "mcr_lab", "qub_lab"]
    all_photographers = ["Rachel Gyte", "Jemima Moses", "Gisli Einarsson"]

    all_files = []
    all_metadata = {}

    file_num = 0

    for i in range(len(all_paths)):
        dir_path = all_paths[i]
        site = all_sites[i]
        photographer = all_photographers[i]
        print(dir_path)

        filenames = os.listdir(dir_path)
        for filename in filenames:
            lighting_location = get_lighting_location(filename)
            metadata_dict = extract_metadata(
                file_num, filename, dir_path, site, photographer, lighting_location
            )
            all_metadata[f"image_{file_num}.jpg"] = metadata_dict

            src_image, src_image_grey = imread_compatible(f"{dir_path}/{filename}")

            h, w, channels = src_image.shape
            src_image_resized = resize_image_to_ref(src_image, h_target, w_target)
            src_image_grey_resized = resize_image_to_ref(
                src_image_grey, h_target, w_target
            )

            all_files.append((src_image_resized, src_image_grey_resized))

            file_num += 1

    return (dst_image, dst_image_grey), all_files, all_metadata


def get_rough_masks(target_image):

    paths_df = pd.read_csv("../files/paths.csv")

    # Load rough masks for location of tube and grey card in target image
    rough_target_mask_path = paths_df[
        paths_df["path_type"] == "target_tube_bounding_box_ann"
    ].iloc[0]["path"]
    rough_target_mask_df = load_label_json_file(rough_target_mask_path)

    rough_target_mask_row = rough_target_mask_df.iloc[0]
    masks = get_region_mask(target_image, rough_target_mask_row, num_regions=2)
    tube_target_mask = masks[0]
    card_target_mask = masks[1]

    return tube_target_mask, card_target_mask


def get_rough_masks_2(target_image):

    paths_df = pd.read_csv("../files/paths.csv")

    # Load rough masks for location of tube and grey card in target image
    rough_target_mask_path = paths_df[paths_df["path_type"] == "target_roi_ann"].iloc[
        0
    ]["path"]
    rough_target_mask_df = load_label_json_file(rough_target_mask_path)

    rough_target_mask_row = rough_target_mask_df.iloc[0]
    masks = get_region_mask(target_image, rough_target_mask_row, num_regions=3)
    roi_mask = masks[0]
    midpoint = masks[2]

    return roi_mask, midpoint


def get_roi_ground_truth_label(filename, site, target_image, paths_df):
    if site == "mcr_clinic":
        annotations_file = paths_df[paths_df["path_type"] == "rg_roi_labels"].iloc[0][
            "path"
        ]
    elif site == "mcr_lab":
        annotations_file = paths_df[paths_df["path_type"] == "jm_roi_labels"].iloc[0][
            "path"
        ]
    elif site == "qub_lab":
        annotations_file = paths_df[paths_df["path_type"] == "qub_roi_labels"].iloc[0][
            "path"
        ]
    else:
        annotations_file = None

    rois_df = load_label_json_file(annotations_file)
    file_roi_row = rois_df[rois_df["filename"] == filename]
    file_roi_row = file_roi_row.reset_index(drop=True, inplace=False).iloc[0]

    masks = get_region_mask(
        target_image,
        file_roi_row,
        num_regions=1,  # NOTE - target_image is only used to determine shape but isn't actually the original image the mask was made for
    )

    roi_label_mask = masks[0]

    return roi_label_mask
