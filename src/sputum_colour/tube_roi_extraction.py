import matplotlib.pyplot as plt
from sputum_colour import data_loading
import cv2
from skimage.util import img_as_ubyte
from sklearn.metrics import jaccard_score
import numpy as np
from skimage import draw
import itertools
import pickle
from skimage.transform import warp


def get_oriented_lines(lines_in_tube_mask, orientation_tolerance_angles):
    vertical_lines = []
    horizontal_lines = []
    for line_num in range(len(lines_in_tube_mask)):
        line = lines_in_tube_mask[line_num]
        # Get start and end pixel of line
        x1, y1, x2, y2 = line.tolist()
        # Get all pixels on the line
        row_pixel_list, col_pixel_list = draw.line(y1, x1, y2, x2)
        """
        img = np.zeros((4032, 3024), dtype=np.uint8)
        img[row_pixel_list, col_pixel_list] = 1
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('One line')
        plt.tight_layout()
        plt.show()
        """

        # Get angle of line
        angle_rads = np.arctan2(y2 - y1, x2 - x1)
        angle = np.degrees(angle_rads)
        if angle < 0:
            angle += 180
        print("ANGLE: ", angle)

        # Filter what is considered v/h based on orientation_tolerance_angles parameter

        if (
            angle >= orientation_tolerance_angles[0][0]
            and angle <= orientation_tolerance_angles[0][1]
        ):
            vertical_lines.append((row_pixel_list, col_pixel_list))
        elif (angle >= 0 and angle <= orientation_tolerance_angles[1][1]) or (
            angle >= orientation_tolerance_angles[1][0] and angle <= 180
        ):
            horizontal_lines.append((row_pixel_list, col_pixel_list))

        # break

    return vertical_lines, horizontal_lines


# Calculate average distance between all lines
def get_average_distance(
    line_type, lines
):  # lines = list of [y_coords, x_coords] (or their keys if need dict of atributes)

    lines_dict = dict(enumerate(lines, start=0))

    line_key_pairs = list(itertools.combinations(lines_dict, 2))
    line_pairs = [(lines_dict[a], lines_dict[b]) for (a, b) in line_key_pairs]

    if line_type == "vertical":
        dist_coord = 1  # x
        fixed_coord = 0  # y
    elif line_type == "horizontal":
        dist_coord = 0  # y
        fixed_coord = 1  # x

    # Need to calculate horizontal distance

    pair_distances = {}
    pair_midpoints = {}
    for pair_n in range(len(line_pairs)):
        (l1, l2) = line_pairs[pair_n]
        (key1, key2) = line_key_pairs[pair_n]
        total_dist = 0
        total_midpoint = 0
        count = 0
        start_align_val_l1 = min(l1[fixed_coord])
        start_align_val_l2 = min(l2[fixed_coord])
        end_align_val_l1 = max(l1[fixed_coord])
        end_align_val_l2 = max(l2[fixed_coord])
        l1_len = abs(end_align_val_l1 - start_align_val_l1)
        l2_len = abs(end_align_val_l2 - start_align_val_l2)
        loop_range = max(l1_len, l2_len)
        for i in range(loop_range):
            l1_index = i + start_align_val_l1
            l2_index = i + start_align_val_l2

            matrix_indices_l1 = [
                idx
                for idx in range(len(l1[fixed_coord]))
                if l1[fixed_coord][idx] == l1_index
            ]
            coord_values_l1 = [
                l1[dist_coord][matrix_idx_l1] for matrix_idx_l1 in matrix_indices_l1
            ]
            if len(coord_values_l1) > 1:
                coord_value_l1 == sum(coord_values_l1) / len(coord_values_l1)
            elif len(coord_values_l1) == 1:
                coord_value_l1 = coord_values_l1[0]
            elif len(coord_values_l1) == 0:
                continue  # Ignore this row/column as only 0/1 line has a pixel here
            else:
                print("whoops1")
                pass

            matrix_indices_l2 = [
                idx
                for idx in range(len(l2[fixed_coord]))
                if l2[fixed_coord][idx] == l2_index
            ]
            coord_values_l2 = [
                l2[dist_coord][matrix_idx_l2] for matrix_idx_l2 in matrix_indices_l2
            ]
            if len(coord_values_l2) > 1:
                coord_value_l2 == sum(coord_values_l2) / len(coord_values_l2)
            elif len(coord_values_l2) == 1:
                coord_value_l2 = coord_values_l2[0]
            elif len(coord_values_l2) == 0:
                continue  # Ignore this row/column as only 0/1 line has a pixel here
            else:
                print("whoops2")
                pass

            lines_distance = abs(coord_value_l1 - coord_value_l2)
            total_dist += lines_distance

            lines_midpoint = min(coord_value_l1, coord_value_l2) + lines_distance / 2
            total_midpoint += lines_midpoint

            count += 1

        average_distance = total_dist / count
        average_midpoint_coord = total_midpoint / count

        pair_distances[(key1, key2)] = average_distance
        pair_midpoints[(key1, key2)] = average_midpoint_coord

    return pair_distances, pair_midpoints


def get_valid_line_pairs(pair_distances, t_low, t_high):

    # If average distance between any pair is = known pixels, add to valid lines

    pairs_valid = {
        k: v for k, v in pair_distances.items() if v >= t_low and v <= t_high
    }
    print("Num valid pairs: ", len(pairs_valid))
    key_pairs_valid = [k for k, v in pairs_valid.items()]  # tuples of valid keys

    return key_pairs_valid


def plot_valid_lines(image, lines, key_pairs_valid):

    key_lines_valid = set(itertools.chain(*key_pairs_valid))  # list of valid keys
    print(key_lines_valid)
    lines_valid = [lines[i] for i in key_lines_valid]  # list of valid lines

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    output_img = image.copy()

    for lineeee in lines_valid:
        rr, cc = lineeee

        # OpenCV uses (x, y) = (column, row)
        points = list(zip(cc, rr))

        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(
                output_img,
                [pts],
                isClosed=False,
                color=(0, 0, 255),  # Red in BGR
                thickness=2,
            )

    ax[0].imshow(output_img)
    ax[0].set_title("Valid lines (v or h)")

    img = np.zeros((4032, 3024), dtype=np.uint8)
    for lineeee in lines_valid:
        (rr, cc) = lineeee
        img[rr, cc] = 1

    ax[1].imshow(img, cmap="gray")
    ax[1].set_title("Valid lines (v or h)")

    plt.tight_layout()
    plt.show()

    return fig


def choose_final_valid_midpoint(key_pairs_valid, lines, pair_midpoints):

    if len(key_pairs_valid) == 0:
        midpoint = None
    else:
        (l1_key, l2_key) = key_pairs_valid[0]
        l1_valid = lines[l1_key]
        l2_valid = lines[l2_key]
        midpoint = pair_midpoints[(l1_key, l2_key)]

    return midpoint


####################
# MAIN FUNCTIONS
#################


def get_valid_tube_midpoint(
    image,
    lines_in_tube_mask,
    orientation_tolerance_angles,
    test_tube_vert_t_low,
    test_tube_vert_t_high,
    test_tube_hor_t_low,
    test_tube_hor_t_high,
    valid_line_filtering_method,
):
    vertical_lines, horizontal_lines = get_oriented_lines(
        lines_in_tube_mask, orientation_tolerance_angles
    )
    x_pair_distances, x_pair_midpoints = get_average_distance(
        "vertical", vertical_lines
    )
    y_pair_distances, y_pair_midpoints = get_average_distance(
        "horizontal", horizontal_lines
    )

    x_key_pairs_valid = get_valid_line_pairs(
        x_pair_distances, test_tube_vert_t_low, test_tube_vert_t_high
    )
    y_key_pairs_valid = get_valid_line_pairs(
        y_pair_distances, test_tube_hor_t_low, test_tube_hor_t_high
    )

    fig1 = plot_valid_lines(image, vertical_lines, x_key_pairs_valid)
    fig2 = plot_valid_lines(image, horizontal_lines, y_key_pairs_valid)

    if len(x_key_pairs_valid) > 1:
        # Add to QC
        print("Unclear vertical test tube edges")

    if len(y_key_pairs_valid) > 1:
        # Add to QC
        print("Unclear horizontal test tube edges")

    # Check there are only 4 valid lines - if not, take the first pair
    if valid_line_filtering_method == "first":
        # Calculate midpoints for estimated ROI
        midpoint_x = choose_final_valid_midpoint(
            x_key_pairs_valid, vertical_lines, x_pair_midpoints
        )
        midpoint_y = choose_final_valid_midpoint(
            y_key_pairs_valid, horizontal_lines, y_pair_midpoints
        )
        src_midpoint_pixel = (midpoint_x, midpoint_y)

    return (
        src_midpoint_pixel,
        len(x_key_pairs_valid),
        len(y_key_pairs_valid),
        fig1,
        fig2,
    )


def tube_to_tube_translate(src_midpoint_pixel, target_image):
    # Find diff between src tube midpoint and target tube midpoint - translate target mask to align values

    h, w, channels = target_image.shape
    target_roi_mask, target_tube_midpoint = data_loading.get_rough_masks_2(target_image)

    if src_midpoint_pixel[0] == None:
        # Downstream effect of no valid vertical pairs
        print("No x translation could be determined")
        tx = 0
    else:
        tx = src_midpoint_pixel[0] - target_tube_midpoint[0]

    if src_midpoint_pixel[1] == None:
        # Downstream effect of no valid horizontal pairs
        print("No y translation could be determined")
        ty = 0
    else:
        ty = src_midpoint_pixel[1] - target_tube_midpoint[1]

    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

    translated_mask = cv2.warpAffine(target_roi_mask, translation_matrix, (w, h))
    translated_mask = img_as_ubyte(translated_mask)

    return translated_mask


def transform_predicted_roi(model_filename, translated_mask):
    loaded_model = None
    try:
        with open(model_filename, "rb") as file:
            loaded_model = pickle.load(file)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file '{model_filename}' not found.")
    except Exception as e:
        print(f"Error loading model: {e}")

    transformed_mask = warp(
        translated_mask,
        loaded_model,
        output_shape=translated_mask.shape,
        preserve_range=False,
    )
    transformed_mask = img_as_ubyte(transformed_mask)

    return transformed_mask


def final_roi_extraction(image, translated_mask):
    # target_roi_mask = target_roi_mask.astype("uint8")

    # Extract same coords where target mask == 1 for final ROI
    final_roi = cv2.bitwise_and(image, image, mask=translated_mask)

    cutout = cv2.bitwise_not(final_roi)

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Image")

    ax[1].imshow(final_roi)
    ax[1].set_title("ROI")
    ax[2].imshow(cutout)

    # TODO - Checks in here

    return final_roi


def get_final_eval_metrics(final_roi_predicted, final_roi_ground_truth):

    y_true = final_roi_ground_truth.flatten() > 0
    y_pred = final_roi_predicted.flatten() > 0

    jaccard = jaccard_score(y_true, y_pred, pos_label=1)
    dice = 2 * jaccard / (1 + jaccard)

    return jaccard, dice
