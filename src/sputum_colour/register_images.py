import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import ORB, match_descriptors, plot_matched_features
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, warp
from skimage import util
from skimage.util import img_as_ubyte
from skimage.morphology import dilation, closing

from sputum_colour.performance import monitored


@monitored
def detect_features(
    descriptor_type,
    image,
    orb_fast_n,
    orb_fast_threshold,
    orb_harris_k,
    filter_out_clumps,
):

    if descriptor_type == "ORB":
        orb_params = {
            name: value
            for name, value in {
                "fast_n": orb_fast_n,
                "fast_threshold": orb_fast_threshold,
                "harris_k": orb_harris_k,
            }.items()
            if value is not None
        }

        # Detect features and descriptors - reduce fast_threshold if none found
        for tries in range(3):
            try:
                orb = ORB(n_keypoints=250, **orb_params)
                orb.detect_and_extract(image)
            except RuntimeError:
                orb_params['fast_threshold'] -= 0.01

        keypoints = orb.keypoints
        descriptors = orb.descriptors

        if len(keypoints) == 0:
            pass
            # Fallback to failure if needed??

        if filter_out_clumps:
            dist = 200
            num_close = 20
            not_valid_indices = set()
            for index, keypoint in enumerate(keypoints):
                distances = np.linalg.norm(keypoints - keypoint, axis=1)
                if np.count_nonzero(distances < dist) >= num_close:
                    not_valid_indices.add(index)

            valid_indices = np.array(
                [
                    index
                    for index in range(len(keypoints))
                    if index not in not_valid_indices
                ],
                dtype=int,
            )
            valid_keypoints = keypoints[valid_indices]
            valid_descriptors = descriptors[valid_indices]

        else:
            valid_keypoints = keypoints
            valid_descriptors = descriptors

    elif descriptor_type == "SIFT" or descriptor_type == "SURF":
        pass

    return valid_descriptors, valid_keypoints


@monitored
def match_features(src_descriptors, dst_descriptors, src_keypoints, dst_keypoints):
    matches = match_descriptors(
        src_descriptors, dst_descriptors, max_distance=128, cross_check=True
    )
    src_matches = src_keypoints[matches[:, 0]]  # acc keypoint coords
    dst_matches = dst_keypoints[matches[:, 1]]
    avg_num_of_matches_per_dst_keypoint = (
        len(matches) / len(dst_keypoints) if len(dst_keypoints) else 0.0
    )

    print("num target keypoints: ", len(dst_keypoints))
    print("num src keypoints: ", len(src_keypoints))
    print("num matches: ", len(matches))

    return src_matches, dst_matches, matches, avg_num_of_matches_per_dst_keypoint


@monitored
def check_feature_matches(src_image, dst_image, matches, src_keypoints, dst_keypoints):

    # Plot keypoints and matches
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt.gray()
    plot_matched_features(
        src_image,
        dst_image,
        keypoints0=src_keypoints,
        keypoints1=dst_keypoints,
        matches=matches,
        ax=ax,
    )
    ax.set_title("Keypoint Matches")
    plt.show()

    return fig


@monitored
def compute_transformation(src_matches, dst_matches, seed):
    transform_model, inliers = ransac(
        (src_matches, dst_matches),
        ProjectiveTransform,
        min_samples=4,
        residual_threshold=2,
        max_trials=2000,
        rng=seed,
    )

    return transform_model, inliers


@monitored
def transform_image(transform_model, image1, image2):

    transformed_image = warp(
        image1, transform_model.inverse, output_shape=image2.shape, preserve_range=False
    )
    new_image = img_as_ubyte(transformed_image)

    return new_image


###################
# MAIN FUNCTIONS
##################


@monitored
def pre_process_image(image):
    """
    closed = closing(image)
    for i in range(20):
        closed = closing(closed)
    blurred = cv2.GaussianBlur(closed, (11, 11), 0)

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    ax.set_title("Closed and blurred image")
    """
    closed = closing(image)
    # alpha = -1.8 # Contrast control (1.0-3.0)
    # adjusted = cv2.convertScaleAbs(closed, alpha=alpha, beta=0)
    blurred = cv2.GaussianBlur(closed, (11, 11), 0)
    blurred = cv2.GaussianBlur(blurred, (11, 11), 0)
    blurred = cv2.GaussianBlur(blurred, (11, 11), 0)
    blurred = cv2.GaussianBlur(blurred, (11, 11), 0)

    return blurred


@monitored
def feature_match(
    src_image,
    dst_image,
    descriptor_type,
    orb_fast_n,
    orb_fast_threshold,
    orb_harris_k,
    filter_out_clumps,
    feature_location_constraint,
):

    if feature_location_constraint == None:
        src_descriptors, src_keypoints = detect_features(
            descriptor_type,
            src_image,
            orb_fast_n,
            orb_fast_threshold,
            orb_harris_k,
            filter_out_clumps,
        )
        dst_descriptors, dst_keypoints = detect_features(
            descriptor_type,
            dst_image,
            orb_fast_n,
            orb_fast_threshold,
            orb_harris_k,
            filter_out_clumps,
        )

        src_matches, dst_matches, matches, avg_num_of_matches_per_dst_keypoint = (
            match_features(
                src_descriptors, dst_descriptors, src_keypoints, dst_keypoints
            )
        )
    else:
        # feature_location_constraint contains (split_x, split_y) where split__ is the number of sections to split each axis of the image into
        split_x = feature_location_constraint[0]
        cell_size_x = src_image.shape[1] / split_x
        ranges_x = []
        for i in range(0, split_x):
            ranges_x.append(
                (i * cell_size_x, (i + 1) * cell_size_x)
            )  # if inclusive:exclusive

        split_y = feature_location_constraint[1]
        cell_size_y = src_image.shape[0] / split_y
        ranges_y = []
        for i in range(0, split_y):
            ranges_y.append(
                (i * cell_size_y, (i + 1) * cell_size_y)
            )  # if inclusive:exclusive

        all_src_keypoints = []
        all_dst_keypoints = []
        all_matches = []
        src_offset = 0
        dst_offset = 0
        all_avgs = 0

        for y0, y1 in ranges_y:
            for x0, x1 in ranges_x:
                src_cell = src_image[int(y0) : int(y1), int(x0) : int(x1)]
                dst_cell = dst_image[int(y0) : int(y1), int(x0) : int(x1)]

                src_descriptors, src_keypoints = detect_features(
                    descriptor_type,
                    src_cell,
                    orb_fast_n,
                    orb_fast_threshold,
                    orb_harris_k,
                    filter_out_clumps,
                )
                dst_descriptors, dst_keypoints = detect_features(
                    descriptor_type,
                    dst_cell,
                    orb_fast_n,
                    orb_fast_threshold,
                    orb_harris_k,
                    filter_out_clumps,
                )
                _, _, matches, avg_num_of_matches_per_dst_keypoint = match_features(
                    src_descriptors, dst_descriptors, src_keypoints, dst_keypoints
                )

                all_src_keypoints.append(src_keypoints + [int(y0), int(x0)])
                all_dst_keypoints.append(dst_keypoints + [int(y0), int(x0)])

                all_matches.append(matches + [src_offset, dst_offset])
                src_offset += len(src_keypoints)
                dst_offset += len(dst_keypoints)
                all_avgs += avg_num_of_matches_per_dst_keypoint

        # Concat all keypoints + descs

        src_keypoints = np.concatenate(all_src_keypoints, axis=0)
        dst_keypoints = np.concatenate(all_dst_keypoints, axis=0)
        matches = np.concatenate(all_matches, axis=0)

        src_matches = src_keypoints[matches[:, 0]]
        dst_matches = dst_keypoints[matches[:, 1]]

        all_avgs = all_avgs / (split_x * split_y)

    fig = check_feature_matches(
        src_image, dst_image, matches, src_keypoints, dst_keypoints
    )

    return (
        src_matches,
        dst_matches,
        len(src_keypoints),
        avg_num_of_matches_per_dst_keypoint,
        fig,
    )


@monitored
def feature_match_with_crop(
    src_image,
    dst_image,
    crop,
    descriptor_type,
    orb_fast_n,
    orb_fast_threshold,
    orb_harris_k,
    filter_out_clumps,
    feature_location_constraint,
):

    if crop == 'fixed':

        # src_crop = ((130, 130), (130, 130))
        dst_crop = ((120, 120), (120, 200))
        # cropped_src_image = crop(src_image, src_crop)
        print("before crop")
        cropped_dst_image = util.crop(dst_image, dst_crop)
        print("after crop")

    elif crop == 'edge_detect':
        edge_image = cv2.Canny(
            src_image,
            threshold1=10,
            threshold2=50,
        )

        fig, ax = plt.subplots()
        ax.imshow(edge_image)
        ax.set_title("Edge Image")


        all_lines = cv2.HoughLinesP(
            edge_image,
            rho=1.0,
            theta=0.017453292519943295,
            threshold=35,
            minLineLength=100,
            maxLineGap=3000
        )

        src_copy = src_image.copy()

        # Draw all hough lines
        for line in all_lines:
            x1, y1, x2, y2 = line.tolist()
            cv2.line(src_copy, (x1, y1), (x2, y2), (0, 0, 0), 3)

        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(src_copy, cv2.COLOR_BGR2RGB))
        ax.set_title("All hough lines Image")


        line_lengths = []
        for line in all_lines:
            x1, y1, x2, y2 = line.tolist()
            length = np.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))
            line_lengths.append(length)
            print(length)

        # TODO - Sort return type and how its passed down

    if feature_location_constraint == None:
        src_descriptors, src_keypoints = detect_features(
            descriptor_type,
            src_image,
            orb_fast_n,
            orb_fast_threshold,
            orb_harris_k,
            filter_out_clumps,
        )
        dst_descriptors, dst_keypoints = detect_features(
            descriptor_type,
            cropped_dst_image,
            orb_fast_n,
            orb_fast_threshold,
            orb_harris_k,
            filter_out_clumps,
        )

        # Shift the values of keypoints by the amount taken off the starting axes
        # so that the pixel position aligns with the original image (from before cropping)
        shifted_dst_keypoints = dst_keypoints + [dst_crop[0][0], dst_crop[1][0]]
        # shifted_src_keypoints = src_keypoints + [src_crop[0][0], src_crop[1][0]]

        src_matches, dst_matches, matches, avg_num_of_matches_per_dst_keypoint = (
            match_features(
                src_descriptors, dst_descriptors, src_keypoints, shifted_dst_keypoints
            )
        )

    else:
        # feature_location_constraint contains (split_x, split_y) where split__ is the number of sections to split each axis of the image into
        split_x = feature_location_constraint[0]
        cell_size_x = src_image.shape[1] / split_x
        ranges_x = []
        for i in range(0, split_x):
            ranges_x.append(
                (i * cell_size_x, (i + 1) * cell_size_x)
            )  # if inclusive:exclusive

        split_y = feature_location_constraint[1]
        cell_size_y = src_image.shape[0] / split_y
        ranges_y = []
        for i in range(0, split_y):
            ranges_y.append(
                (i * cell_size_y, (i + 1) * cell_size_y)
            )  # if inclusive:exclusive

        all_src_keypoints = []
        all_dst_keypoints = []
        all_matches = []
        src_offset = 0
        dst_offset = 0
        all_avgs = 0

        for y0, y1 in ranges_y:
            for x0, x1 in ranges_x:
                src_cell = src_image[int(y0) : int(y1), int(x0) : int(x1)]
                dst_cell = cropped_dst_image[int(y0) : int(y1), int(x0) : int(x1)]

                src_descriptors, src_keypoints = detect_features(
                    descriptor_type,
                    src_cell,
                    orb_fast_n,
                    orb_fast_threshold,
                    orb_harris_k,
                    filter_out_clumps,
                )
                dst_descriptors, dst_keypoints = detect_features(
                    descriptor_type,
                    dst_cell,
                    orb_fast_n,
                    orb_fast_threshold,
                    orb_harris_k,
                    filter_out_clumps,
                )
                _, _, matches, avg_num_of_matches_per_dst_keypoint = match_features(
                    src_descriptors, dst_descriptors, src_keypoints, dst_keypoints
                )

                all_src_keypoints.append(src_keypoints + [int(y0), int(x0)])
                all_dst_keypoints.append(dst_keypoints + [int(y0), int(x0)])

                all_matches.append(matches + [src_offset, dst_offset])
                src_offset += len(src_keypoints)
                dst_offset += len(dst_keypoints)
                all_avgs += avg_num_of_matches_per_dst_keypoint

        # Concat all keypoints + descs

        src_keypoints = np.concatenate(all_src_keypoints, axis=0)
        dst_keypoints = np.concatenate(all_dst_keypoints, axis=0)
        shifted_dst_keypoints = dst_keypoints + [dst_crop[0][0], dst_crop[1][0]]

        matches = np.concatenate(all_matches, axis=0)

        src_matches = src_keypoints[matches[:, 0]]
        dst_matches = shifted_dst_keypoints[matches[:, 1]]

        all_avgs = all_avgs / (split_x * split_y)

    fig = check_feature_matches(
        src_image, dst_image, matches, src_keypoints, shifted_dst_keypoints
    )

    return (
        src_matches,
        dst_matches,
        len(src_keypoints),
        avg_num_of_matches_per_dst_keypoint,
        fig,
    )


@monitored
def transform(src_matches, dst_matches, src_image, dst_image, seed):
    """Transforms the COLOUR image using the defined projective transform model"""
    transform_model, inliers = compute_transformation(src_matches, dst_matches, seed)

    cv_src = src_matches[:, [1, 0]].astype(np.float32)
    cv_dst = dst_matches[:, [1, 0]].astype(np.float32)

    matrix1, mask = cv2.findHomography(
        cv_src, cv_dst, method=cv2.RANSAC, ransacReprojThreshold=1
    )

    # Warp source image into destination image coordinates
    warped1 = cv2.warpPerspective(src_image, matrix1, (3024, 4032))

    fig, ax = plt.subplots()
    ax.imshow(warped1)
    ax.set_title("OpenCV Warped Image (pre-ransac)")

    inlier_points = inliers == True
    inlier_src_matches = src_matches[inlier_points]
    inlier_dst_matches = dst_matches[inlier_points]
    inlier_matches = np.column_stack(
        (np.arange(len(inlier_src_matches)), np.arange(len(inlier_dst_matches)))
    )

    print("First RANSAC src marker position: ", inlier_src_matches[0])

    print(
        "First RANSAC src marker position, after transform: ",
        transform_model(inlier_src_matches[0]),
    )

    print(
        "First RANSAC dst marker position (desired): ",
        transform_model(inlier_dst_matches[0]),
    )

    cv_src_inliers = inlier_src_matches[:, [1, 0]].astype(np.float32)
    cv_dst_inliers = inlier_dst_matches[:, [1, 0]].astype(np.float32)

    matrix2, mask = cv2.findHomography(cv_src_inliers, cv_dst_inliers, method=0)

    # Warp source image into destination image coordinates
    warped2 = cv2.warpPerspective(src_image, matrix2, (3024, 4032))

    fig, ax = plt.subplots()
    ax.imshow(warped2)
    ax.set_title("OpenCV Warped Image (post-ransac)")

    check_feature_matches(
        src_image,
        dst_image,
        inlier_matches,
        inlier_src_matches,  # Only the matched kps will be plotted as kps
        inlier_dst_matches,  # Only the matched kps will be plotted as kps
    )

    transformed_image = transform_image(transform_model, src_image, dst_image)

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB))
    ax.set_title("Target Reference Image")

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    ax.set_title("Registered Image")

    return transformed_image, transform_model
