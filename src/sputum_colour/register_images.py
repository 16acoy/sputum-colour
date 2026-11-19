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

    elif descriptor_type == "SIFT" or descriptor_type == "SURF":
        pass

    return descriptors, keypoints

def remove_clumps(keypoints, descriptors, dist, num_close):
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

    return valid_keypoints, valid_descriptors


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

    print("num target keypoints in matches: ", len(dst_keypoints))
    print("num src keypoints in matches: ", len(src_keypoints))
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
    inlier_votes = np.zeros(len(src_matches), dtype=np.uint8)
    for ransac_seed in (seed, seed + 5, seed + 10, seed + 15, seed+20):
        _, inliers = ransac(
            (src_matches, dst_matches),
            ProjectiveTransform,
            min_samples=4,
            residual_threshold=8,
            max_trials=2000,
            rng=ransac_seed,
        )
        if inliers is None or len(inliers) != len(src_matches):
            raise RuntimeError(
                f"RANSAC with seed {ransac_seed} did not return a valid inlier mask"
            )
        inlier_votes += np.asarray(inliers, dtype=np.uint8)

    inliers = inlier_votes >= 3
    inlier_src_matches = src_matches[inliers]
    inlier_dst_matches = dst_matches[inliers]
    if len(inlier_src_matches) < 4:
        raise ValueError(
            "At least four correspondences must be inliers in 3 or more "
            "RANSAC runs to estimate a homography"
        )

    inlier_indices = np.flatnonzero(inliers)
    inlier_matches = np.column_stack((inlier_indices, inlier_indices))
    print("num of consensus ransac inlier keypoints: ", len(inlier_matches))

    cv_src_inliers = inlier_src_matches[:, [1, 0]].astype(np.float32)
    cv_dst_inliers = inlier_dst_matches[:, [1, 0]].astype(np.float32)

    Hmatrix, mask = cv2.findHomography(cv_src_inliers, cv_dst_inliers, method=0)
    if Hmatrix is None:
        raise RuntimeError("Could not estimate a homography from consensus inliers")

    return Hmatrix, inliers, inlier_matches


@monitored
def transform_image(transform_matrix, image1, image2):

    # Warp source image into destination image coordinates
    warped = cv2.warpPerspective(
        image1, transform_matrix, (image2.shape[1], image2.shape[0])
    )

    # new_image = img_as_ubyte(warped)

    return warped


###################
# MAIN FUNCTIONS
##################


@monitored
def pre_process_image(image):

    closed = closing(image)
    blurred = cv2.GaussianBlur(closed, (11, 11), 0)
    blurred = cv2.GaussianBlur(blurred, (11, 11), 0)
    blurred = cv2.GaussianBlur(blurred, (11, 11), 0)
    blurred = cv2.GaussianBlur(blurred, (11, 11), 0)

    return blurred

@monitored
def feature_match(
    src_image,
    dst_image,
    crop,
    descriptor_type,
    orb_fast_n,
    orb_fast_threshold,
    orb_harris_k,
    filter_out_clumps,
    clump_dist,
    clump_num_close,
    feature_location_constraint,
):

    if crop != None:
        cropped_dst_image = util.crop(dst_image, crop)

    if feature_location_constraint == None:
        src_descriptors, src_keypoints = detect_features(
            descriptor_type,
            src_image,
            orb_fast_n,
            orb_fast_threshold,
            orb_harris_k,
        )
        dst_descriptors, dst_keypoints = detect_features(
            descriptor_type,
            cropped_dst_image,
            orb_fast_n,
            orb_fast_threshold,
            orb_harris_k,
        )

        if filter_out_clumps:
            src_keypoints_valid, src_descriptors_valid = remove_clumps(src_keypoints, src_descriptors, clump_dist, clump_num_close)
            dst_keypoints_valid, dst_descriptors_valid = remove_clumps(dst_keypoints, dst_descriptors, clump_dist, clump_num_close)
        else:
            src_keypoints_valid = src_keypoints
            src_descriptors_valid = src_descriptors
            dst_keypoints_valid = dst_keypoints
            dst_descriptors_valid = dst_descriptors

        # Shift the values of keypoints by the amount taken off the starting axes
        # so that the pixel position aligns with the original image (from before cropping)
        shifted_dst_keypoints_valid = dst_keypoints_valid + [crop[0][0], crop[1][0]]

        src_matches, dst_matches, matches, avg_num_of_matches_per_dst_keypoint = (
            match_features(
                src_descriptors_valid, dst_descriptors_valid, src_keypoints_valid, shifted_dst_keypoints_valid
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

        src_descriptor_clumps = []
        dst_descriptor_clumps = []
        src_keypoint_clumps = []
        dst_keypoint_clumps = []

        for y0, y1 in ranges_y:
            for x0, x1 in ranges_x:
                src_cell = src_image[int(y0) : int(y1), int(x0) : int(x1)]
                dst_cell = cropped_dst_image[int(y0) : int(y1), int(x0) : int(x1)]

                src_cell_descriptors, src_cell_keypoints = detect_features(
                    descriptor_type,
                    src_cell,
                    orb_fast_n,
                    orb_fast_threshold,
                    orb_harris_k,
                )
                dst_cell_descriptors, dst_cell_keypoints = detect_features(
                    descriptor_type,
                    dst_cell,
                    orb_fast_n,
                    orb_fast_threshold,
                    orb_harris_k,
                )

                src_descriptor_clumps.append(src_cell_descriptors)
                dst_descriptor_clumps.append(dst_cell_descriptors)
                src_keypoint_clumps.append(src_cell_keypoints + [int(y0), int(x0)])
                dst_keypoint_clumps.append(dst_cell_keypoints + [int(y0), int(x0)])

        src_descriptors = np.concatenate(src_descriptor_clumps, axis=0)
        dst_descriptors = np.concatenate(dst_descriptor_clumps, axis=0)
        src_keypoints = np.concatenate(src_keypoint_clumps, axis=0)
        dst_keypoints = np.concatenate(dst_keypoint_clumps, axis=0)

        print('num source keypoints: ', len(src_keypoints))


        if filter_out_clumps:
            src_keypoints_valid, src_descriptors_valid = remove_clumps(src_keypoints, src_descriptors, clump_dist, clump_num_close)
            dst_keypoints_valid, dst_descriptors_valid = remove_clumps(dst_keypoints, dst_descriptors, clump_dist, clump_num_close)
        else:
            src_keypoints_valid = src_keypoints
            src_descriptors_valid = src_descriptors
            dst_keypoints_valid = dst_keypoints
            dst_descriptors_valid = dst_descriptors

        print('num source keypoints after clump filtering: ', len(src_keypoints))

        all_matches = []
        all_avgs = []

        for y0, y1 in ranges_y:
            for x0, x1 in ranges_x:
                src_indices = np.flatnonzero(
                    (src_keypoints_valid[:, 0] >= y0)
                    & (src_keypoints_valid[:, 0] < y1)
                    & (src_keypoints_valid[:, 1] >= x0)
                    & (src_keypoints_valid[:, 1] < x1)
                )
                dst_indices = np.flatnonzero(
                    (dst_keypoints_valid[:, 0] >= y0)
                    & (dst_keypoints_valid[:, 0] < y1)
                    & (dst_keypoints_valid[:, 1] >= x0)
                    & (dst_keypoints_valid[:, 1] < x1)
                )

                _, _, clump_matches, avg_num_of_matches_per_dst_keypoint = (
                    match_features(
                        src_descriptors_valid[src_indices],
                        dst_descriptors_valid[dst_indices],
                        src_keypoints_valid[src_indices],
                        dst_keypoints_valid[dst_indices],
                    )
                )

                all_matches.append(
                    np.column_stack(
                        (
                            src_indices[clump_matches[:, 0]],
                            dst_indices[clump_matches[:, 1]],
                        )
                    )
                )
                all_avgs.append(avg_num_of_matches_per_dst_keypoint)

        matches = np.concatenate(all_matches, axis=0)

        src_matches = src_keypoints_valid[matches[:, 0]]
        shifted_dst_keypoints_valid = dst_keypoints_valid + [crop[0][0], crop[1][0]]
        dst_matches = shifted_dst_keypoints_valid[matches[:, 1]]

        avg_num_of_matches_per_dst_keypoint = sum(all_avgs) / len(all_avgs)

    fig = check_feature_matches(
        src_image, dst_image, matches, src_keypoints_valid, shifted_dst_keypoints_valid
    )

    return (
        src_matches,
        dst_matches,
        len(src_keypoints_valid),
        avg_num_of_matches_per_dst_keypoint,
        fig,
    )


@monitored
def transform(src_matches, dst_matches, src_image, dst_image, seed):
    """Transforms the COLOUR image using the defined projective transform model"""
    transform_matrix, inliers, inlier_matches = compute_transformation(
        src_matches, dst_matches, seed
    )

    check_feature_matches(
        src_image,
        dst_image,
        inlier_matches,
        src_matches,
        dst_matches,
    )

    transformed_image = transform_image(transform_matrix, src_image, dst_image)

    fig, ax = plt.subplots()
    ax.imshow(transformed_image)
    ax.set_title("Registered Image")

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB))
    ax.set_title("Target Reference Image")

    return transformed_image, transform_matrix
