from skimage.feature import ORB, match_descriptors, plot_matched_features
import matplotlib.pyplot as plt
from skimage.transform import ProjectiveTransform, warp, resize
from skimage.measure import ransac
from skimage.util import img_as_ubyte, crop
import cv2


def detect_features(descriptor_type, image):
    if descriptor_type == "ORB":
        orb = ORB(n_keypoints=250)

        # Detect features and descriptors
        orb.detect_and_extract(image)
        keypoints = orb.keypoints
        descriptors = orb.descriptors

    elif descriptor_type == "SIFT":
        pass
    elif descriptor_type == "SURF":
        pass

    return descriptors, keypoints


def match_features(src_descriptors, dst_descriptors, src_keypoints, dst_keypoints):
    matches = match_descriptors(src_descriptors, dst_descriptors, cross_check=True)
    src_matches = src_keypoints[matches[:, 0]]
    dst_matches = dst_keypoints[matches[:, 1]]
    avg_num_of_matches_per_dst_keypoint = (
        len(matches) / len(dst_keypoints) if len(dst_keypoints) else 0.0
    )

    return src_matches, dst_matches, matches, avg_num_of_matches_per_dst_keypoint


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


def compute_transformation(src_matches, dst_matches, seed):
    transform_model, inliers = ransac(
        (src_matches, dst_matches),
        ProjectiveTransform,
        min_samples=4,
        residual_threshold=2,
        max_trials=1000,
        rng=seed,
    )

    return transform_model


def transform_image(transform_model, image1, image2):

    transformed_image = warp(
        image1, transform_model.inverse, output_shape=image2.shape, preserve_range=False
    )
    new_image = img_as_ubyte(transformed_image)

    return new_image


###################
# MAIN FUNCTIONS
##################


def pre_process_image(image):
    processed_image = image
    return processed_image


def feature_match(src_image, dst_image):
    src_descriptors, src_keypoints = detect_features("ORB", src_image)
    dst_descriptors, dst_keypoints = detect_features("ORB", dst_image)

    src_matches, dst_matches, matches, avg_num_of_matches_per_dst_keypoint = (
        match_features(src_descriptors, dst_descriptors, src_keypoints, dst_keypoints)
    )

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


def feature_match_with_crop(src_image, dst_image):

    cropped_dst_image = crop(dst_image, ((120, 120), (120, 200)))

    src_descriptors, src_keypoints = detect_features("ORB", src_image)
    dst_descriptors, dst_keypoints = detect_features("ORB", cropped_dst_image)

    # Shift the values of dst_keypoints by the amount taken off the starting axes
    # so that the pixel position aligns with the original image (from before cropping)
    shifted_dst_keypoints = dst_keypoints + [120, 120]

    src_matches, dst_matches, matches, avg_num_of_matches_per_dst_keypoint = (
        match_features(
            src_descriptors, dst_descriptors, src_keypoints, shifted_dst_keypoints
        )
    )
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


def transform(src_matches, dst_matches, src_image, dst_image, seed):
    """Transforms the COLOUR image using the defined projective transform model"""
    transform_model = compute_transformation(src_matches, dst_matches, seed)
    transformed_image = transform_image(transform_model, src_image, dst_image)

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB))
    ax.set_title("Target Reference Image")

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    ax.set_title("Registered Image")

    return transformed_image, transform_model
