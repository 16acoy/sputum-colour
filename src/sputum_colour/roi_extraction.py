import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import ORB, match_descriptors, plot_matched_features
from skimage.transform import ProjectiveTransform, warp, resize
from skimage.measure import ransac


# Resize images to be the same size
#image1 = resize(image1, (500, 500), anti_aliasing=True)
#image2 = resize(image2, (500, 500), anti_aliasing=True)

# TODO - Load images
# TODO - Pre-processing ?
# TODO - Research + edit all params
# TODO - DVC pipeline-arise

def pre_process_image(image):
    pass

def detect_features(descriptor_type, image):
    print(descriptor_type)
    print(image)
    if descriptor_type == 'ORB':
        orb = ORB(n_keypoints=500)

        # Detect features and descriptors
        orb.detect_and_extract(image)
        keypoints = orb.keypoints
        descriptors = orb.descriptors


    elif descriptor_type == 'SIFT':
        pass
    elif descriptor_type == 'SURF':
        pass

    return descriptors, keypoints

def match_features(dst_descriptors, src_descriptors, dst_keypoints, src_keypoints):
    matches = match_descriptors(dst_descriptors, src_descriptors, cross_check=True)
    dst_matches = dst_keypoints[matches[:, 0]]
    src_matches = src_keypoints[matches[:, 1]]

    return dst_matches, src_matches, matches


def check_feature_matches(dst_image, src_image, matches, dst_keypoints, src_keypoints):

    # Plot keypoints and matches
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt.gray()
    plot_matched_features(
    dst_image,
    src_image,
    keypoints0=dst_keypoints,
    keypoints1=src_keypoints,
    matches=matches,
    ax=ax
)
    ax.axis('off')
    ax.set_title("Keypoint Matches")
    plt.show()

    num_matches = len(matches)
    print(f"Number of matches: {num_matches}")
    if num_matches < 4:
        print("Too few matches!")

def compute_transformation(dst_matches, src_matches):
    transform_model, inliers = ransac((dst_matches, src_matches), ProjectiveTransform, min_samples=4, residual_threshold=2, max_trials=1000)

    return transform_model

def transform_image(transform_model, image1, image2):
    transformed_image = warp(image1, transform_model.inverse, output_shape=image2.shape)

    return transformed_image

def register_target_to_source_image(dst_image, src_image):

    dst_descriptors, dst_keypoints = detect_features('ORB', dst_image)
    src_descriptors, src_keypoints = detect_features('ORB', src_image)
    dst_matches, src_matches, matches = match_features(dst_descriptors, src_descriptors, dst_keypoints, src_keypoints)
    check_feature_matches(dst_image, src_image, matches, dst_keypoints, src_keypoints)
    transform_model = compute_transformation(dst_matches, src_matches)
    transformed_image = transform_image(transform_model, dst_image, src_image)
    plt.imshow(transformed_image, cmap='gray')



def extract_tube_rough_area():
    pass

def extract_card_rough_area():
    pass

def edge_detect():
    pass

def hough_line_detect():
    pass

def detect_test_tube():
    # Call edge detect, hough line detect, then filter for desired distance
    pass

def extract_roi():
    pass

def check_roi_noise():
    pass


# TODO - Sync uv.lock once tested all code works

#################################



def roi_extraction_pipeline(param1, param2):
    pass


