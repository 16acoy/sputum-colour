from sputum_colour import data_loading, edge_line_detection, tube_roi_extraction
import cv2
import numpy as np
import os
import pandas as pd
import json
from cv2 import imwrite, line


def main(
    sample_range,
    blur_ksize,
    edge_detector_type,
    canny_L2gradient,
    canny_threshold1,
    canny_threshold2,
    canny_aperture_size,
    dog_param1,
    hough_shape,
    hough_line_rho,
    hough_line_theta,
    hough_line_threshold,
    hough_line_min_line_length,
    hough_line_max_line_gap,
    hough_circle_param1,
    orientation_tolerance_angles,
    test_tube_vert_t_low,
    test_tube_vert_t_high,
    test_tube_hor_t_low,
    test_tube_hor_t_high,
    valid_line_filtering_method,
    results_dir="/Users/ameliecoy/GitRepos/sputum-colour/results",
    seed=16,
):

    cv2.setRNGSeed(seed)

    paths_df = pd.read_csv("../files/paths.csv")
    target_image_path = paths_df[paths_df["path_type"] == "target_image_raw"].iloc[0][
        "path"
    ]
    dst_image, dst_image_grey = data_loading.imread_compatible(target_image_path)

    dir_path = f"{results_dir}/transformed_images"
    num_images_available = len(os.listdir(dir_path))

    all_metadata = json.load(open(f"{results_dir}/metadata.json"))
    eval_paths_df = pd.read_csv("../files/roi_eval_subset.csv")
    eval_subset_filenames = eval_paths_df["filename"].values

    if sample_range == None:
        sample_range_list = range(0, len(num_images_available))
    elif type(sample_range) == tuple:
        sample_range_list = range(sample_range[0], sample_range[1])
    elif type(sample_range) == list:
        sample_range_list = sample_range
    else:
        print("sample range not valid")

    for i in sample_range_list:
        src_image, src_image_grey = data_loading.imread_compatible(
            f"{dir_path}/image_{i}.jpg"
        )

        # Get the rough dst image masks
        tube_target_mask, card_target_mask = data_loading.get_rough_masks(dst_image)

        edges_image = edge_line_detection.edge_detect(
            src_image_grey,
            blur_ksize=blur_ksize,
            edge_detector_type=edge_detector_type,
            canny_L2gradient=canny_L2gradient,
            canny_threshold1=canny_threshold1,
            canny_threshold2=canny_threshold2,
            canny_aperture_size=canny_aperture_size,
        )

        lines_in_tube_mask, num_lines_total = (
            edge_line_detection.hough_detect_for_test_tube(
                edges_image,
                tube_target_mask,
                hough_shape=hough_shape,
                rho=hough_line_rho,
                theta=hough_line_theta,
                hough_threshold=hough_line_threshold,
                min_line_length=hough_line_min_line_length,
                max_line_gap=hough_line_max_line_gap,
                src_image=src_image_grey,
            )
        )
        if lines_in_tube_mask != None:
            # Draw hough lines on image for eval
            lines_image = src_image.copy()
            for linee in lines_in_tube_mask:
                x1, y1, x2, y2 = linee.tolist()
                lines_image = line(lines_image, (x1, y1), (x2, y2), (0, 0, 255))

            (
                src_midpoint_pixel,
                num_vert_lines,
                num_hor_lines,
                vert_lines_fig,
                hor_lines_fig,
            ) = tube_roi_extraction.get_valid_tube_midpoint(
                src_image_grey,
                lines_in_tube_mask,
                orientation_tolerance_angles,
                test_tube_vert_t_low,
                test_tube_vert_t_high,
                test_tube_hor_t_low,
                test_tube_hor_t_high,
                valid_line_filtering_method,
            )
            translated_mask = tube_roi_extraction.tube_to_tube_translate(
                src_midpoint_pixel, dst_image
            )
        else:
            translated_mask = tube_roi_extraction.tube_to_tube_translate(
                (None, None), dst_image
            )

        # ROI extraction on COLOUR image
        final_roi_on_image = tube_roi_extraction.final_roi_extraction(
            src_image, translated_mask
        )  # Could also inverse transform this back if want to visualise

        # Evaluate final similarities to ground truth label annotations
        inverse_transformed_mask = tube_roi_extraction.transform_predicted_roi(
            f"{results_dir}/transform_model_objects/image_{i}.pkl",
            translated_mask,
        )
        filename = all_metadata[f"image_{i}.jpg"]["filename"]
        site = all_metadata[f"image_{i}.jpg"]["site"]
        roi_ground_truth_label = data_loading.get_roi_ground_truth_label(
            filename, site, src_image, paths_df
        )

        jaccard, dice = tube_roi_extraction.get_final_eval_metrics(
            inverse_transformed_mask, roi_ground_truth_label
        )

        # Save eval metrics and images to results dir for this image
        eval_metrics_df = pd.DataFrame(
            [
                {
                    "num_hough_lines_total": num_lines_total,
                    "num_hough_lines_in_region": len(lines_in_tube_mask),
                    "num_valid_pairs_vertical": num_vert_lines,
                    "num_valid_pairs_horizontal": num_hor_lines,
                    "IoU": jaccard,
                    "Dice": dice,
                }
            ]
        )

        # Save ONLY subset images to results dir

        if filename in eval_subset_filenames:
            success = imwrite(f"{results_dir}/edge_detection/{filename}", edges_image)
            if success == 0:
                print(f"Error saving edges image for {filename}")

            success = imwrite(f"{results_dir}/hough_lines/{filename}", lines_image)
            if success == 0:
                print(f"Error saving hough line image for {filename}")

            vert_lines_fig.savefig(
                f"{results_dir}/valid_vertical_hough_lines/{filename}"
            )
            hor_lines_fig.savefig(
                f"{results_dir}/valid_horizontal_hough_lines/{filename}"
            )

            success = imwrite(f"{results_dir}/final_roi/{filename}", final_roi_on_image)
            if success == 0:
                print(f"Error saving final ROI image for {filename}")

        eval_metrics_df.to_csv(
            f"{results_dir}/intermediate_results/eval_results_stage2_{filename}.csv",
            index=False,
        )

        # Save all masks to roi_masks directory for donwstream use
        success = imwrite(
            f"{results_dir}/roi_masks/image_{i}.jpg",
            inverse_transformed_mask,
        )
        if success == 0:
            print(f"Error saving image_{i}.jpg")


if __name__ == "__main__":
    main(
        sample_range=None,
        blur_ksize=None,
        edge_detector_type=None,
        canny_L2gradient=None,
        canny_threshold1=None,
        canny_threshold2=None,
        canny_aperture_size=None,
        dog_param1=None,
        hough_shape=None,
        hough_line_rho=None,
        hough_line_theta=None,
        hough_line_threshold=None,
        hough_line_min_line_length=None,
        hough_line_max_line_gap=None,
        hough_circle_param1=None,
        orientation_tolerance_angles=None,
        test_tube_vert_t_low=None,
        test_tube_vert_t_high=None,
        test_tube_hor_t_low=None,
        test_tube_hor_t_high=None,
        valid_line_filtering_method=None,
        results_dir="/results",
        seed=16,
    )
