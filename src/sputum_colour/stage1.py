from sputum_colour import data_loading, register_images
import cv2
from cv2 import imwrite, polylines
import json

# import dvc.api
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np


def main(
    sample_range,
    crop,
    descriptor_type,
    orb_fast_n,
    orb_fast_threshold,
    orb_harris_k,
    filter_out_clumps,
    feature_location_constraint,
    feature_match_max_distance,
    feature_match_max_ratio,
    feature_match_cross_check,
    ransac_min_samples,
    ransac_residual_threshold,
    ransac_max_trials,
    warp_order,
    seed=16,
    results_dir="/Users/ameliecoy/GitRepos/sputum-colour/results",
    reg_method="feature_detection",
):

    cv2.setRNGSeed(seed)

    print("before data collection")
    (dst_image, dst_image_grey), all_files, all_metadata = (
        data_loading.data_collection()
    )
    print("after data collection")

    dst_image_grey_processed = register_images.pre_process_image(dst_image_grey)

    print("before reading files")
    eval_paths_df = pd.read_csv("../files/roi_eval_subset.csv")
    eval_subset_filenames = eval_paths_df["filename"].values
    print("after reading files")

    if sample_range == None:
        sample_range_list = range(0, len(all_files))
    elif type(sample_range) == tuple:
        sample_range_list = range(sample_range[0], sample_range[1])
    elif type(sample_range) == list:
        sample_range_list = sample_range
    else:
        print("sample range not valid")

    for i in sample_range_list:
        file_pair = all_files[i]
        src_image = file_pair[0]
        src_image_grey = file_pair[1]
        src_image_grey_processed = register_images.pre_process_image(src_image_grey)
        transformed_image = None
        num_keypoints = None
        avg_num_of_matches_per_dst_keypoint = None
        fig = None

        if reg_method == "intensity":
            img1 = dst_image_grey_processed.astype(np.float32) / 255.0
            img2 = src_image_grey_processed.astype(np.float32) / 255.0
            mapper = cv2.reg.MapperGradProj()
            mapper_pyramid = cv2.reg.MapperPyramid(mapper)
            transform_model = mapper_pyramid.calculate(img1, img2)
            print("Is transform model none?: ", transform_model == None)
            print(type(transform_model))

            # Projective transform of the COLOUR image
            transformed_image = transform_model.inverseWarp(src_image)

            fig, ax = plt.subplots()
            ax.imshow(cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB))
            ax.set_title("Original Image")

            fig, ax = plt.subplots()
            ax.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
            ax.set_title("Registered Image")

        elif reg_method == "feature_detection":
            if crop == True:
                (
                    src_matches,
                    dst_matches,
                    num_keypoints,
                    avg_num_of_matches_per_dst_keypoint,
                    fig,
                ) = register_images.feature_match_with_crop(
                    src_image_grey_processed,
                    dst_image_grey_processed,
                    descriptor_type,
                    orb_fast_n,
                    orb_fast_threshold,
                    orb_harris_k,
                    filter_out_clumps,
                    feature_location_constraint,
                )
            else:
                (
                    src_matches,
                    dst_matches,
                    num_keypoints,
                    avg_num_of_matches_per_dst_keypoint,
                    fig,
                ) = register_images.feature_match(
                    src_image_grey_processed,
                    dst_image_grey_processed,
                    descriptor_type,
                    orb_fast_n,
                    orb_fast_threshold,
                    orb_harris_k,
                    filter_out_clumps,
                    feature_location_constraint,
                )

            # Projective transform of the COLOUR images
            transformed_image, transform_model = register_images.transform(
                src_matches,
                dst_matches,
                src_image,
                dst_image,
                seed,
            )

        # Draw rough tube box onto transformed image for eval check
        paths_df = pd.read_csv("../files/paths.csv")
        rough_target_mask_path = paths_df[
            paths_df["path_type"] == "target_tube_bounding_box_ann"
        ].iloc[0]["path"]
        rough_target_mask_df = data_loading.load_label_json_file(rough_target_mask_path)
        rough_target_mask_row = rough_target_mask_df.iloc[0]
        x_points = rough_target_mask_row["regions"][0]["shape_attributes"][
            "all_points_x"
        ]
        y_points = rough_target_mask_row["regions"][0]["shape_attributes"][
            "all_points_y"
        ]
        point_array = data_loading.get_point_array(x_points, y_points)
        transformed_image_copy = transformed_image.copy()
        transformed_image_copy = polylines(
            transformed_image_copy, [point_array], isClosed=True, color=(0, 0, 255)
        )

        filename = all_metadata[f"image_{i}.jpg"]["filename"]

        # Save eval metrics and images to results dir for this image
        if num_keypoints != None and avg_num_of_matches_per_dst_keypoint != None:
            eval_metrics_df = pd.DataFrame(
                [
                    {
                        "num_keypoints": num_keypoints,
                        "avg_num_of_matches_per_target_keypoint": avg_num_of_matches_per_dst_keypoint,
                    }
                ]
            )
            eval_metrics_df.to_csv(
                f"{results_dir}/intermediate_results/eval_results_stage1_{filename}.csv",
                index=False,
            )

        # Save ONLY subset images to results dir
        print(eval_subset_filenames)
        if filename in eval_subset_filenames:
            if fig is not None:
                fig.savefig(f"{results_dir}/feature_matches/{filename}")

            success = imwrite(
                f"{results_dir}/tube_in_box/{filename}", transformed_image_copy
            )
            if success == 0:
                print(f"Error saving tube in box image for {filename}")

        # Save all transformed images to results/transformed_images with filenames in number order
        success = imwrite(
            f"{results_dir}/transformed_images/image_{i}.jpg",
            transformed_image,
        )
        if success == 0:
            print(f"Error saving image_{i}.jpg")

        # Save all transform models to results/transform_model_objects with filenames in number order
        try:
            with open(
                f"{results_dir}/transform_model_objects/image_{i}.pkl",
                "wb",
            ) as file:
                # Use pickle.dump to serialize the object and write to the file
                pickle.dump(transform_model, file)
            print(f"Object successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving object: {e}")

    # Also save metadata into one big json file
    saveable_metadata = {
        key: {
            k: v.isoformat() if isinstance(v, pd.Timestamp) else v
            for k, v in values.items()
        }
        for key, values in all_metadata.items()
    }
    with open(f"{results_dir}/metadata.json", "w") as file:
        json.dump(saveable_metadata, file)


if __name__ == "__main__":
    # params = dvc.api.params_show()
    # sample_range = params['sample_range'] # (x, y) for range of all images that we want to process, or None if want to use all
    main(
        sample_range=None,
        reg_method=None,
        crop=None,
        descriptor_type=None,
        orb_fast_n=None,
        orb_fast_threshold=None,
        orb_harris_k=None,
        feature_location_constraint=None,
        feature_match_max_distance=None,
        feature_match_max_ratio=None,
        feature_match_cross_check=None,
        ransac_min_samples=None,
        ransac_residual_threshold=None,
        ransac_max_trials=None,
        warp_order=None,
        seed=16,
        results_dir="/Users/ameliecoy/GitRepos/sputum-colour/results",
    )
