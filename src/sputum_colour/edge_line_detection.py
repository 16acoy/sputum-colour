import cv2
import matplotlib.pyplot as plt


def edge_detect(
    image,
    blur_ksize,
    edge_detector_type,
    canny_L2gradient,
    canny_threshold1,
    canny_threshold2,
    canny_aperture_size,
):

    blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

    if edge_detector_type == "Canny":
        if canny_L2gradient == None and canny_aperture_size == None:
            edge_only_image = cv2.Canny(
                blurred, threshold1=canny_threshold1, threshold2=canny_threshold2
            )
        elif canny_L2gradient == None:
            edge_only_image = cv2.Canny(
                blurred,
                threshold1=canny_threshold1,
                threshold2=canny_threshold2,
                apertureSize=canny_aperture_size,
            )
        elif canny_aperture_size == None:
            edge_only_image = cv2.Canny(
                blurred,
                threshold1=canny_threshold1,
                threshold2=canny_threshold2,
                L2gradient=canny_L2gradient,
            )
        else:
            edge_only_image = cv2.Canny(
                blurred,
                threshold1=canny_threshold1,
                threshold2=canny_threshold2,
                apertureSize=canny_aperture_size,
                L2gradient=canny_L2gradient,
            )

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Original Image")

        ax[1].imshow(edge_only_image, cmap="gray")
        ax[1].set_title("Edge Image")

        plt.tight_layout()
        plt.show()

    return edge_only_image


def hough_detect_for_test_tube(
    edge_only_image,
    rough_mask,
    hough_shape,
    rho,
    theta,
    hough_threshold,
    min_line_length,
    max_line_gap,
    src_image,
):

    if hough_shape == "lines":
        all_lines = cv2.HoughLinesP(
            edge_only_image,
            rho,
            theta,
            hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        src_copy = src_image.copy()

        # Draw all hough lines
        for line in all_lines:
            x1, y1, x2, y2 = line.tolist()
            cv2.line(src_copy, (x1, y1), (x2, y2), (0, 0, 0), 3)

        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(src_copy, cv2.COLOR_BGR2RGB))
        ax.set_title("All hough lines Image")

        if all_lines is not None:
            lines_in_tube_mask = []
            for line in all_lines:
                x1, y1, x2, y2 = line.tolist()
                # Check if start and end of detected line are within rough target mask for test tube
                if rough_mask[y1][x1] == 1 and rough_mask[y2][x2] == 1:
                    lines_in_tube_mask.append(line)
                    cv2.line(
                        edge_only_image, (x1, y1), (x2, y2), (255, 0, 0), 3
                    )  # Draw lines to check

            return lines_in_tube_mask, len(all_lines)

        else:
            return None, 0

    else:
        return None, 0
