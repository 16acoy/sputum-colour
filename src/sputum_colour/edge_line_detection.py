import cv2
import matplotlib.pyplot as plt


def edge_detect(image, edge_detector_type, blur_type):
    print("pre blur")

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    if edge_detector_type == "Canny":
        edge_only_image = cv2.Canny(blurred, threshold1=40, threshold2=100)

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
    rho,
    theta,
    hough_threshold,
    min_line_length,
    max_line_gap,
):

    all_lines = cv2.HoughLinesP(
        edge_only_image,
        rho,
        theta,
        hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

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
