"""
Wrong Side Driving Detection using YOLOv4
Module for marking (annotating) centroids on detected cars
Done By,
Sriram N C
Srinandan KS
Jyoti Shetty
R.V. College of Engineering, Bangalore
"""

import cv2
import numpy as np
from numpy import array
from constants import *


def annotate_img(image: array, wrong_direction: dict, point1: tuple, point2: tuple, fps=1, show=False, save=True,
                 output_file="centroid_result.jpeg", centroid_dict=None, save_violation=False,
                 violations_output_file=None) -> array:

    """
    Annotate or mark a centroid of a detected object's bounding box in red or green color on an image and save or display the resultant image
    :param image: input image or video frame as a np array
    :param wrong_direction: Dictionary, keys are object IDs, values are non-zero if the object is penalised, zero otherwise
    :param point1: One point on the imaginary median calculated by the main program and passed to this function
    :param point2: Another point on the imaginary module
    :param fps: Frame rate of the video
    :param show: bool parameter, if true show the image using cv2.imshow(), do not set this to true on colab since colab does not support imshow
    :param save: bool parameter, if true save the frame as an image file
    :param output_file: if save is true, then this parameter is the output image filename
    :param centroid_dict: Dictionary, keys are centroid IDs assigned to detected objects by centroid tracking module, values are centroid coordinates
    :param save_violation: bool parameter, if true save the violations to image files
    :param violations_output_file: if save_violation is true, then this parameter is the output image filename
    :return: Resultant image as a np array
    """

    if not centroid_dict:
        centroid_dict = {}

    window_name = 'Image'
    violation = False

    if point1 != () and point2 != ():
        # Blue color in BGR
        color = (255, 0, 0)

        thickness = 5

        # image = cv2.line(image, start_point, end_point, color, thickness)
        pts = np.array([point1, point2], dtype=np.int32)
        image = cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)

    # Radius of circle
    radius = 5

    # Line thickness of -1 px
    thickness = -1

    # ids = [0 for i in range(len(centroids))]

    for obj_id, centroid in centroid_dict.items():
        centroid = tuple([int(x) for x in centroid])
        # convert tuple values to int to be used in the opencv circle function
        if obj_id in wrong_direction and wrong_direction[obj_id] > 1:
            # wrong direction, red circle color BGR
            violation = True
            circle_color = (0, 0, 255)
        else:
            # green
            circle_color = (0, 255, 0)

        image = cv2.circle(image, centroid, radius, circle_color, thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 1
        org = (centroid[0], centroid[1] - 10)  # id number text starts 10 units below centroid
        image = cv2.putText(image, str(obj_id), org, font, font_scale, circle_color, font_thickness, cv2.LINE_AA)

    if show:
        cv2.imshow(window_name, image)
        if fps == 1:
            cv2.waitKey(1000)

        else:
            cv2.waitKey(int(1000 / fps) - int(CV2_OFFSET * (30 / fps)))

    if save:
        cv2.imwrite(output_file, image)

    if violation and violations_output_file and save_violation:
        cv2.imwrite(violations_output_file, image)

    return image


def main():
    path = 'agera.jpeg'
    image = cv2.imread(path)
    annotate_img(image, {}, (), (), save=True, show=True)


if __name__ == '__main__':
    main()
