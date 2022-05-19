"""
Wrong Side Driving Detection using YOLOv4

Program to utilise our custom trained model's weights
to perform vehicle detections on a frame in the numpy-array format
Program is present in the darknet folder(and not utils)
so that its convenient for us to use Darknet's helper modules and scripts

Done By,
Sriram N C
Srinandan KS
Jyoti Shetty
R.V. College of Engineering, Bangalore
"""

import os

from numpy import array

import scripts.darknet as darknet
import cv2
import sys
import matplotlib.pyplot as plt
sys.path.append("..")
from utils.constants import *
from utils.annotate_img import annotate_img

config_file = "cfg/yolov4-obj.cfg"
data_file = "data/obj.data"
weights = "../weights/yolov4-obj_last.weights"


def centroid(lx: float, ty: float, w: float, h: float) -> tuple:
    """
    Compute centroid of a bounding box given its coordinates, origin of the cartesian system is upper left corner, right is +x, down is +y axes
    :param lx: leftmost x coordinate
    :param ty: topmost horizontal line y coordinate
    :param w: width of bounding box
    :param h: height of bounding box
    :return: centroid coordinates
    """
    return lx + w / 2, ty + h / 2


def convert2relative(bbox: tuple, darknet_width: float, darknet_height: float) -> tuple:
    """
    YOLO format uses relative coordinates for annotation
    Function converts bounding box to relative coordinates
    :param bbox: 4-tuple of bounding box coordinates (x-origin, y-origin, width, height)
    :param darknet_width: width of darknet frame
    :param darknet_height: height of darknet frame
    :return: relative coordinates
    """

    x, y, w, h = bbox
    _height = darknet_height
    _width = darknet_width
    return x / _width, y / _height, w / _width, h / _height


def convert2original(image: array, bbox: tuple, darknet_width: float, darknet_height: float) -> tuple:
    """
    Converts relative coordinates obtained above to coordinates corresponding to the original image
    :param image: numpy array representing the image or video frame
    :param bbox: bounding box of an object
    :param darknet_width: width of darknet frame
    :param darknet_height: height of darknet frame
    :return:
    """
    x, y, w, h = convert2relative(bbox, darknet_width, darknet_height)

    image_h, image_w, __ = image.shape  # get original image dimensions

    orig_x = int(x * image_w)
    orig_y = int(y * image_h)
    orig_width = int(w * image_w)
    orig_height = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def load_darknet_network() -> tuple:
    """
    A run-once function to load the pre-programmed darknet neural network into the CPU before framewise object detection
    :return: tuple consisting of the darknet network, names of detected classes, colors to be used for those classes' bounding boxes
    and the darknet frame's height and width
    """
    d = os.path.abspath(os.getcwd())  # so that we are in the correct directory to access weights and network files
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # change to darknet folder
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=1
    )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    os.chdir(d)  # change to original directory
    return network, class_names, class_colors, darknet_width, darknet_height


def predict_frame(frame: array, network, class_names, class_colors, darknet_width: float, darknet_height: float, compute_output_image: bool = False) -> tuple:
    """
    Run darknet predictions based on given parameters
    :param frame: image or video frame on which to run our custom trained object detections
    :param network: loaded darknet network
    :param class_names: names of classes taken into consideration for object detection, loaded from a text file
    :param class_colors: colors to be used for those classes' bounding boxes
    :param darknet_width: darknet frame's width
    :param darknet_height: darknet frame's height
    :param compute_output_image: bool variable, if true compute the image (as a np array) with the bounding boxes drawn on them,
    False by default since we usually only need the centroid coordinates
    :return:
    """
    d = os.path.abspath(os.getcwd())
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                               interpolation=cv2.INTER_LINEAR)
    img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

    detections = darknet.detect_image(network, class_names, img_for_detect)
    detections_adjusted = []
    for label, confidence, bbox in detections:
        bbox_adjusted = convert2original(frame, bbox, darknet_width, darknet_height)
        detections_adjusted.append((str(label), confidence, bbox_adjusted))
    if compute_output_image:
        image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
    else:
        image = None
    os.chdir(d)
    return image, detections_adjusted


def centroid_calculation(detections: list) -> dict:
    """
    Custom function to calculate centroids as per our wrong side detection project requirements
    :param detections: list of object detections, consisting of bounding box coordinates
    :return: a dictionary consisting of centroids of detected objects
    """
    prediction = []
    for detection in detections:
        mydict = {"Accuracy": detection[1], "left_x": detection[2][0], "top_y": detection[2][1],
                  "width": detection[2][2],
                  "height": detection[2][3]}
        if int(mydict["width"]) > MAX_WIDTH or int(mydict["height"]) > MAX_HEIGHT:
            continue

        mydict["centroid"] = centroid(int(mydict["left_x"]), int(mydict["top_y"]), int(mydict["width"]),
                                      int(mydict["height"]))

        if len(prediction) and abs(float(prediction[-1]["left_x"]) - float(mydict["left_x"])) < MIN_DIST and abs(
                float(prediction[-1]["top_y"]) - float(mydict["top_y"])) < MIN_DIST:
            continue

        prediction.append(mydict)

    returned_dict = {'all_data': prediction}
    return returned_dict


def convert_detections(detections: list) -> list:
    """
    Convert to bounding box format
    :param detections: detections contains centroid coordinates and height and width of the frame
    :return: New list of detections, this time with bounding boxes
    """

    new_detections = []
    for detection in detections:
        new_detection = [detection[0], detection[1]]
        coordinates = []
        x, y, w, h = detection[2]
        coordinates.append(x-w/2)
        coordinates.append(y-h/2)
        coordinates.append(w)
        coordinates.append(h)
        new_detection.append(tuple(coordinates))
        new_detections.append(new_detection)

    return new_detections


def main():
    network, class_names, class_colors, darknet_width, darknet_height = load_darknet_network()
    darknet_image = "../trial.png"

    frame = cv2.imread(darknet_image)
    image, detections = predict_frame(frame, network, class_names, class_colors, darknet_width, darknet_height,
                                      compute_output_image=True)

    detections = convert_detections(detections)
    preds = centroid_calculation(detections)['all_data']
    obj_id = 0
    centroid_dict = {}
    for obj in preds:
        centroid_dict[obj_id] = obj['centroid']
        obj_id += 1
    # cv2.imwrite("tmp_pred3.jpg", image)
    final_image = annotate_img(image, {}, (), (), centroid_dict=centroid_dict)
    plt.imshow(final_image)
    plt.show()


if __name__ == '__main__':
    main()
