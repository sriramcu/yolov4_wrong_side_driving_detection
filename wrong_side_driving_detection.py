"""
Wrong Side Driving Detection using YOLOv4
Driver Program for our project
Done By,
Sriram N C
Srinandan KS
Jyoti Shetty
R.V. College of Engineering, Bangalore
"""

import cProfile
import datetime
import os
import shutil
import sys
import shelve
import time

import cv2
import numpy as np
import argparse
import youtube_dl
import utils.firebasecode as firebasecode
from utils.annotate_img import annotate_img
from utils.centroidtracker import CentroidTracker
from utils.constants import *
from darknet import predict_frame


class LogicalException(Exception):
    """
    Custom exception class for various logical errors in our program
    """
    pass


class YoutubeDlError(Exception):
    """
    Custom exception thrown for all youtube-dl related download errors.
    Could mean that the stream requested doesn't exist, or youtube-dl
    itself is down due to copyright disputes
    """
    pass


def get_longest_increasing_subsequence(arr: list) -> list:
    """
    Used to return the longest increasing subsequence as a lit
    :param arr: input array
    :return: longest increasing subsequence in the array
    """
    # L[i] - The longest increasing sub-sequence
    # ends with arr[i]
    n = len(arr)
    l = [[] for _ in range(n)]

    l[0].append(arr[0])

    for i in range(1, n):
        for j in range(i):

            # L[i] = {Max(L[j])} + arr[i]
            # where j < i and arr[j] < arr[i]
            if arr[i] > arr[j] and (len(l[i]) < len(l[j]) + 1):
                l[i] = l[j].copy()

        l[i].append(arr[i])

    # L[i] now stores increasing sub-sequence of
    # arr[0..i] that ends with arr[i]
    res = l[0]

    # LIS will be max of all increasing sub-
    # sequences of arr
    for x in l:
        if len(x) > len(res):
            res = x

    return res


def longest_increasing_subsequence_length(arr: list) -> int:
    """
    Returns length of longest increasing subsequence- separate function is used to reduce time complexity
    :param arr: input array
    :return: length of LIS as an integer
    """
    n = len(arr)
    # Creating the sorted list
    b = sorted(list(set(arr)))
    m = len(b)

    # Creating dp table for storing the answers of sub problems
    dp = [[-1 for _ in range(m + 1)] for _ in range(n + 1)]

    # Finding Longest common Subsequence of the two arrays
    for i in range(n + 1):

        for j in range(m + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif arr[i - 1] == b[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def median(arr: list) -> float:
    """
    Finds median of an array
    :param arr: input array
    :return: median
    """
    return sorted(arr)[int(len(arr) / 2)]


def average(arr: list) -> float:
    """
    Finds mean or average of an array
    :param arr: input array
    :return: average
    """
    return sum(arr) / len(arr)


def eightieth_percentile(arr: list, rev: bool) -> float:
    """
    Returns the element present at the 80th percentile of the list; i.e top 20% element or bottom 20% element
    This is to avoid having to use potentially anomalous min or max values in the divider calculation code
    :param arr: input array
    :param rev: bool variable to check whether to reverse the array before computing the 80th percentile value
    :return: 80th percentile value of the list
    """
    return sorted(arr, reverse=rev)[int(len(arr) * 0.8)]


def segregate_by_id(frames: list) -> dict:
    """
    Segregate the movements of each vehicle detected by their id, assigned by the centroid tracking algorithm
    :param frames: list of dicts, each dict represents an object id- centroid  pairing
    :return: dictionary whose keys are vehicle id and values are lists are their corresponding centroid coordinates
    """
    centroid_positions = {}
    for frame in frames:
        for vehicle_id in frame.keys():
            if vehicle_id not in centroid_positions:
                centroid_positions[vehicle_id] = [frame[vehicle_id]]

            else:
                centroid_positions[vehicle_id].append(frame[vehicle_id])

    return centroid_positions


def calculate_line(centroid_positions: dict, width: float, height: float) -> tuple:
    """
    Function to compute the imaginary median of the roadway, based on movements of objects
    :param centroid_positions: dictionary returned by the segregate_by_id function
    :param width: Width of the video frame (unused parameter, kept for future enhancements)
    :param height: Height of the video frame
    :return: two reference points for the image annotation module to construct an imaginary divider using OpenCV methods
    """

    # Assume left is negative y and right is positive y
    left_side_x = []
    right_side_x = []

    for centroid_position_list in centroid_positions.values():
        # each iteration represents one vehicle (since the keys of the input dict represent vehicle id)
        x_values = []
        y_values = []
        for centroid in centroid_position_list:  # represents the centroid positions of one id number
            x_values.append(centroid[0])
            y_values.append(centroid[1])  # isolate x and y direction movements of a vehicle

        l1 = longest_increasing_subsequence_length(y_values)
        l2 = longest_increasing_subsequence_length(list(reversed(y_values)))

        if l1 > l2:
            # if centroid's y values are increasing (positive y direction), it is assumed to be on the right side of the road
            right_side_x.extend(x_values)

        else:
            # if centroid's y values are decreasing (negative y direction), it is assumed to be on the left side of the road
            left_side_x.extend(x_values)

    try:
        left_divider_x = eightieth_percentile(left_side_x, False)
        """
        Based on left_side_x values, we calculate a divider for the vehicles on the left side of the road,
        ignoring vehicles on the right side. Obviously this assumes that no vehicles are breaking the law during the configuration phase
        wherein the program computes the divider. For the time being the divider is vertical, x = left_divider_x represents the divider line.
        At least 80% of the vehicles driving in negative y direction are to the left of x = left_divider_x line. Not 100% since we want to
        avoid extreme values affecting divider calculation        
        """
        right_divider_x = eightieth_percentile(right_side_x, True)

    except IndexError:
        raise LogicalException("Not enough cars to detect a divider. Try increasing config_frames for better results")

    divider = (left_divider_x + right_divider_x) / 2
    if divider == 0:
        divider = 1  # make sure divider does not pass through origin

    return (divider, 0), (divider, height + 10)


def get_wrong_direction_ids(centroid_positions: dict, point1: tuple, point2: tuple) -> list:
    """
    Given the divider line and a dict of centroid positions, determine which vehicles are on the wrong side of the road
    :param centroid_positions: dictionary returned by the segregate_by_id function
    :param point1: One point on the divider line computed
    :param point2: Another point on the divider line
    :return: list of vehicle id's travelling on the wrong side
    """
    wrong_direction_ids = []

    for id_key, centroid_position_list in centroid_positions.items():
        x_values = []
        y_values = []
        for centroid in centroid_position_list:  # represents the centroid positions (list of tuples) of one id number
            x_values.append(centroid[0])
            y_values.append(centroid[1])

        l1 = get_longest_increasing_subsequence(y_values)
        l2 = get_longest_increasing_subsequence(list(reversed(y_values)))

        # we now need to find a reference point to check whether the vehicle is to the left or right  of the separator vector
        if len(l1) > len(
                l2):  # positive y direction, downwards, right side- now if point is left of divider then wrong side
            y_val = l1[0]
            idx = y_values.index(y_val)
            x_val = x_values[idx]
            legal_position = "right"

        else:
            y_val = l2[0]
            idx = y_values.index(y_val)
            x_val = x_values[idx]
            legal_position = "left"

        # assume divider vector points upward (-ve y) and left of the vector is moving on the left side of the road
        # upwards (-ve y)

        if point1[1] < point2[1]:
            a = point1
            b = point2

        else:
            a = point2
            b = point1

        # Now left of divider should be origin (barring edge cases)

        a = np.array(a)
        b = np.array(b)
        p = np.array([x_val, y_val])
        if (np.cross(p - a, b - a) < 0 and legal_position == "right") or (
                np.cross(p - a, b - a) > 0 and legal_position == "left"):
            # find cross product of direction and divider line to determine the actual direction of the vehicle
            # compare actual direction to legal position, if there is a mismatch append this vehicle id to the resultant array
            wrong_direction_ids.append(int(id_key))

    return wrong_direction_ids


def in_traffic_jam(object_movements, obj_id, width, height) -> bool:
    """
    A function to check whether a vehicle was penalised because it was stuck in traffic.
    This is to resolve a bug in our program where a vehicle would be falsely flagged as a violation
    in certain traffic jams
    :param object_movements: list of tuples of centroid coordinates
    :param obj_id: Vehicle id of object to be considered (unused)
    :param width: Width of video frame (unused)
    :param height: Height of video frame
    :return: boolean, based on whether the vehicle was stuck in traffic recently
    """

    # Consider only the last TRAFFIC_JAM_CONTEXT frames in determining traffic jam.
    # If less than MIN_HEIGHT_PROPORTION of height was traversed in that time, then traffic jam

    y_values = [centroid[1] for centroid in object_movements][:-TRAFFIC_JAM_CONTEXT]

    val1 = eightieth_percentile(y_values, False)
    val2 = eightieth_percentile(y_values, True)

    if abs(val1 - val2) < MIN_HEIGHT_PROPORTION * height:
        return True

    return False


def main(program_args):
    """
    Main Function of the program
    :param program_args: Command line arguments passed by the program
    :return: No return value
    """
    # todo- ticket/challan generation with LPR, reduce time taken to flag vehicle, handle slanted cams and right sided countries

    if "frames" not in os.listdir():
        os.mkdir("frames")

    for f in os.listdir('frames'):
        if "placeholder" not in f:
            os.remove(os.path.join('frames', f))

    if "violationframes" not in os.listdir():
        os.mkdir("violationframes")

    for f in os.listdir('violationframes'):
        if "placeholder" not in f:
            os.remove(os.path.join('violationframes', f))

    vidfile = program_args.input
    if program_args.input_mode == 'yt':
        try:
            ip_link = program_args.youtube_link
            ydl_opts = {'ignoreerrors': True, 'outtmpl': '%(title)s.%(ext)s', 'format': '135'}
            # make sure your input yt link has at least a 480p stream and in mp4
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([ip_link])
                info_dict = ydl.extract_info(ip_link, download=False)
                video_title = info_dict.get('title', None)
            print("Input video downloaded from YouTube")
            vidfile = video_title + '.mp4'

        except Exception as e:
            print(e)
            raise YoutubeDlError(
                "Program encountered the above error when trying to download your youtube video. Please ensure that your YouTube video is uploaded in at least 480p.")

    vidcap = cv2.VideoCapture(vidfile)
    print(os.getcwd())
    print(vidfile)
    if not vidcap.isOpened():
        print("Error opening video")
    success, image = vidcap.read()
    if not success:
        print("Video file cannot be read! Exiting...")
        sys.exit(-1)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    counter = 0
    print("Video size = {} X {}, FPS = {}".format(width, height, fps))
    print("Tracking vehicles...")

    movement = []
    point1 = ()
    point2 = ()
    wrong_direction = {}
    ct = CentroidTracker(MAX_DISAPPEARED)
    # load our custom trained darknet network
    network, class_names, class_colors, darknet_width, darknet_height = predict_frame.load_darknet_network()
    print("Darknet network loaded successfully")
    start_time = time.time()
    while True:
        success, image = vidcap.read()
        if not success:
            print("Video fully read. Predictions complete.")
            break

        # run darknet predictions on the trained network
        pred_img, detections = predict_frame.predict_frame(image, network, class_names, class_colors, darknet_width,
                                                           darknet_height, compute_output_image=False)

        detections = predict_frame.convert_detections(detections)
        prediction = predict_frame.centroid_calculation(detections)['all_data']

        centroids = []
        if not prediction or prediction == []:
            centroids.append((0, 0))

        for obj in prediction:
            centroids.append(obj['centroid'])

        objects = ct.update(np.array(centroids), len(prediction))
        # update id numbers in our centroid tracking module
        wd_ids = list(objects.keys())
        centroids = list(objects.values())
        centroids = [tuple(x) for x in centroids]

        centroid_dict = {}
        for obj_id, centroid in zip(wd_ids, centroids):
            centroid_dict[obj_id] = centroid
            if obj_id not in wrong_direction.keys():
                wrong_direction[obj_id] = 0

        movement.append(centroid_dict)
        segregated = segregate_by_id(movement)
        if len(movement) == CONFIG_FRAMES:
            print("Configuration stage complete, drawing divider lines now")
            point1, point2 = calculate_line(segregated, width, height)
            if point1 == () or point2 == () or point1[1] == point2[1]:
                raise LogicalException("Configuration did not create a divider")

        if len(movement) > CONFIG_FRAMES and len(movement) % WSD_CHECKING_INTERVAL == 0:
            first_slice = max((len(movement) - 2 * WSD_CHECKING_INTERVAL), CONFIG_FRAMES)  # wrong side analysis window is twice of interval

            first_slice = int(first_slice)

            segregated_recent = segregate_by_id(movement[first_slice:])
            wd_ids = get_wrong_direction_ids(segregated_recent, point1, point2)
            # print("Wrong direction ids are: ", ids)

            for id_key in wrong_direction.keys():
                if id_key not in wd_ids and wrong_direction[id_key] >= 1:
                    # if not wrong direction id in this iteration, reduce by 1 and stop reducing when zero, after that acquit the vehicle
                    wrong_direction[id_key] -= 1
            for obj_id in wd_ids:
                if wrong_direction[obj_id] <= 4:
                    wrong_direction[obj_id] += 1
                    # increase wrong direction count until it reaches 4, so that the vehicle is not immediately acquitted in one frame

                object_movements = segregated[obj_id]  # segregate the movements only for the violating vehicles
                if len(object_movements) < TRAFFIC_JAM_CONTEXT:
                    # wrong_direction[id] = 0
                    pass
                elif in_traffic_jam(object_movements, obj_id, width, height):
                    wrong_direction[obj_id] = 0
                    print("Id number {} is in a traffic jam, manually changed color to green".format(obj_id))

        if counter % WSD_CHECKING_INTERVAL == 0:
            save_violation = True  # to prevent the same violation getting saved repeatedly
        else:
            save_violation = False

        annotate_img(image, wrong_direction, point1, point2,
                     fps=fps, save=True,
                     save_violation=save_violation,
                     show=program_args.show_frames,
                     output_file=os.path.join("frames", "{}.jpg".format(counter)),
                     centroid_dict=centroid_dict,
                     violations_output_file=os.path.join("violationframes", "{}.jpg".format(counter)))

        counter += 1

    end_time = time.time()
    print("Input video took {} seconds to process.".format(int(end_time-start_time)))
    print("Frames saved")

    if program_args.save_output_video:
        os.chdir('frames')
        cmd1 = "ffmpeg -framerate {} -i %d.jpg output.mp4".format(fps)
        os.system(cmd1)
        shutil.move("output.mp4", "../output/output.mp4")
        os.chdir("..")

    if not program_args.use_firebase:
        return

    db, db_id, firebasea, firebaseb, storage, user = firebasecode.authenticate()
    urls = []
    sf1 = shelve.open("urls.sf")

    for f in os.listdir('violationframes'):
        if 'placeholder' in f:
            continue
        fpath = os.path.join('violationframes', f)
        car_id = "Unknown"
        timestamp = datetime.datetime.now().strftime("%H:%M:%S %d-%B-%Y")
        url = firebasecode.add_record(storage, firebaseb, car_id, timestamp, fpath, db_id, user)
        urls.append(url)

    sf1['urls'] = urls
    sf1.close()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Run the Wrong Side detection code")

    parser.add_argument('--input', default='demo_data/thai_cctv.mp4',
                        help="input video file path, default: %(default)s")
    parser.add_argument('--youtube_link', default='https://www.youtube.com/watch?v=ATq6ZbRQtDY',
                        help="input video youtube link, default: %(default)s")
    parser.add_argument('--input_mode', default='yt',
                        help="Mode of input, yt for youtube, fl for file, default: %(default)s")
    parser.add_argument('--profile', default=0,
                        help="Perform Python profiling to analyse bottlenecks, default: %(default)s")
    parser.add_argument('--save_output_video', default=1, help="Save output video file, default: %(default)s")
    parser.add_argument('--show_frames', default=0,
                        help="Show output frames as detection is taking place, default: %(default)s")
    parser.add_argument('--use_firebase', default=0,
                        help="Use your firebase db to store violation images, make sure to create sensitive_data.json in the same directory as this program, default: %(default)s")

    args = parser.parse_args()

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        main(args)
        pr.disable()
        pr.dump_stats('analysis.prof')
        sys.exit(0)

    main(args)
