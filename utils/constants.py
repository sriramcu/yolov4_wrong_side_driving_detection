"""
Wrong Side Driving Detection using YOLOv4
Module where commonly used constants for our main program are defined
Done By,
Sriram N C
Srinandan KS
Jyoti Shetty
R.V. College of Engineering, Bangalore
"""


INFINITY = int(10 ** 8)
CONFIG_FRAMES = 500  # number of frames the model observes the footage to construct a divider, without penalising violations in the meantime
WSD_CHECKING_INTERVAL = 100  # check wrong direction periodically every few frames, 100 frames interval is roughly 3 seconds
CV2_OFFSET = 30  # milliseconds to deduct for processing
MAX_HEIGHT = 230  # remove bounding boxes of huge dimensions
MAX_WIDTH = 230  # remove bounding boxes of huge dimensions
MIN_DIST = 75    # remove clustered (i.e too close to each other) centroids or boxes
TRAFFIC_JAM_CONTEXT = 100   # number of frames to consider while evaluating whether a vehicle is stuck in a traffic jam
MIN_HEIGHT_PROPORTION = 0.03  # if the vehicle travels less than this portion of the video frame height in traffic_jam_context, it is considered to be stuck in traffic
MAX_DISAPPEARED = 30   # number of frames after which an ID number for a centroid is discarded, i.e the particular object has left the frame


def near_infinity(num):
    """
    Function to check whether a given number is close to the infinity value defined above
    :param num: number to be checked
    :return: bool, true if object is near infinity, false otherwise
    """
    if num >= (0.9 * INFINITY):
        return True

    return False
