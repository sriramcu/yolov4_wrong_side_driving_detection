"""
Wrong Side Driving Detection using YOLOv4
Centroid tracking module for our project
Done By,
Sriram N C
Srinandan KS
Jyoti Shetty
R.V. College of Engineering, Bangalore
"""

from scipy.spatial import distance as dist
from collections import OrderedDict
from .constants import *
import numpy as np


class CentroidTracker:
    """
    Class defined to track the movements of centroids of objects detected
    """
    def __init__(self, max_disappeared=MAX_DISAPPEARED):
        """
        initialize the next unique object ID along with two ordered
        dictionaries used to keep track of mapping a given object
        ID to its centroid and number of consecutive frames it has
        been marked as "disappeared", respectively
        :param max_disappeared: maximum consecutive frames a given
        object is allowed to be marked as "disappeared" until we
        need to unregister the object from tracking
        """

        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = max_disappeared

    def register(self, centroid):
        """
        Register an object with an ID
        :param centroid: centroid coordinates
        :return: None
        """
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def unregister(self, object_id):
        """
        Unregister, i.e. remove an id number mapped to an object
        :param object_id: ID of object to be unregistered
        :return: None
        """
        # to unregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids, my_len):
        """
        Update centroid coordinates of all objects, use Euclidean distance to determine the ID number corresponding to a movement,
        i.e. the closest centroid to the previous frame's centroids retains the object ID number for that centroid
        :param input_centroids: all detected objects' centroids
        :param my_len: length of list of input bounding box rectangles
        :return: set of trackable objects
        """
        # check to see if the list of input bounding box rectangles
        # is empty
        if my_len == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, unregister it
                if self.disappeared[object_id] > self.maxDisappeared:
                    self.unregister(object_id)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        # inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # # loop over the bounding box rectangles
        # for (i, (startX, startY, endX, endY)) in enumerate(rects):
        # # use the bounding box coordinates to derive the centroid
        # cX = int((startX + endX) / 2.0)
        # cY = int((startY + endY) / 2.0)
        # inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            distances = dist.cdist(np.array(object_centroids), input_centroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = distances.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = distances.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or unregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            used_rows = set()
            used_cols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in used_rows or col in used_cols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                used_rows.add(row)
                used_cols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unused_rows = set(range(0, distances.shape[0])).difference(used_rows)
            unused_cols = set(range(0, distances.shape[1])).difference(used_cols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if distances.shape[0] >= distances.shape[1]:
                # loop over the unused row indexes
                for row in unused_rows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants unregistering the object
                    if self.disappeared[object_id] > self.maxDisappeared:
                        self.unregister(object_id)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        # return the set of trackable objects
        return self.objects
