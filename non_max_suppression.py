

import numpy as np
import random
from datetime import date

def nms(boxes, scores, classes, overlap=0.45, min_score=.4):
    """ Does Non Max Suppression on the given boxes, scores, and classes.
    Any boxes with score < min_score are ignored. Boxes are chosen based on
    decreasing score.
    When a box is chosen, any overlapping boxes of the same class with IoU > overlap
    are discarded.

    Parameters:
      boxes: Numpy array of shape (k, 4). Each row consists of left, top, right, and bottom.
      scores: NumPy array of shape (k). Each value is in the range [0.0, 1.0]
      classes: NumPy array of shape (k). Each value is an integer in the range 0..100.
     

    Return result:
      a NumPy array with j elements in it, each representing a chosen box.
      Each of the j elements is an index into the original array.
    """
    if not boxes.size:
        empty = np.array([])
        return empty

    # sort scores in descending order
    indices = np.argsort(scores)[::-1]

    left, top, right, bottom = boxes[:,0] , boxes[:,1] , boxes[:,2] , boxes[:,3]

    area = np.multiply((bottom-top),(right - left))
    
    good_indices = []

    # while indices is non-empty and the biggest score is greater than or equal to min_score
    while indices.size and scores[indices[0]]>=min_score:

        first_index = indices[0]    #the index of biggest score we currently have
        
        good_indices.append(first_index)   

        # separate out the same from different class indices
        same_class_indices = indices[np.equal(classes[first_index],classes[indices])]
        dif_class_indices = indices[np.not_equal(classes[first_index],classes[indices])]

        # calculate iou of first_index box with every other box from the same class
        iou = vectorized_iou(left, top, right, bottom, first_index,same_class_indices, area)

        # a boolean array specifying whether or not the corresponding entries have iou > overlap
        bool_filter = np.greater(iou,overlap)   
        
        # delete the entries that had true bool values (keep the ones that had false)
        same_class_indices = same_class_indices[1:]
        same_class_indices = same_class_indices[np.logical_not(bool_filter)]  

        # remove highest score used and merge filtered same class indices with (unchanged) class indices
        indices = indices[1:]                   
        indices = np.append(same_class_indices,dif_class_indices)
        
    return np.asarray(good_indices)

def vectorized_iou(left, top, right, bottom, first_index, indices, area):
    """ vectorized implementation of iou
        Parameters:
            left:  an array of all the left sides of all the boxes
            top:  an array of all the top sides of all the boxes
            bottom:  an array of all the bottom sides of all the boxes
            first_index: the index of the box to which the iou of every other box will be calculated
            indices: the list of remaining indices
            area: the vectorized area of all the boxes
        Return:
            an array of the area of the first_index with every other box
    """
    
    inter_left = np.maximum(left[first_index],left[indices[1:]])
    inter_top = np.maximum(top[first_index], top[indices[1:]])

    inter_right = np.minimum(right[first_index], right[indices[1:]])
    inter_bottom = np.minimum(bottom[first_index],bottom[indices[1:]])
    
    inter_width = np.maximum(0, inter_right - inter_left)
    inter_height = np.maximum(0, inter_bottom - inter_top)
    inter_area = np.multiply(inter_width, inter_height)

    union_area = (area[first_index] + area[indices[1:]]) - inter_area
    overlap = np.divide(inter_area, union_area)

    return overlap
