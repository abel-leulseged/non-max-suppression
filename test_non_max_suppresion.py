# Assignment 4
# CS 152
# Fall, 2018
# test_non_max_suppression.py

# Abel Leulseged, Dhruv Sawhney

from collections import Counter
from datetime import date
from non_max_suppression import nms
import numpy as np
import random
import unittest


class TestNonMaxSuppression_StudentTests(unittest.TestCase):

    def assertEqualAnyOrder(self, a, b):
        """ Asserts that a and b contain the same values, in any order.
        Assumes that a and b are one-dimensional NumPy arrays. """
        a.sort()
        b.sort()
        self.assertEqual(list(a), list(b))

    # input: 3 boxes with {scores below min_threshold (0.4 default)}
    # output: empty list
    def test_below_min_score(self):
        boxes = [[1,2,3,4],[2,1,3,4],[1,3,4,5]]
        boxes = np.asarray(boxes)
        scores = [0.3, 0.2, 0.1]
        scores = np.asarray(scores)
        classes = [0,1,2]
        classes = np.asarray(classes)
        self.assertEqualAnyOrder(nms(boxes, scores, classes),np.asarray([]))


    # input: 2 boxes {no overlap, same class, above threshold}
    # output: 2 boxes
    def test_same_class_no_overlap(self):
        boxes = [[10,10,20,20],[30,10,40,20]]
        boxes = np.asarray(boxes)
        scores = [0.9, 0.9]
        scores = np.asarray(scores)
        classes = [1,1]
        classes = np.asarray(classes)
        self.assertEqualAnyOrder(nms(boxes, scores, classes),np.asarray([0,1]))

    # input: 2 boxes {overlap, same class, above threshold}
    # output: 1 box
    def test_same_class_multiple_overlap(self):
        boxes = [[1, 5, 4, 10], [1, 5, 4, 10]]
        boxes = np.asarray(boxes)
        scores = [0.9, 0.8]
        scores = np.asarray(scores)        
        classes = [2, 2]
        classes = np.array(classes)
        self.assertEqualAnyOrder(nms(boxes, scores, classes),np.asarray([0]))    

    # input: 3 boxes with {diff class, same score, above threshold}
    # output: 3 boxes
    def test_multiple_class_no_overlap(self):
        boxes = [[1,2,3,4],[2,1,3,4],[5,10,7,11]]
        boxes = np.asarray(boxes)
        scores = [0.9, 0.8, 0.7]
        scores = np.asarray(scores)
        classes = [0,1,2]
        classes = np.asarray(classes)
        self.assertEqualAnyOrder(nms(boxes, scores, classes),np.asarray([0,1,2]))


class TestNonMaxSuppression(unittest.TestCase):

    def assertEqualAnyOrder(self, a, b):
        """ Asserts that a and b contain the same values, in any order.
        Assumes that a and b are one-dimensional NumPy arrays. """
        a.sort()
        b.sort()
        self.assertEqual(list(a), list(b))
       

    def test_no_boxes_returns_empty(self):
        empty = np.array([])
        self.assertEqual(0, nms(empty, empty, empty).size)


    def test_random_overlapping(self):
        """Tests by creating a random set of boxes: one true one per class
        and a number of random ones.

        Don't start using this test until you've got all other,
        more basic tests working."""

        def random_in_range(a_min, a_max):
            return a_min + random.random() * (a_max - a_min)

        # Use today's date as a seed, so that we are reproducible all day
        # today (while debugging:)
        random.seed(str(date.today()))
        MIN_COORD = 0
        MAX_COORD = 1000

        for run in range(20):
            boxes = []
            scores = []
            classes = []
            good_indexes = []
            for c in range(99):
                # For each class, create one real box, and a number
                # of boxes with lower scores that overlap the real box
                left = random.randrange(MAX_COORD-1)
                top = random.randrange(MAX_COORD-1)
                right = random.randrange(left+1, MAX_COORD)
                bottom = random.randrange(top+1, MAX_COORD)
                real_box = [left, top, right, bottom]
                real_score = random_in_range(.3, .9)
                boxes.append(real_box)
                scores.append(real_score)
                classes.append(c)
                if real_score > 0.4:
                    good_indexes.append(len(boxes) - 1)
                for fake_boxes in range(20):
                    h_factor = random_in_range(-.2, .2)
                    v_factor = random_in_range(-.2, .2)
                    delta_h = (right - left) * h_factor
                    delta_v = (bottom - top) * v_factor
                    fake_box = [
                            left + delta_h, top + delta_v,
                            right + delta_h, bottom + delta_v]
                    boxes.append(fake_box)
                    fake_score = random_in_range(real_score - .2, real_score - .001)
                    scores.append(fake_score)
                    classes.append(c)

                result = nms(np.array(boxes), np.array(scores), np.array(classes))
                expected = np.array(good_indexes)

                self.assertEqualAnyOrder(np.array(good_indexes),
                    nms(np.array(boxes), np.array(scores), np.array(classes)))

if __name__ == '__main__':
    unittest.main()
