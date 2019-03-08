"""

Functions for computations related to geometry of the page

This code was written by Kyle Lo.

https://github.com/allenai/corvid/blob/master/corvid/util/geom.py
"""

from typing import List, Dict
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])


class Box(object):
    def __init__(self, llx: float, lly: float, urx: float, ury: float):
        self.ll = Point(x=llx, y=lly)
        self.ur = Point(x=urx, y=ury)

    @property
    def height(self) -> float:
        return self.ur.y - self.ll.y

    @property
    def width(self) -> float:
        return self.ur.x - self.ll.x

    def __repr__(self):
        return 'x: [{},{}]\ny: [{},{}]'.format(self.ll.x, self.ur.x,
                                               self.ur.y, self.ll.y)

    def __str__(self):
        return 'x: [{},{}]\ny: [{},{}]'.format(self.ll.x, self.ur.x,
                                               self.ur.y, self.ll.y)

    @classmethod
    def from_json(cls, json: Dict) -> 'Box':
        box = Box(llx=json['llx'],
                  lly=json['lly'],
                  urx=json['urx'],
                  ury=json['ury'])
        return box

    def to_json(self) -> Dict:
        json = {
            'llx': self.ll.x,
            'lly': self.ll.y,
            'urx': self.ur.x,
            'ury': self.ur.y
        }
        return json

    @classmethod
    def is_x_overlap(cls, box1: 'Box', box2: 'Box') -> float:
        is_left_x_within_box2 = box2.ll.x < box1.ll.x < box2.ur.x
        is_right_x_within_box2 = box2.ll.x < box1.ur.x < box2.ur.x
        is_contains_box2 = box1.ll.x < box2.ll.x and box1.ur.x > box2.ur.x
        return is_left_x_within_box2 or is_right_x_within_box2 or is_contains_box2

    @classmethod
    def is_y_overlap(cls, box1: 'Box', box2: 'Box') -> float:
        is_left_y_within_box2 = box2.ll.y < box1.ll.y < box2.ur.y
        is_right_y_within_box2 = box2.ll.y < box1.ur.y < box2.ur.y
        is_contains_box2 = box1.ll.y < box2.ll.y and box1.ur.y > box2.ur.y
        return is_left_y_within_box2 or is_right_y_within_box2 or is_contains_box2

    @classmethod
    def is_above(cls, above_box: 'Box', below_box: 'Box') -> bool:
        if Box.is_y_overlap(above_box, below_box):
            return False
        elif not Box.is_x_overlap(above_box, below_box):
            return False
        else:
            return above_box.ll.y > below_box.ur.y

    @classmethod
    def min_x_dist(cls, box1: 'Box', box2: 'Box') -> float:
        assert not Box.is_x_overlap(box1, box2)
        return min(abs(box1.ll.x - box2.ur.x), abs(box1.ur.x - box2.ll.x))

    @classmethod
    def min_y_dist(cls, box1: 'Box', box2: 'Box') -> float:
        assert not Box.is_y_overlap(box1, box2)
        return min(abs(box1.ll.y - box2.ur.y), abs(box1.ur.y - box2.ll.y))

    @classmethod
    def compute_bounding_box(cls, boxes: List['Box']) -> 'Box':
        """Finds the bounding box that tightly bounds all provided boxes"""

        max_lower_left_x, max_lower_left_y = float('inf'), float('inf')
        max_upper_right_x, max_upper_right_y = -float('inf'), -float('inf')

        for box in boxes:
            if box.ll.x < max_lower_left_x:
                max_lower_left_x = box.ll.x
            if box.ll.y < max_lower_left_y:
                max_lower_left_y = box.ll.y
            if box.ur.x > max_upper_right_x:
                max_upper_right_x = box.ur.x
            if box.ur.y > max_upper_right_y:
                max_upper_right_y = box.ur.y

        return Box(llx=max_lower_left_x, lly=max_lower_left_y,
                   urx=max_upper_right_x, ury=max_upper_right_y)
