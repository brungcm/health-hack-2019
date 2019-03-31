# -*- coding: utf-8 -*-
"""
.. module:: engine
   :platform: Unix
   :synopsis: Definition of trackable objects and base object detection estimator class

.. moduleauthor:: Rodrigo Pereira <rodrigofp@ciandt.com>

"""

import abc


class TrackingObject(object):
    """ Defines a trackable object.

        In details, it holds a bounding box, life and tracker.

        The life of an object is defined by the number of cycles (detect & track) 
        in which the object should be kept for tracking when not detected.
        This is necessary once detection may fail.
        The one who uses this class is responsable to deallocate dead objects.

        A tracker is any method to keep track of an object throught a serie of frames.
        One may use `DLIB <http://dlib.net/correlation_tracker.py.html>`_, `OpenCV <https://docs.opencv.org/3.4.1/d9/df8/group__tracking.html>`_ or any other lib

    """

    def __init__(self, bbox_xmin=-1, bbox_ymin=-1, bbox_xmax=-1, bbox_ymax=-1,
                 life=-1, tracker=None, start_time=None,
                 obj_id=None, obj_type_id=None, obj_type_name=None, obj_score=None):
        """ Constructor

            Args:
                ``bbox_xmin`` (int): bbox xmin coordinate
                ``bbox_ymin`` (int): bbox ymin coordinate
                ``bbox_xmax`` (int): bbox xmax coordinate
                ``bbox_ymax`` (int): bbox ymax coordinate
                ``life`` (int): life of the object
                ``tracker`` (any): object tracker
                ``obj_id`` (int): object id
                ``obj_type_id`` (int): object type id
                ``obj_type_name`` (str): object type name
                ``obj_score`` (float): object confidence, if any, given by an object detector
        """

        super(TrackingObject, self).__init__()
        self.bbox_xmin = bbox_xmin
        self.bbox_ymin = bbox_ymin
        self.bbox_xmax = bbox_xmax
        self.bbox_ymax = bbox_ymax
        self.life = life
        self.tracker = tracker
        self.obj_id = obj_id
        self.obj_type_id = obj_type_id
        self.obj_type_name = obj_type_name
        self.obj_score = obj_score
        self.start_time = start_time

    @property
    def centroid(self):
        """ Returns the centroid coordinates (x,y) of the object
        """
        return ((self.bbox_xmax + self.bbox_xmin) / 2, (self.bbox_ymax + self.bbox_ymin) / 2)

    @property
    def bbox(self):
        """ Returns the bbox (xmin,ymin,xmax,ymax) of the object
        """
        return (self.bbox_xmin, self.bbox_ymin, self.bbox_xmax, self.bbox_ymax)

    def update_bbox(self, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax):
        """ Update bbox coordinates

            Args:
                ``bbox_xmin`` (int): bbox xmin coordinate
                ``bbox_ymin`` (int): bbox ymin coordinate
                ``bbox_xmax`` (int): bbox xmax coordinate
                ``bbox_ymax`` (int): bbox ymax coordinate
        """
        self.bbox_xmin = bbox_xmin
        self.bbox_ymin = bbox_ymin
        self.bbox_xmax = bbox_xmax
        self.bbox_ymax = bbox_ymax

    def __repr__(self):
        s = '************************\n'\
            'obj_id = {} \nobj_type_id = {} \nobj_type_name = {}\n'\
            'bbox(xmin,xmax,ymin,ymax) = ({},{},{},{})'\
            '\n************************'.format(
                self.obj_id, self.obj_type_id, self.obj_type_name,
                self.bbox_xmin, self.bbox_xmax, self.bbox_ymin, self.bbox_ymax)
        return s


class BaseObjectEstimator(object):
    """ Base class for object location estimation.            
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kw):
        """ Constructor
        """
        super(BaseObjectEstimator, self).__init__()

    def initialize(self):
        """ Perform any initialization
        """
        raise NotImplementedError("Not implemented yet.")

    def process(self, frame):
        """ Process frame to update bboxes coordinates
        """
        raise NotImplementedError("Not implemented yet.")

    def get_objects(self):
        """ Perform any preprocessing and return trackable objects
        """
        raise NotImplementedError("Not implemented yet.")
