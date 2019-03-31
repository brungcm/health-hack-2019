# -*- coding: utf-8 -*-

import abc


class BaseVideoStream(object):
    """ Inherit from this class to build your video stream provider
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """ Constructor
        """
        super(BaseVideoStream,self).__init__()

    def initialize(self):
        """ Perform any initialization
        """
        raise NotImplementedError('Not implemented yet.')

    def next_frame(self):
        """ Returns next frame
        """
        raise NotImplementedError('Not implemented yet.')
