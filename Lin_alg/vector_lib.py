#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:31:16 2017

@author: anne.novesteras
"""

class Vector(object):
    def __init__(self, object):
        self.coordinates = object
        
    def plus(self, v):
        new_coordinates = [x+y for x,y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)
    
    def minus(self, v):
        new_coordinates = [x-y for x,y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)
    
    def times_scalar(self, c):
        new_coordinates = [c*x for x in self.coordinates]
        return Vector(new_coordinates)
        
    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)
    
    def __eq__(self, v):
        return self.coordinates == v.coordinates
        
v = Vector([8.218, -9.341])
w = Vector([-1.129, 2.111])
print v.plus(w)

v = Vector([8.218, -9.341])
w = Vector([-1.129, 2.111])
print v.minus(w)

v = Vector([8.218, -9.341])
print v.times_scalar(2.)

