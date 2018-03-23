#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy
def moreCos(vec1,vec2):
    num = float(numpy.sum(vec1 * vec2))
    denom = numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2)
    cos = num / denom
    return 1-cos
def ComputerNearestNeighbor(line_xl,pipei_xl):
    distance=[]
    for i in range(len(pipei_xl)):
        dist = moreCos(numpy.array(line_xl), numpy.array(pipei_xl[i]))
        distance.append((dist,i))
    distance.sort()
    return distance