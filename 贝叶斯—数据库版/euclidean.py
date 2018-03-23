#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.metrics.pairwise import euclidean_distances
import numpy
def ComputerNearestNeighbor(line_xl,pipei_xl):
    distance=[]
    for i in range(len(pipei_xl)):
        dist = euclidean_distances(line_xl, pipei_xl[i])[0][0]
        distance.append((dist,i))
    distance.sort()
    return distance