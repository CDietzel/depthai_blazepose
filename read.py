#!/usr/bin/env python3

import pickle

objects = []

filename = "/home/locobot/Documents/Repos/depthai_blazepose/outputs/smoothed moving"

with open(filename + ".pickle", "rb") as file:
    while True:
        try:
            objects.append(pickle.load(file))
        except EOFError:
            break

print(objects)
