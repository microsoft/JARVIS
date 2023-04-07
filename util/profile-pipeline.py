#!/usr/bin/env python2
# Compute the overall processing latency for an image.
# Brandon Amos
# 2016-01-19
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import os

import numpy as np
np.set_printoptions(precision=2)

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('img', type=str, help="Input image.")
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--numIters', type=int,
                    help="Number of iterations.", default=100)

args = parser.parse_args()

print("Argument parsing and loading libraries took {:0.4f} seconds.".format(
    time.time() - start))

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
print("Loading the dlib and OpenFace models took {:0.4f} seconds.".format(
    time.time() - start))


def getTimes(rgbImg):
    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face.")

    detectionTime = time.time() - start

    start = time.time()
    alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    alignmentTime = time.time() - start

    start = time.time()
    net.forward(alignedFace)
    repTime = time.time() - start
    return (detectionTime, alignmentTime, repTime)

bgrImg = cv2.imread(args.img)
if bgrImg is None:
    raise Exception("Unable to load image: {:0.4f}".format(args.img))
rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
print("Image size: {}".format(rgbImg.shape))

detectionTimes = []
alignmentTimes = []
repTimes = []
totalTimes = []
for i in range(args.numIters):
    (dTime, aTime, repTime) = getTimes(rgbImg)
    detectionTimes.append(dTime)
    alignmentTimes.append(aTime)
    repTimes.append(repTime)
    totalTimes.append(dTime + aTime + repTime)

print('Number of iterations: {}'.format(args.numIters))
avg = np.mean(detectionTimes)
std = np.std(detectionTimes)
print('Average Detection Time (seconds): {:0.4f} +/- {:0.4f}'.format(avg, std))
avg = np.mean(alignmentTimes)
std = np.std(alignmentTimes)
print('Average Alignment Time (seconds): {:0.4f} +/- {:0.4f}'.format(avg, std))
avg = np.mean(repTimes)
std = np.std(repTimes)
print('Average Neural Net Representation Time (seconds): {:0.4f} +/- {:0.4f}'.format(avg, std))
avg = np.mean(totalTimes)
std = np.std(totalTimes)
print('Average Total Time (seconds): {:0.4f} +/- {:0.4f}'.format(avg, std))
