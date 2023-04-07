#!/usr/bin/env python2
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


# Example usage: ./util/annotate-image.py /data/path_to_your_image.jpg outerEyesAndNose

import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

import argparse
import cv2

from openface.align_dlib import AlignDlib

modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def main(args):
    align = AlignDlib(args.dlibFacePredictor)

    bgrImg = cv2.imread(args.img)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(args.img))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(args.img))

    landmarks = align.findLandmarks(rgbImg, bb)
    if landmarks is None:
        raise Exception("Unable to find landmarks within image: {}".format(args.img))

    bl = (bb.left(), bb.bottom())
    tr = (bb.right(), bb.top())
    cv2.rectangle(bgrImg, bl, tr, color=(153, 255, 204), thickness=3)
    for landmark in landmarks:
        cv2.circle(bgrImg, center=landmark, radius=3, color=(102, 204, 255), thickness=-1)
    print("Saving image to 'annotated.png'")
    cv2.imwrite("annotated.png", bgrImg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('img', type=str, help="Input image.")
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('landmarks', type=str,
                        choices=['outerEyesAndNose', 'innerEyesAndBottomLip'],
                        help='The landmarks to align to.')
    parser.add_argument('--size', type=int, help="Default image size.",
                        default=96)
    args = parser.parse_args()

    main(args)
