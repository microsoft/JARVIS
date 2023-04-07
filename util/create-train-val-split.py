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

import argparse
import errno
import os
import random
import shutil


def mkdirP(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def getImgs(imageDir):
    exts = ["jpg", "png"]

    # All images with one image from each class put into the validation set.
    allImgsM = []
    classes = set()
    valImgs = []
    for subdir, dirs, files in os.walk(imageDir):
        for fName in files:
            (imageClass, imageName) = (os.path.basename(subdir), fName)
            if any(imageName.lower().endswith("." + ext) for ext in exts):
                if imageClass not in classes:
                    classes.add(imageClass)
                    valImgs.append((imageClass, imageName))
                else:
                    allImgsM.append((imageClass, imageName))
    print("+ Number of Classes: '{}'.".format(len(classes)))
    return (allImgsM, valImgs)


def createTrainValSplit(imageDir, valRatio):
    print("+ Val ratio: '{}'.".format(valRatio))

    (allImgsM, valImgs) = getImgs(imageDir)

    trainValIdx = int((len(allImgsM) + len(valImgs)) * valRatio) - len(valImgs)
    assert(trainValIdx > 0)  # Otherwise, valRatio is too small.

    random.shuffle(allImgsM)
    valImgs += allImgsM[0:trainValIdx]
    trainImgs = allImgsM[trainValIdx:]

    print("+ Training set size: '{}'.".format(len(trainImgs)))
    print("+ Validation set size: '{}'.".format(len(valImgs)))

    for person, img in trainImgs:
        origPath = os.path.join(imageDir, person, img)
        newDir = os.path.join(imageDir, 'train', person)
        newPath = os.path.join(imageDir, 'train', person, img)
        mkdirP(newDir)
        shutil.move(origPath, newPath)

    for person, img in valImgs:
        origPath = os.path.join(imageDir, person, img)
        newDir = os.path.join(imageDir, 'val', person)
        newPath = os.path.join(imageDir, 'val', person, img)
        mkdirP(newDir)
        shutil.move(origPath, newPath)

    for person, img in valImgs:
        d = os.path.join(imageDir, person)
        if os.path.isdir(d):
            os.rmdir(d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'imageDir', type=str, help="Directory of images to partition in-place to 'train' and 'val' directories.")
    parser.add_argument('--valRatio', type=float, default=0.10,
                        help="Validation to training ratio.")
    args = parser.parse_args()

    createTrainValSplit(args.imageDir, args.valRatio)
