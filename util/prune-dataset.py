#!/usr/bin/env python3

import argparse
import os
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inPlaceDir', type=str,
                        help="Directory to prune in-place.")
    parser.add_argument('--numImagesThreshold', type=int,
                        help="Delete directories with less than this many images.",
                        default=10)
    args = parser.parse_args()

    exts = ["jpg", "png"]

    for subdir, dirs, files in os.walk(args.inPlaceDir):
        if subdir == args.inPlaceDir:
            continue
        nImgs = 0
        for fName in files:
            (imageClass, imageName) = (os.path.basename(subdir), fName)
            if any(imageName.lower().endswith("." + ext) for ext in exts):
                nImgs += 1
        if nImgs < args.numImagesThreshold:
            print("Removing {}".format(subdir))
            shutil.rmtree(subdir)
