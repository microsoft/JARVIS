#!/bin/bash
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

cd $(dirname $0)/..

TAG=lfw.nn4.v2
WORK_DIR=2

LATEST_MODEL=$(ls -t training/work/$WORK_DIR/model_* | \
  head -1  | sed 's/.*model_\(.*\)\.t7/\1/')

printf "\n=== TAG: $TAG\n"
printf "=== WORK_DIR: $WORK_DIR\n"
printf "=== Model: $LATEST_MODEL\n\n"

set -x -e -u

./batch-represent/main.lua \
  -outDir evaluation/$TAG.e$LATEST_MODEL \
  -model ./training/work/$WORK_DIR/model_$LATEST_MODEL.t7 \
  -data data/lfw/dlib.affine.sz:96.OuterEyesAndNose \
  -batchSize 100 \
  -cuda

cd evaluation
./lfw.py --workDir $TAG.e$LATEST_MODEL

tail $TAG.*/accuracies.txt -n 1
