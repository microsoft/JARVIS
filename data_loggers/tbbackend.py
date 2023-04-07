#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
""" A TensorBoard backend.

Writes logs to a file using a Google's TensorBoard protobuf format.
"""
import os
import tensorboardX as tbx
import numpy as np


class TBBackend(object):
    def __init__(self, log_dir):
        self.writers = []
        self.log_dir = log_dir
        self.writers.append(tbx.FileWriter(log_dir))

    def scalar_summary(self, tag, scalar, step):
        """From TF documentation:
            tag: name for the data. Used by TensorBoard plugins to organize data.
            value: value associated with the tag (a float).
        """
        summary = tbx.summary.Summary(value=[tbx.summary.Summary.Value(tag=tag, simple_value=scalar)])
        self.writers[0].add_summary(summary, step)

    def list_summary(self, tag, list, step, multi_graphs):
        """Log a relatively small list of scalars.

        We want to track the progress of multiple scalar parameters in a single graph.
        The list provides a single value for each of the parameters we are tracking.
        
        NOTE: There are two ways to log multiple values in TB and neither one is optimal.
        1. Use a single writer: in this case all of the parameters use the same color, and
           distinguishing between them is difficult.
        2. Use multiple writers: in this case each parameter has its own color which helps
           to visually separate the parameters.  However, each writer logs to a different
           file and this creates a lot of files which slow down the TB load.
        """
        for i, scalar in enumerate(list):
            if multi_graphs and (i+1 > len(self.writers)):
                self.writers.append(tbx.FileWriter(os.path.join(self.log_dir, str(i))))
            summary = tbx.summary.Summary(value=[tbx.summary.Summary.Value(tag=tag, simple_value=scalar)])
            self.writers[0 if not multi_graphs else i].add_summary(summary, step)

    def histogram_summary(self, tag, tensor, step):
        """
        From the TF documentation:
        tf.summary.histogram takes an arbitrarily sized and shaped Tensor, and
        compresses it into a histogram data structure consisting of many bins with
        widths and counts.

        TensorFlow uses non-uniformly distributed bins, which is better than using
        numpy's uniform bins for activations and parameters which converge around zero,
        but we don't add that logic here.

        https://www.tensorflow.org/programmers_guide/tensorboard_histograms
        """
        hist, edges = np.histogram(tensor, bins=200)
        tfhist = tbx.summary.HistogramProto(
            min=np.min(tensor),
            max=np.max(tensor),
            num=int(np.prod(tensor.shape)),
            sum=np.sum(tensor),
            sum_squares=np.sum(np.square(tensor)))

        # From the TF documentation:
        #   Parallel arrays encoding the bucket boundaries and the bucket values.
        #   bucket(i) is the count for the bucket i.  The range for a bucket is:
        #    i == 0:  -DBL_MAX .. bucket_limit(0)
        #    i != 0:  bucket_limit(i-1) .. bucket_limit(i)
        tfhist.bucket_limit.extend(edges[1:])
        tfhist.bucket.extend(hist)

        summary = tbx.summary.Summary(value=[tbx.summary.Summary.Value(tag=tag, histo=tfhist)])
        self.writers[0].add_summary(summary, step)

    def sync_to_file(self):
        for writer in self.writers:
            writer.flush()
