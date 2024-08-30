# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for I3D model code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import i3d

_IMAGE_SIZE = 224
_NUM_CLASSES = 400


class I3dTest(tf.test.TestCase):
  """Test of Inception I3D model, without real data."""

  def testModelShapesWithSqueeze(self):
    """Test shapes after running some fake data through the model."""
    i3d_model = i3d.InceptionI3d(
        num_classes=_NUM_CLASSES, final_endpoint='Predictions')
    inp = tf.placeholder(tf.float32, [None, 64, _IMAGE_SIZE, _IMAGE_SIZE, 3])
    predictions, end_points = i3d_model(
        inp, is_training=True, dropout_keep_prob=0.5)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init_op)
      sample_input = np.zeros((5, 64, _IMAGE_SIZE, _IMAGE_SIZE, 3))
      out_predictions, out_logits = sess.run(
          [predictions, end_points['Logits']], {inp: sample_input})
      self.assertEqual(out_predictions.shape, (5, _NUM_CLASSES))
      self.assertEqual(out_logits.shape, (5, _NUM_CLASSES))

  def testModelShapesWithoutSqueeze(self):
    """Test that turning off `spatial_squeeze` changes the output shape.

    Also try setting different values for `dropout_keep_prob` and snt.BatchNorm
    `is_training`.
    """
    i3d_model = i3d.InceptionI3d(
        num_classes=_NUM_CLASSES, spatial_squeeze=False,
        final_endpoint='Predictions')
    inp = tf.placeholder(tf.float32, [None, 64, _IMAGE_SIZE, _IMAGE_SIZE, 3])
    predictions, end_points = i3d_model(
        inp, is_training=False, dropout_keep_prob=1.0)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init_op)
      sample_input = np.zeros((5, 64, _IMAGE_SIZE, _IMAGE_SIZE, 3))
      out_predictions, out_logits = sess.run(
          [predictions, end_points['Logits']], {inp: sample_input})
      self.assertEqual(out_predictions.shape, (5, 1, 1, _NUM_CLASSES))
      self.assertEqual(out_logits.shape, (5, 1, 1, _NUM_CLASSES))

  def testInitErrors(self):
    # Invalid `final_endpoint` string.
    with self.assertRaises(ValueError):
      _ = i3d.InceptionI3d(
          num_classes=_NUM_CLASSES, final_endpoint='Conv3d_1a_8x8')

    # Dropout keep probability must be in (0, 1].
    i3d_model = i3d.InceptionI3d(num_classes=_NUM_CLASSES)
    inp = tf.placeholder(tf.float32, [None, 64, _IMAGE_SIZE, _IMAGE_SIZE, 3])
    with self.assertRaises(ValueError):
      _, _ = i3d_model(inp, is_training=False, dropout_keep_prob=0)

    # Height and width dimensions of the input should be _IMAGE_SIZE.
    i3d_model = i3d.InceptionI3d(num_classes=_NUM_CLASSES)
    inp = tf.placeholder(tf.float32, [None, 64, 10, 10, 3])
    with self.assertRaises(ValueError):
      _, _ = i3d_model(inp, is_training=False, dropout_keep_prob=0.5)


if __name__ == '__main__':
  tf.test.main()
