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

import tensorflow as tf

import i3d

_IMAGE_SIZE = 224
_NUM_CLASSES = 400


class I3dTest(tf.test.TestCase):
    """Test of Inception I3D model, without real data."""

    def testModelShapesWithSqueeze(self):
        """
        Test shapes after running some fake data through the model.
        """

        i3d_model = i3d.InceptionI3d(
            num_classes=_NUM_CLASSES,
            final_endpoint="Predictions",
            is_training=True,
            dropout_keep_prob=0.5,
        )

        # Create a dummy input tensor
        inp = tf.zeros([5, 64, _IMAGE_SIZE, _IMAGE_SIZE, 3], dtype=tf.float32)

        # Forward pass
        predictions, end_points = i3d_model(inp)

        # Assert output shapes
        self.assertEqual(predictions.shape, (5, _NUM_CLASSES))
        self.assertEqual(end_points["Logits"].shape, (5, _NUM_CLASSES))

    def testModelShapesWithoutSqueeze(self):
        """
        Test that turning off `spatial_squeeze` changes the output shape.

        Also try setting different values for `dropout_keep_prob`.
        """
        i3d_model = i3d.InceptionI3d(
            num_classes=_NUM_CLASSES,
            spatial_squeeze=False,
            final_endpoint="Predictions",
            is_training=False,
            dropout_keep_prob=1.0,
        )

        # Create a dummy input tensor
        inp = tf.zeros([5, 64, _IMAGE_SIZE, _IMAGE_SIZE, 3], dtype=tf.float32)

        # Forward pass
        predictions, end_points = i3d_model(inp)

        # Assert output shapes
        self.assertEqual(predictions.shape, (5, 1, 1, _NUM_CLASSES))
        self.assertEqual(end_points["Logits"].shape, (5, 1, 1, _NUM_CLASSES))

    def testInitErrors(self):
        """
        Test that the model raises errors for invalid arguments.
        """

        # Invalid `final_endpoint` string.
        with self.assertRaises(ValueError):
            _ = i3d.InceptionI3d(num_classes=_NUM_CLASSES, final_endpoint="Conv3d_1a_8x8")

        # Dropout keep probability must be in (0, 1].
        with self.assertRaises(ValueError):
            _ = i3d.InceptionI3d(num_classes=_NUM_CLASSES, dropout_keep_prob=0.0)

        # Height and width dimensions of the input should be _IMAGE_SIZE.
        i3d_model = i3d.InceptionI3d(
            num_classes=_NUM_CLASSES, is_training=False, dropout_keep_prob=0.5
        )
        inp = tf.zeros([5, 64, 10, 10, 3], dtype=tf.float32)
        with self.assertRaises(ValueError):
            _, _ = i3d_model(inp)


if __name__ == "__main__":
    tf.test.main()
