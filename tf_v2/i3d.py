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
"""Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.

The model is introduced in:

  Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
  Joao Carreira, Andrew Zisserman
  https://arxiv.org/pdf/1705.07750v1.pdf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf


class Unit3D(snt.Module):
    """Basic unit containing Conv3D + BatchNorm + non-linearity."""

    def __init__(
        self,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        activation_fn=tf.nn.relu,
        use_batch_norm=True,
        use_bias=False,
        is_training=False,
        name="unit_3d",
    ):
        """
        Initializes Unit3D module.

        Args:
            output_channels: number of output channels (int).
            kernel_shape: shape of the convolutional kernel (iterable of 3 ints).
            stride: shape of the convolutional stride (iterable of 3 ints).
            activation_fn: activation function (callable).
            use_batch_norm: whether to use batch normalization (boolean).
            use_bias: whether to use bias (boolean).
            is_training: whether to use training mode for snt.BatchNorm (boolean).
            name: name of the module (string).
        """
        super(Unit3D, self).__init__(name=name)

        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._is_training = is_training

        # layers

        self.conv3d = snt.Conv3D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=stride,
            padding="SAME",
            with_bias=use_bias,
            name="conv_3d",
        )

        self.batch_norm = snt.BatchNorm(
            create_scale=False,
            create_offset=True,
        )

    def __call__(self, net):
        """
        Connects the module to inputs.

        Args:
          net: Inputs to the Unit3D component.

        Returns:
          Outputs from the module.
        """
        net = self.conv3d(net)

        if self._use_batch_norm:
            net = self.batch_norm(net, is_training=self._is_training, test_local_stats=False)

        if self._activation_fn is not None:
            net = self._activation_fn(net)

        return net


class MixedLayerBranch(snt.Module):
    """
    Used to create the branches of the mixed layers.
    """

    def __init__(
        self,
        name,
        is_training,
        a_output_channels,
        a_kernel_shape,
        a_name,
        b_output_channels,
        b_kernel,
        b_name,
    ):
        """
        Initializes MixedLayerBranch module.

        Args:
            name: name of the module (string).
            is_training: whether to use training mode for snt.BatchNorm (boolean).
            a_output_channels: number of output channels for branch A (int).
            a_kernel_shape: shape of the convolutional kernel for branch A (iterable of 3 ints).
            a_name: name of branch A (string).
            b_output_channels: number of output channels for branch B (int).
            b_kernel: shape of the convolutional kernel for branch B (iterable of 3 ints).
            b_name: name of branch B (string).
        """
        super(MixedLayerBranch, self).__init__(name=name)

        if a_output_channels is not None and a_kernel_shape is not None:
            self.branch_a = Unit3D(
                output_channels=a_output_channels,
                kernel_shape=a_kernel_shape,
                is_training=is_training,
                name=a_name,
            )
        else:
            self.branch_a = lambda x: tf.nn.max_pool3d(
                x,
                ksize=[1, 3, 3, 3, 1],
                strides=[1, 1, 1, 1, 1],
                padding="SAME",
                name="MaxPool3d_0a_3x3",
            )

        if b_output_channels is not None and b_kernel is not None:
            self.branch_b = Unit3D(
                output_channels=b_output_channels,
                kernel_shape=b_kernel,
                is_training=is_training,
                name=b_name,
            )
        else:
            self.branch_b = lambda x: x

    def __call__(self, net):
        """
        Connects the module to inputs.

        Args:
            net: Inputs to the MixedLayerBranch component.

        Returns:
            Outputs from the module.
        """

        return self.branch_b(self.branch_a(net))


class MixedLayer(snt.Module):
    """
    Inception layer.
    """

    def __init__(
        self,
        name,
        is_training,
        branch_0_a_output_channels,
        branch_1_a_output_channels,
        branch_1_b_output_channels,
        branch_2_a_output_channels,
        branch_2_b_output_channels,
        branch_3_b_output_channels,
        branch_0_a_name="Conv3d_0a_1x1",
        branch_0_a_kernel_shape=[1, 1, 1],
        branch_1_a_name="Conv3d_0a_1x1",
        branch_1_a_kernel=[1, 1, 1],
        branch_1_b_name="Conv3d_0b_3x3",
        branch_1_b_kernel=[3, 3, 3],
        branch_2_a_name="Conv3d_0a_1x1",
        branch_2_a_kernel=[1, 1, 1],
        branch_2_b_name="Conv3d_0b_3x3",
        branch_2_b_kernel=[3, 3, 3],
        branch_3_b_name="Conv3d_0b_1x1",
        branch_3_b_kernel=[1, 1, 1],
    ):
        """
        Initializes MixedLayer module.

        Args:
            name: name of the module (string).
            is_training: whether to use training mode for snt.BatchNorm (boolean).
            branch_0_a_output_channels: number of output channels for branch 0A (int).
            branch_1_a_output_channels: number of output channels for branch 1A (int).
            branch_1_b_output_channels: number of output channels for branch 1B (int).
            branch_2_a_output_channels: number of output channels for branch 2A (int).
            branch_2_b_output_channels: number of output channels for branch 2B (int).
            branch_3_b_output_channels: number of output channels for branch 3B (int).
            branch_0_a_name: name of branch 0A (string).
            branch_0_a_kernel_shape: shape of the convolutional kernel for branch 0A (iterable of 3 ints).
            branch_1_a_name: name of branch 1A (string).
            branch_1_a_kernel: shape of the convolutional kernel for branch 1A (iterable of 3 ints).
            branch_1_b_name: name of branch 1B (string).
            branch_1_b_kernel: shape of the convolutional kernel for branch 1B (iterable of 3 ints).
            branch_2_a_name: name of branch 2A (string).
            branch_2_a_kernel: shape of the convolutional kernel for branch 2A (iterable of 3 ints).
            branch_2_b_name: name of branch 2B (string).
            branch_2_b_kernel: shape of the convolutional kernel for branch 2B (iterable of 3 ints).
            branch_3_b_name: name of branch 3B (string).
            branch_3_b_kernel: shape of the convolutional kernel for branch 3B (iterable of 3 ints).
        """
        super(MixedLayer, self).__init__(name=name)

        self.branch_0 = MixedLayerBranch(
            name="Branch_0",
            is_training=is_training,
            a_output_channels=branch_0_a_output_channels,
            a_kernel_shape=branch_0_a_kernel_shape,
            a_name=branch_0_a_name,
            b_output_channels=None,
            b_kernel=None,
            b_name=None,
        )

        self.branch_1 = MixedLayerBranch(
            name="Branch_1",
            is_training=is_training,
            a_output_channels=branch_1_a_output_channels,
            a_kernel_shape=branch_1_a_kernel,
            a_name=branch_1_a_name,
            b_output_channels=branch_1_b_output_channels,
            b_kernel=branch_1_b_kernel,
            b_name=branch_1_b_name,
        )

        self.branch_2 = MixedLayerBranch(
            name="Branch_2",
            is_training=is_training,
            a_output_channels=branch_2_a_output_channels,
            a_kernel_shape=branch_2_a_kernel,
            a_name=branch_2_a_name,
            b_output_channels=branch_2_b_output_channels,
            b_kernel=branch_2_b_kernel,
            b_name=branch_2_b_name,
        )

        self.branch_3 = MixedLayerBranch(
            name="Branch_3",
            is_training=is_training,
            a_output_channels=None,
            a_kernel_shape=None,
            a_name=None,
            b_output_channels=branch_3_b_output_channels,
            b_kernel=branch_3_b_kernel,
            b_name=branch_3_b_name,
        )

    def __call__(self, net):
        """
        Connects the module to inputs.

        Args:
            net: Inputs to the MixedLayer component.

        Returns:
            Outputs from the module.
        """

        branch_0 = self.branch_0(net)
        branch_1 = self.branch_1(net)
        branch_2 = self.branch_2(net)
        branch_3 = self.branch_3(net)

        return tf.concat([branch_0, branch_1, branch_2, branch_3], 4)


class Logits(snt.Module):
    """
    Logits layer.
    """

    def __init__(
        self,
        num_classes,
        spatial_squeeze=True,
        is_training=False,
        dropout_keep_prob=1.0,
        name="logits",
    ):
        super(Logits, self).__init__(name=name)

        self._spatial_squeeze = spatial_squeeze
        self._dropout_keep_prob = dropout_keep_prob

        self.logits = Unit3D(
            output_channels=num_classes,
            kernel_shape=[1, 1, 1],
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            is_training=is_training,
            name="Conv3d_0c_1x1",
        )

    def __call__(self, net):
        net = tf.nn.avg_pool3d(
            net,
            ksize=[1, 2, 7, 7, 1],
            strides=[1, 1, 1, 1, 1],
            padding="VALID",
        )

        net = tf.nn.dropout(
            net,
            1 - self._dropout_keep_prob,
        )

        net = self.logits(net)

        if self._spatial_squeeze:
            net = tf.squeeze(net, [2, 3], name="SpatialSqueeze")

        return tf.reduce_mean(net, axis=1)


class InceptionI3d(snt.Module):
    """Inception-v1 I3D architecture.

    The model is introduced in:

      Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
      Joao Carreira, Andrew Zisserman
      https://arxiv.org/pdf/1705.07750v1.pdf.

    See also the Inception architecture, introduced in:

      Going deeper with convolutions
      Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
      Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
      http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(
        self,
        num_classes=400,
        spatial_squeeze=True,
        is_training=False,
        dropout_keep_prob=1.0,
        final_endpoint="Logits",
        name="inception_i3d",
    ):
        """Initializes I3D model instance.

        Args:
            num_classes: The number of outputs in the logit layer (default 400, which
                matches the Kinetics dataset).
            spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
                before returning (default True).
            is_training: whether to use training mode for snt.BatchNorm (boolean).
            dropout_keep_prob: Probability for the tf.nn.dropout layer (float in
                (0, 1]).
            final_endpoint: The model contains many possible endpoints.
                `final_endpoint` specifies the last endpoint for the model to be built
                up to. In addition to the output at `final_endpoint`, all the outputs
                at endpoints up to `final_endpoint` will also be returned, in a
                dictionary. `final_endpoint` must be one of
                InceptionI3d.VALID_ENDPOINTS (default 'Logits').
            name: A string (optional). The name of this module.

        Raises:
            ValueError:
                if `final_endpoint` is not recognized.
                if `dropout_keep_prob` is not in range (0, 1].

        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f"Unknown final endpoint {final_endpoint}")

        if not 0 < dropout_keep_prob <= 1:
            raise ValueError("dropout_keep_prob must be in range (0, 1]")

        super(InceptionI3d, self).__init__(name=name)

        self._final_endpoint = final_endpoint

        #
        # layers
        #

        # Conv3d_1a_7x7

        self.Conv3d_1a_7x7 = Unit3D(  # pylint: disable=invalid-name
            output_channels=64,
            kernel_shape=[7, 7, 7],
            stride=[2, 2, 2],
            is_training=is_training,
            name="Conv3d_1a_7x7",
        )

        # MaxPool3d_2a_3x3

        self.MaxPool3d_2a_3x3 = lambda x: tf.nn.max_pool3d(  # pylint: disable=invalid-name
            x,
            ksize=[1, 1, 3, 3, 1],
            strides=[1, 1, 2, 2, 1],
            padding="SAME",
            name="MaxPool3d_2a_3x3",
        )

        # Conv3d_2b_1x1

        self.Conv3d_2b_1x1 = Unit3D(  # pylint: disable=invalid-name
            output_channels=64,
            kernel_shape=[1, 1, 1],
            is_training=is_training,
            name="Conv3d_2b_1x1",
        )

        # Conv3d_2c_3x3

        self.Conv3d_2c_3x3 = Unit3D(  # pylint: disable=invalid-name
            output_channels=192,
            kernel_shape=[3, 3, 3],
            is_training=is_training,
            name="Conv3d_2c_3x3",
        )

        # MaxPool3d_3a_3x3

        self.MaxPool3d_3a_3x3 = lambda x: tf.nn.max_pool3d(  # pylint: disable=invalid-name
            x,
            ksize=[1, 1, 3, 3, 1],
            strides=[1, 1, 2, 2, 1],
            padding="SAME",
            name="MaxPool3d_3a_3x3",
        )

        # Mixed_3b

        self.Mixed_3b = MixedLayer(  # pylint: disable=invalid-name
            name="Mixed_3b",
            is_training=is_training,
            branch_0_a_output_channels=64,
            branch_1_a_output_channels=96,
            branch_1_b_output_channels=128,
            branch_2_a_output_channels=16,
            branch_2_b_output_channels=32,
            branch_3_b_output_channels=32,
        )

        # Mixed_3c

        self.Mixed_3c = MixedLayer(  # pylint: disable=invalid-name
            name="Mixed_3c",
            is_training=is_training,
            branch_0_a_output_channels=128,
            branch_1_a_output_channels=128,
            branch_1_b_output_channels=192,
            branch_2_a_output_channels=32,
            branch_2_b_output_channels=96,
            branch_3_b_output_channels=64,
        )

        # MaxPool3d_4a_3x3

        self.MaxPool3d_4a_3x3 = lambda x: tf.nn.max_pool3d(  # pylint: disable=invalid-name
            x,
            ksize=[1, 3, 3, 3, 1],
            strides=[1, 2, 2, 2, 1],
            padding="SAME",
            name="MaxPool3d_4a_3x3",
        )

        # Mixed_4b

        self.Mixed_4b = MixedLayer(  # pylint: disable=invalid-name
            name="Mixed_4b",
            is_training=is_training,
            branch_0_a_output_channels=192,
            branch_1_a_output_channels=96,
            branch_1_b_output_channels=208,
            branch_2_a_output_channels=16,
            branch_2_b_output_channels=48,
            branch_3_b_output_channels=64,
        )

        # Mixed_4c

        self.Mixed_4c = MixedLayer(  # pylint: disable=invalid-name
            name="Mixed_4c",
            is_training=is_training,
            branch_0_a_output_channels=160,
            branch_1_a_output_channels=112,
            branch_1_b_output_channels=224,
            branch_2_a_output_channels=24,
            branch_2_b_output_channels=64,
            branch_3_b_output_channels=64,
        )

        # Mixed_4d

        self.Mixed_4d = MixedLayer(  # pylint: disable=invalid-name
            name="Mixed_4d",
            is_training=is_training,
            branch_0_a_output_channels=128,
            branch_1_a_output_channels=128,
            branch_1_b_output_channels=256,
            branch_2_a_output_channels=24,
            branch_2_b_output_channels=64,
            branch_3_b_output_channels=64,
        )

        # Mixed_4e

        self.Mixed_4e = MixedLayer(  # pylint: disable=invalid-name
            name="Mixed_4e",
            is_training=is_training,
            branch_0_a_output_channels=112,
            branch_1_a_output_channels=144,
            branch_1_b_output_channels=288,
            branch_2_a_output_channels=32,
            branch_2_b_output_channels=64,
            branch_3_b_output_channels=64,
        )

        # Mixed_4f

        self.Mixed_4f = MixedLayer(  # pylint: disable=invalid-name
            name="Mixed_4f",
            is_training=is_training,
            branch_0_a_output_channels=256,
            branch_1_a_output_channels=160,
            branch_1_b_output_channels=320,
            branch_2_a_output_channels=32,
            branch_2_b_output_channels=128,
            branch_3_b_output_channels=128,
        )

        # MaxPool3d_5a_2x2

        self.MaxPool3d_5a_2x2 = lambda x: tf.nn.max_pool3d(  # pylint: disable=invalid-name
            x,
            ksize=[1, 2, 2, 2, 1],
            strides=[1, 2, 2, 2, 1],
            padding="SAME",
            name="MaxPool3d_5a_2x2",
        )

        # Mixed_5b

        self.Mixed_5b = MixedLayer(  # pylint: disable=invalid-name
            name="Mixed_5b",
            is_training=is_training,
            branch_0_a_output_channels=256,
            branch_1_a_output_channels=160,
            branch_1_b_output_channels=320,
            branch_2_a_output_channels=32,
            branch_2_b_output_channels=128,
            branch_2_b_name="Conv3d_0a_3x3",  # NOTE: this is different from the original tf v1 implementation
            branch_3_b_output_channels=128,
        )

        # Mixed_5c

        self.Mixed_5c = MixedLayer(  # pylint: disable=invalid-name
            name="Mixed_5c",
            is_training=is_training,
            branch_0_a_output_channels=384,
            branch_1_a_output_channels=192,
            branch_1_b_output_channels=384,
            branch_2_a_output_channels=48,
            branch_2_b_output_channels=128,
            branch_3_b_output_channels=128,
        )

        # Logits

        self.Logits = Logits(  # pylint: disable=invalid-name
            num_classes=num_classes,
            spatial_squeeze=spatial_squeeze,
            is_training=is_training,
            dropout_keep_prob=dropout_keep_prob,
            name="Logits",
        )

        # Predictions

        self.Predictions = lambda x: tf.nn.softmax(  # pylint: disable=invalid-name
            x,
            name="Predictions",
        )

    def __call__(self, net):
        """Connects the model to inputs.

        Args:
          net: Inputs to the model, which should have dimensions
              `batch_size` x `num_frames` x 224 x 224 x `num_channels`.

        Returns:
          A tuple consisting of:
            1. Network output at location `self._final_endpoint`.
            2. Dictionary containing all endpoints up to `self._final_endpoint`,
               indexed by endpoint name.

        Raises:
            ValueError:
                if net shape is not `batch_size` x `num_frames` x 224 x 224 x `num_channels`
        """

        if len(net.shape) != 5 or net.shape[2] != 224 or net.shape[3] != 224:
            raise ValueError(
                "Input tensor shape must be [batch_size, num_frames, 224, 224, num_channels]"
            )

        endpoints = {}

        for endpoint in self.VALID_ENDPOINTS:
            net = getattr(self, endpoint)(net)
            endpoints[endpoint] = net
            if endpoint == self._final_endpoint:
                break

        return net, endpoints
