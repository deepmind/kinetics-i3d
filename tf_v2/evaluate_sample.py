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
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
import numpy as np
import tensorflow as tf

import i3d

_SAMPLE_PATHS = {
    "rgb": "data/v_CricketShot_g04_c01_rgb.npy",
    "flow": "data/v_CricketShot_g04_c01_flow.npy",
}

_CHECKPOINT_PATHS_SCRATCH = {
    "rgb": "data/checkpoints/rgb_scratch/model.ckpt",
    "flow": "data/checkpoints/flow_scratch/model.ckpt",
    "rgb600": "data/checkpoints/rgb_scratch_kin600/model.ckpt",
}

_CHECKPOINT_PATHS_IMAGENET = {
    "rgb": "data/checkpoints/rgb_imagenet/model.ckpt",
    "flow": "data/checkpoints/flow_imagenet/model.ckpt",
}

_LABEL_MAP_PATH = "data/label_map.txt"
_LABEL_MAP_PATH_600 = "data/label_map_600.txt"


def parse_args() -> argparse.Namespace:
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_type",
        default="joint",
        choices=["rgb", "flow", "joint", "rgb600"],
        help="Type of evaluation",
    )

    parser.add_argument(
        "--imagenet_pretrained",
        action="store_true",
        help="Use ImageNet pretrained weights. Not availble for rgb600",
    )

    return parser.parse_args()


def main():
    """
    Main function to evaluate I3D on Kinetics.
    """

    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    eval_type = args.eval_type
    imagenet_pretrained = args.imagenet_pretrained

    if eval_type == "rgb600" and imagenet_pretrained:
        raise ValueError("Kinetics 600 not available for ImageNet pretrained model")

    _checkpoint_paths = (
        _CHECKPOINT_PATHS_IMAGENET if imagenet_pretrained else _CHECKPOINT_PATHS_SCRATCH
    )

    kinetics_classes = (
        [x.strip() for x in open(_LABEL_MAP_PATH_600, encoding="utf-8")]
        if eval_type == "rgb600"
        else [x.strip() for x in open(_LABEL_MAP_PATH, encoding="utf-8")]
    )

    num_classes = 600 if eval_type == "rgb600" else 400

    if eval_type in ["rgb", "rgb600", "joint"]:
        # Instantiate the model for RGB
        rgb_model = i3d.InceptionI3d(num_classes, spatial_squeeze=True, final_endpoint="Logits")

        # Restore the checkpoint
        tf.train.Checkpoint(model=rgb_model).restore(
            _checkpoint_paths["rgb600" if eval_type == "rgb600" else "rgb"]
        )
        logging.info("RGB checkpoint restored")

        # Load the sample video
        rgb_sample = tf.convert_to_tensor(np.load(_SAMPLE_PATHS["rgb"]), dtype=tf.float32)
        logging.info("RGB sample loaded")

        # Run the model
        rgb_logits, _ = rgb_model(rgb_sample)

    if eval_type in ["flow", "joint"]:
        # Instantiate the model for flow
        flow_model = i3d.InceptionI3d(num_classes, spatial_squeeze=True, final_endpoint="Logits")

        # Restore the checkpoint
        tf.train.Checkpoint(model=flow_model).restore(_checkpoint_paths["flow"])
        logging.info("Flow checkpoint restored")

        # Load the sample video
        flow_sample = tf.convert_to_tensor(np.load(_SAMPLE_PATHS["flow"]), dtype=tf.float32)
        logging.info("Flow sample loaded")

        # Run the model
        flow_logits, _ = flow_model(flow_sample)

    if eval_type in ["rgb", "rgb600"]:
        out_logits = rgb_logits
    elif eval_type == "flow":
        out_logits = flow_logits
    else:
        out_logits = rgb_logits + flow_logits

    out_predictions = tf.nn.softmax(out_logits)

    out_logits = out_logits[0]
    out_predictions = out_predictions[0]
    sorted_indices = np.argsort(out_predictions)[::-1]

    print(f"Norm of logits: {np.linalg.norm(out_logits)}")
    print("\nTop classes and probabilities")
    for index in sorted_indices[:20]:
        print(out_predictions[index].numpy(), out_logits[index].numpy(), kinetics_classes[index])


if __name__ == "__main__":
    main()
