#!/usr/bin/env bash
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

python evaluate_sample.py --eval_type rgb > out/no_imagenet_rgb.txt
python evaluate_sample.py --eval_type flow > out/no_imagenet_flow.txt
python evaluate_sample.py --eval_type joint > out/no_imagenet_joint.txt
python evaluate_sample.py --eval_type rgb600 > out/no_imagenet_rgb600.txt
python evaluate_sample.py --imagenet_pretrained --eval_type rgb > out/imagenet_rgb.txt
python evaluate_sample.py --imagenet_pretrained --eval_type flow > out/imagenet_flow.txt
python evaluate_sample.py --imagenet_pretrained --eval_type joint > out/imagenet_joint.txt
