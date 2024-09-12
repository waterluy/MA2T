#!/bin/bash

python ./analysis_tools/visualize/run.py \
    --predroot ./test/base_e2e/attack_no_noise_ql2.pkl \
    --out_folder ./test/base_e2e/attack_no_noise_ql2/vis \
    --demo_video now.avi \
    --project_to_cam True



