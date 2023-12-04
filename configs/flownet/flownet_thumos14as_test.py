""" flownet_thumos14as_test.py

For testing (ie. extracting flows) on THUMOS14 actionformer subset using flownet.
This is the config that would get passed to ../../tools/dist_test.sh
"""

# No scheduler config, as we are not training at this stage.
_base_ = [
    "../_base_/models/flownetc.py",
    "../_base_/datasets/thumos14as_test.py",
    "../_base_/default_runtime.py",
]
