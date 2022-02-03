# Intro

Implementation of our currently unpublished work [Certainty Volume Prediction for Unsupervised Domain Adaptation](https://arxiv.org/abs/2111.02901).

# Basic setup

- Install environment with ```conda env create --file=environment.yaml --name CVP```
- Modify dataset paths in ```configs/global_config.yaml```, keep the \<domain>, \<class> and \<image> template parameters in the path
- Activate env with ```conda activate CVP```
- Run the script with:
```
CUDA_VISIBLE_DEVICES=0,1 python3 src/run.py \
--source-dataset Adaptiope/real_life \
--target-dataset Adaptiope/synthetic \
--configs configs/datasets/adaptiope.yaml configs/archs/resnet101.yaml \
--sub-dir TESTING --comment CHECKING_SETUP
```

- Configs passed with the ```--config``` flag are read from left to right, i.e. keys in later configs can overwrite matching keys in earlier configs. 
  ```configs/global_config.yaml``` is always read first.
- The training output directory is also defined in ```global_config.yaml``` and makes use of the ```--sub-dir``` and ```--comment``` flags.


# Note on reproducibility

For reproducibility, we recommend using Ubuntu 18.04 with NVIDIA driver version 450.102.04 while using 2x 1080Ti GPUs and the above setup steps.
The code itself is fully deterministic, multiple runs with the same seed should yield the exact same result.
