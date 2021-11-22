UNet Model for Image Segmentation
---------------------------------

#### Setup

This was tested with Poplar SDK Version 2.3. 
After setup of this SDK, additional Python Packages have to be installed with `pip3 install -r requirements.txt`

#### Models

* custom_unet: Original UNet model. 
* custom_unet_small: A small model that fits on a single IPU

The models will be selected according to the `--num_IPU` parameter of the `training.py` traing script
With `--num_IPU` set > 1 the custom_unet model can be pipelined across the IPUs using the`-lnp` argument. See `--help` output of `training.py` for more details.

#### Training commands

* Single IPU with reporting enabled: 
`python training.py --num_IPU 1 --gradient-accumulation-count 16 --profile true`
* Pipelined across four IPUs with reporting enabled and AMP values set per IPU: 
`python training.py --num_IPU 4 --available-memory-proportion 0.6 0.6 0.2 0.6 -lnp 7 34 50 --profile true`




