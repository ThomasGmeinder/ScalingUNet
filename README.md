# Query: UNet Model for Image Segmentation\n-------------------------------------
# ContextLines: 1

UNet Model for Image Segmentation
---------------------------------

#### Setup

This was tested with Poplar SDK Version 2.1. 
After setup of this SDK, additional Python Packages have to be installed with `pip3 install -r requirements.txt`

#### Models

* custom_unet_four_IPUs: A pipelined model using four IPUs
* custom_unet_six_IPUs: A pipelined model using six IPUs
* custom_unet_small: A small model that fits on a single IPU

The models will be selected according to the `--num_IPU` parameter of the `training.py` traing script

#### Training commands
* Single IPU with reporting enabled: `POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./report_single_IPU"}' python training.py --num_IPU 1 --gradient-accumulation-count 16`
* Four IPUs with reporting enabled and AMP values set per IPU:  POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./report_four_IPUs_AMPperIPU"}'  python training.py --num_IPU 4 --available-memory-proportion 0.6 0.6 0.2 0.6




