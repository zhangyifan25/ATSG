## ATSG: Adaptive Token Linking With Segment Anything Model Guidance For Weakly Supervised Remote Sensing Image Semantic Segmentation

This is the official repository of TGRS 2026 paper: *ATSG: Adaptive Token Linking With Segment Anything Model Guidance For Weakly Supervised Remote Sensing Image Semantic Segmentation*.


<div align="center">

<br>
  <img width="100%" alt="Framework of CTFA" src="./doc/overall.png">
</div>



## Data Preparations

11111





## Create environment
We provide our requirements file for building the environemnt. Note that extra packages may be downloaded.
``` bash 
## Download Dependencies.
pip install -r requirements.txt 
```

### Build Reg Loss

To use the regularized loss, download and compile the python extension, see [Here](https://github.com/meng-tang/rloss/tree/master/pytorch#build-python-extension-module).

### Train
Please refer to the scripts folder, where all scripts are clared by their name. You can also modify them to distributed training, which cost more GPUs. A simple startup like this:
```bash
## for iSAID
python dist_train_iSAID_seg_neg_fp.py
## for potsdam
python dist_train_postdam_seg_neg_fp.py
## for deepglobe
python dist_train_deepglobe_seg_neg_fp.py
```

You should remember to change the data path to your own and make sure all setting are matched.

I will try my best to reorganize the code to minimize issues. Apologize for any inconvenience caused by the code issues and thank you for your understanding.




## Acknowledgement

Our work is built on the codebase of [SAMRS](https://github.com/sstary/SSRS) and [CTFA](https://github.com/ZaiyiHu/CTFA). We sincerely thank for their exceptional work.
