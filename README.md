## ATSG: Adaptive Token Linking With Segment Anything Model Guidance For Weakly Supervised Remote Sensing Image Semantic Segmentation

This is the official repository of TGRS 2026 paper: *ATSG: Adaptive Token Linking With Segment Anything Model Guidance For Weakly Supervised Remote Sensing Image Semantic Segmentation*.


<div align="center">

<br>
  <img width="100%" alt="Framework of CTFA" src="./docs/imgs/Framework of CTFA.jpg">
</div>



## Data Preparations
<details>
<summary>
iSAID dataset
</summary>

#### 1. Data Download

You may download the iSAID dataset from their official webiste https://captain-whu.github.io/iSAID/dataset.html.


#### 2. Data Preprocessing
After downloading, you may craft your own dataset. Please refer to datasets/iSAID/make_data.py.

</details>

<details>

<summary>
ISPRS Potsdam dataset
</summary>

#### 1. Data Download
Datasets for ISPRS Potsdam are widely accessible on the Internet. You may find the original content on: https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx. 

#### 2. Data Preprocessing
You may refer to the datasets/potsdam/potsdam_clip_dataset.py provided by [OME](https://github.com/NJU-LHRS/OME). Great thanks for their brilliant works.
</details>

<details>

<summary>
Deepglobe Land Cover Classification Dataset
</summary>

#### 1. Data Download
You may find the original content on:http://deepglobe.org/challenge.html. 

#### 2. Data Preprocessing
Please refer to datasets/deepglobe/deepglobe_clip_dataset.py.

</details>

We also provide the BaiduNetDiskDownload link for processed dataset at [Here](https://pan.baidu.com/s/1GzSzHtYr2kRr0bl2ymoKYg). Code: CTFA

Checkpoints may try [this](https://pan.baidu.com/s/18wz4jmMXXNSdC_0acJsJZw). Code:r1k6 



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
