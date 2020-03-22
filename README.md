
Latent Domain Factorization for Multisource Domain Adaptation
==================

Recent works have shown that domain adaptation can be improved in multi-source domain settings if we align each pair of source domains apart from typical source to target domain adaptation. We posit that any source domain is inherently composed of multiple source domains, either by means of appearance, pose or style and propose a solution that discovers latent domains from the source domain automatically. Our end-to-end novel 'soft-MMD' loss function ensures that those latent domains whose domain shifts when aligned improve target domain adaptation. Our method outperforms domain adaptation methods that presume a single underlying domain.

### Requirements
-----------
 - Python 3.6+
 - PyTorch 1.0
- Linux (tested on Ubuntu 18.04)
- Torch vision from the source.
- torchnet as follows
```
pip install git+https://github.com/pytorch/tnt.git@master
```

To clone the repository
``` bash
$ git clone git@github.com:NiteshBharadwaj/code_MSDA_digit.git
```

- GPU: 8Gi, CPU: 8Gi, Batch Size: 128, 
	- Training Time USPS: < 5min Acc 96% (Same as paper in 3 epochs) 
	- Training Time MNIST-M: < 2hrs Acc 73% (Same as in paper in 80 epochs)

## Result Replication From Report
- Setup dataset using the instructions from subsequent sections
- Download the pre-trained models from the following google drive [link] and extract to "records" folder.
- Change the flag "eval_only" in main.py to "True"
- Bash scripts:
```
    bash ./experiment_do.sh [TAR DOM] 100 0 record/[EXPT_DIRECTORY] [DATA] [NUM CLUSTERS] no 	
```
Choices: 
DATA - digits, cars, office
TAR_DOM (digits) - mnistm, svhn, usps, syn, mnist
TAR_DOM (cars) - CCSurv
TAR_DOM (office) - amazon, dslr
NUM_CLUSTERS - >1 (Choose from report)
EXPT_DIRECTORY - Any top level directory inside records folder (to replicate the corresponding experiment's results)


e.g.
```
   bash ./experiment_do.sh CCSurv 100 0 record/cars_src_agg source_only cars 6 no
   bash ./experiment_do.sh CCSurv 100 0 record/cars_src_tar source_only cars 6 no
   bash ./experiment_do.sh CCSurv 100 0 record/cars_soft6_agg soft_cluster cars 6 no
   bash ./experiment_do.sh dslr 100 0 record/odslr_src_agg_3000 source_only office 6 no
   bash ./experiment_do.sh dslr 100 0 record/odslr_src_tr_long source_target_only office 2 no
   bash ./experiment_do.sh dslr 100 0 record/odslr_soft12_agg source_target_only office 12 no
   bash ./experiment_do.sh dslr 100 0 record/odslr_soft6_agg_2 source_target_only office 6 no                             
```
## Setup for different Datasets

### Digit-Five
- Download dataset 
	- https://drive.google.com/open?id=1A4RJOFj4BJkmliiEL7g9WzNIDUHLxfmm
	- **Note:** MNIST-M generated based on [1], (differs from the DANN paper)
- If you're running headless, use the following snippet: 
```
    ggID='1A4RJOFj4BJkmliiEL7g9WzNIDUHLxfmm'  
    ggURL='https://drive.google.com/uc?export=download'  
    filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"  
    getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"  
    curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${filename}"  
```    
- Create a folder "data" and "record" in main working directory
- Extract contents of the zip file to ./data
- Run the following bash script to train
```
    bash experiment_do.sh mnistm 100 0 record/mnistm soft_cluster digits 4 yes
```
- Change second argument to usps/svhn/syn to train corresponding models



### Office-31
- Download dataset 
	- https://mega.nz/#F!dTAEDaaT!McxSMcL4Mf_hfID1Q7tSGA

- Create a folder "data" and "record" in main working directory
- Extract contents of the zip file to ./data
- Run the following bash script
```
    bash ./experiment_do.sh amazon 200 0 record/amazon_source_only source_only office 2 False
```
- Change second argument to dslr/webcam


### CompCars     
- In 'data' folder create 'CCWeb' and 'CCSurv' sub folders
- From main directory, run 
```
bash download_cc.sh
cd data/CCSurv/
zip -F sv_data.zip --out sv_data_combined.zip
unzip -P  d89551fd190e38 sv_data_combined.zip
cd ../CCWeb/
zip -F data.zip --out data_combined.zip
unzip -P  d89551fd190e38 data_combined.zip
```

- Training Compcars
```
bash experiment_do.sh  CCSurv 100 0 record/compcats soft_cluster cars
```
## Citation
```
@article{peng2018moment,
        title={Moment Matching for Multi-Source Domain Adaptation},
        author={Peng, Xingchao and Bai, Qinxun and Xia, Xide and Huang, Zijun and Saenko, Kate and Wang, Bo},
        journal={arXiv preprint arXiv:1812.01754},
        year={2018}
        }
```
             

