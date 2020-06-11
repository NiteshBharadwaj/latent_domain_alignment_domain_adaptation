
Latent Domain Alignment for Domain Adaptation
==================

Deep unsupervised domain adaptation aligns a labeled source domain with an unlabeled target domain, to achieve better classification in the latter. In this work, we posit that a source domain may inherently consist of multiple domains due to known variations such as pose, style, and appearance, or even unknown variations. We propose to improve domain adaptation by automatically discovering and aligning such latent domains. In contrast to prior works on multi-source adaptation that assume known or one-hot domain assignments, our novel Latent Domain Alignment for Domain Adaptation (LDADA) framework discovers a soft assignment of source samples to latent domains, while aligning them amongst themselves and with the target data. We present theoretical support and conduct experiments on public datasets to show that our method outperforms others that assume a single underlying domain shift. 

### Requirements
-----------
- Python 3.6+
- PyTorch 1.0
- Linux (tested on Ubuntu 18.04)
- Torch vision from the source.
- torchnet as follows

```bash
pip install git+https://github.com/pytorch/tnt.git@master
```

### Digit-Five
#### Dataset Setup
- Download dataset 
	- https://drive.google.com/open?id=1A4RJOFj4BJkmliiEL7g9WzNIDUHLxfmm
	
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


#### Model Download Setup
- Download the pre-trained models from the following google drive
- [link](download the zip file or models) and extract to "records" folder

### Experiments
##### For evaluation
- Change the flag "eval_only" in main.py to "True"

##### For training

```python
   python3 main.py --target [TARGET_DOMAIN] --dl_type [CLUSTERING_METHOD] --num_domain [NUM_LATENT_DOMAIN] --class_disc [BOOL_CLASS_DISCREPANCY] --record_folder [RECORD_FOLDER] --seed 0 --office_directory [DIGITS_DIRECTORY] --data 'digits' --max_epoch [NUM_EPOCHS] --kl_wt [KL_WEIGHT] --entropy_wt [ENTROPY_WEIGHT] --to_detach 'yes' --msda_wt [MSDA_WEIGHT]
```

Choices: 
- DATA - digits, cars, office
- TARGET_DOMAIN -> target
- CLUSTERING_METHOD - soft_cluster, hard_cluster, source_only, source_target_only
- NUM_LATENT_DOMAIN - Number of inherent latent domains (Choose from paper)
- BOOL_CLASS_DISCREPANCY - yes/no for choosing Maximum Classifier Discrepancy
- RECORD_FOLDER - Top level directory inside records folder (to replicate the corresponding experiment's results)
- KL_WEIGHT, ENTROPY_WEIGHT, MSDA_WEIGHT - hyperparameter scaling factors for different losses as reported

E.g.
1. SVHN with soft_cluster (k=4)
```python
python3 main.py --target 'svhn' --dl_type 'soft_cluster' --num_domain 4 --class_disc 'no' --record_folder '/results/svhn-sc0' --seed 0 --office_directory '/data/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.001
```

2. SVHN Baseline
```python
python3 main.py --target 'svhn' --dl_type 'source_target_only' --num_domain 4 --class_disc 'no' --record_folder '/results/svhn-sc0' --seed 0 --office_directory '/data/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.001
```
