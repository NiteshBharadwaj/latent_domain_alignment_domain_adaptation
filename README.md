
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

- GPU: 8Gi, CPU: 8Gi, Batch Size: 128, 
	- Training Time USPS: < 5min Acc 96%
	- Training Time MNIST-M: < 2hrs Acc 73% 

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
```

```
    bash experiment_do.sh mnistm 100 0 record/mnistm soft_cluster digits 4 yes
```
- Change second argument to usps/svhn/syn to train corresponding models
