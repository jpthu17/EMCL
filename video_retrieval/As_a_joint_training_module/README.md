## EMCL as a joint training module (EMCL+MMT)

EMCL can be easy to be incorporated into any existing methods.

### Requirements
* Python 3.7 
* Pytorch 1.4.0
* Transformers 3.1.0
* Numpy 1.18.1

```bash
# Install the requirements
pip install -r requirements.txt
```
### Preparing data
The video features can be found in sharing from [MMT](https://github.com/gabeur/mmt), i.e.,
```bash
# Create and move to mmt/data directory
mkdir data
cd data
# Download the video features
wget http://pascal.inrialpes.fr/data2/vgabeur/video-features/MSRVTT.tar.gz
wget http://pascal.inrialpes.fr/data2/vgabeur/video-features/activity-net.tar.gz
wget http://pascal.inrialpes.fr/data2/vgabeur/video-features/LSMDC.tar.gz
# Extract the video features
tar -xvf MSRVTT.tar.gz
tar -xvf activity-net.tar.gz
tar -xvf LSMDC.tar.gz
```

### Training
```bash
python -m train --config configs/*
```


## Acknowledge
* This code implementation are adopted from [MMT](https://github.com/gabeur/mmt).
We sincerely appreciate for their contributions. We sincerely appreciate for their contributions.