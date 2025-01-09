## MedSAM2 within Labelme  

We incorporated MedSAM2 into [Labelme](https://github.com/wkentaro/labelme) to perform breast cancer image segmentation.  
MedSAM2 achieves better performance in medical image segmentation. Based on this, we performed fine-tuning.  
  

<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/main%20window.png" width=100%>

## Usage

Run `labelme --help` for detail.  

```bash
conda activate labelme
labelme  # just open gui
```

### Semi-automatic segmentation
You don't need to modify the AI model.  
Open your image folder, and click `Semi-automatic segmentation` button.
Or use the `Ctrl + M` shortcut to perform semi-automatic segmentation with MedSAM2.
Then a red dot will appear at the location you click on the window.  
Click on the window , and press Enter after finishing. 
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/_2025-01-08%20111411.gif" width="100%" />

### Automatic segmentation
Open your image folder, and click `Automatic segmentation` button.  
Or use the `Ctrl + Shift + M` shortcut to perform automatic segmentation with MedSAM2. 
You don't need to modify the AI model. 
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/_2025-01-08%20112905.gif" width=100%>

### Save json file
It will automatically save a JSON file, and the file will be named as `filename.json`.  
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/screen%202025-01-08%20112424.png" width="100%" />

## Installation

### Our environment
```bash
Windows
Python3.12.6
Torch 2.5.1
CUDA 12.4
```

There are options:

- Platform agnostic installation: [Anaconda](#anaconda)
- Platform specific installation: [Ubuntu](#ubuntu), [macOS](#macos), [Windows](#windows)
- Pre-build binaries from [the release section](https://github.com/labelmeai/labelme/releases)

### Python version
Please use [Python](https://www.python.org/downloads/windows/) 3.12, check your python version  
```bash
python -V
```
If your Python version is not 3.12,  
```bash
# Anaconda
conda install python=3.12
# Ubuntu
sudo apt install python3.12
# pip
pip install python==3.12.6
```

### Anaconda

You need install [Anaconda](https://www.continuum.io/downloads), then run below:

```bash
# python3
conda create --name=labelme python=3
source activate labelme
# conda install -c conda-forge pyside2
# conda install pyqt
# pip install pyqt5  # pyqt5 can be installed via pip on python3
pip install labelme
# or you can install everything by conda command
# conda install labelme -c conda-forge
```

### Ubuntu

```bash
sudo apt-get install labelme

# or
sudo pip3 install labelme

# or install standalone executable from:
# https://github.com/labelmeai/labelme/releases

# or install from source
pip3 install git+https://github.com/labelmeai/labelme
```

### macOS

```bash
brew install pyqt  # maybe pyqt5
pip install labelme

# or install standalone executable/app from:
# https://github.com/labelmeai/labelme/releases

# or install from source
pip3 install git+https://github.com/labelmeai/labelme
```

### Windows

Install [Anaconda](https://www.continuum.io/downloads), then in an Anaconda Prompt run:

```bash
conda create --name=labelme python=3
conda activate labelme
pip install labelme

# or install standalone executable/app from:
# https://github.com/labelmeai/labelme/releases

# or install from source
pip3 install git+https://github.com/labelmeai/labelme
```

## Download Pytoch
We need to use [Pytorch](https://pytorch.org/) gpu version
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## Setup MedSAM2
Move the files inside `MedSAM2`(our folder) to `anaconda3/envs/labelme/Lib/site-packages/labelme`, and replace the old `app.py` with the new `app.py` .

You can download MedSAM2_pretrain checkpoint from checkpoints folder`MedSAM2-click/checkpoints`:
```bash
bash download_ckpts.sh

#or you can open download_ckpt.sh then copy the url and download on your browser.
```
Then, move the downloaded checkpoint `MedSAM2_pretrain.pth` to the folder `anaconda3/envs/labelme/Lib/site-packages/labelme/checkpoints`.  
### If you can't find it, make good use of the file manager's search function.  

### Set shortcuts
In folder `anaconda3/envs/labelme/Lib/site-packages/labelme/config/default_config.yaml`  
Add these two lines of code at the end.  
```bash
  auto_med: Ctrl+Shift+M
  semi_med: Ctrl+M
```

### Set Icons
Download icons `medical.png` in our folder `MedSAM2-auto/icons`, then place it into the folder`anaconda3/envs/labelme/Lib/site-packages/labelme/icons`.

### After downloading the checkpoint
You need to open `app.py`, and update the path to the checkpoint file, which is around line 1883.  
It is recommended to use an absolute path to ensure no errors occur.

## Some possible errors  
### You might encounter this Qt error on Windows 11.  
<img src="https://github.com/worldstar/MedSAM2-labelme/blob/main/examples/medsam2/error1.png"/>  
You need to rename the `Qt5` folder to `Qt` in the `anaconda3/envs/labelme/Library/qml/` directory.
  
### No such file or directory for your checkpoint.  
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'path/to/checkpoints/MedSAM2_pretrain.pth'
```
You need to change it to an absolute path.

### For other `ModuleNotFoundError: No module named` errors.
You just need to use pip or conda to install the corresponding module.

## Train yourself

We need to use [roboflow](https://roboflow.com/).  
Roboflow supports exporting segmentation datasets to the SAM-2.1 format, ideal for use in this guide. You can upload segmentation datasets in the COCO JSON Segmentation format then convert them to SAM-2.1 for use in this guide.  
<img src="https://github.com/worldstar/MedSAM2-labelme/blob/main/examples/medsam2/sam2export.png">
We then download a SAM-2.1 training YAML file which we will use to configure our model training job.

Finally, we install SAM-2.1 and download the model checkpoints.

Replace the below code with the code to export your dataset. You can also use the same code above to fine-tune our car parts dataset. Note: If you use the car parts dataset pre-filled below, you will still need to add a `Roboflow API key`.

### Download Roboflow
```bash
!pip install roboflow

from roboflow import Roboflow
import os

rf = Roboflow(api_key="your-api-key")
project = rf.workspace("workspace-name").project("project-name")
version = project.version(1)
dataset = version.download("sam2")

# rename dataset.location to "data"
os.rename(dataset.location, "/content/data")
```
```bash
!git clone https://github.com/facebookresearch/sam2.git
```
```bash
!wget -O /content/sam2/sam2/configs/train.yaml 'https://drive.usercontent.google.com/download?id=11cmbxPPsYqFyWq87tmLgBAQ6OZgEhPG3'

%cd ./sam2/
```

### Install SAM-2
The SAM-2 installation process may take several minutes.  
```bash
!pip install -e .[dev] -q
```
download checkpoint  
```bash
!cd ./checkpoints && ./download_ckpts.sh
```

### Modify Dataset File Names
SAM-2.1 requires dataset file names to be in a particular format. Run the code snippet below to format your dataset file names as required.  
```bash
# Script to rename roboflow filenames to something SAM 2.1 compatible.
# Maybe it is possible to remove this step tweaking sam2/sam2/configs/train.yaml.
import os
import re

FOLDER = "/content/data/train"

for filename in os.listdir(FOLDER):
    # Replace all except last dot with underscore
    new_filename = filename.replace(".", "_", filename.count(".") - 1)
    if not re.search(r"_\d+\.\w+$", new_filename):
        # Add an int to the end of base name
        new_filename = new_filename.replace(".", "_1.")
    os.rename(os.path.join(FOLDER, filename), os.path.join(FOLDER, new_filename))
```

### Start Traning
You can now start training a SAM-2.1 model. The amount of time it will take to train the model will vary depending on the GPU you are using and the number of images in your dataset.  
```bash
!python training/train.py -c 'configs/train.yaml' --use-cluster 0 --num-gpus 1
```

You can visualize the model training graphs with Tensorboard:
```bash
%load_ext tensorboard
%tensorboard --bind_all --logdir ./sam2_logs/
```

### Visualize Model Results
With a trained model ready, we can test the model on an image from our test set.  
To assist with visualizing model predictions, we are going to use Roboflow supervision, an open source computer vision Python package with utilities for working with vision model outputs  
```bash
!pip install supervision -q
```

### Load SAM2-1
```bash
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import supervision as sv
import os
import random
from PIL import Image
import numpy as np

# use bfloat16 for the entire notebook
# from Meta notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

checkpoint = "/content/sam2/sam2_logs/configs/train.yaml/checkpoints/checkpoint.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2 = build_sam2(model_cfg, checkpoint, device="cuda")
mask_generator = SAM2AutomaticMaskGenerator(sam2)

checkpoint_base = "/content/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg_base = "configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2_base = build_sam2(model_cfg_base, checkpoint_base, device="cuda")
mask_generator_base = SAM2AutomaticMaskGenerator(sam2_base)
```

### Run Inference on an Image in Automatic Mask Generation Mode
```bash
validation_set = os.listdir("/content/data/valid")

# choose random with .json extension
image = random.choice([img for img in validation_set if img.endswith(".jpg")])
image = os.path.join("/content/data/valid", image)
opened_image = np.array(Image.open(image).convert("RGB"))
result = mask_generator.generate(opened_image)

detections = sv.Detections.from_sam(sam_result=result)

mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
annotated_image = opened_image.copy()
annotated_image = mask_annotator.annotate(annotated_image, detections=detections)

base_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)

base_result = mask_generator_base.generate(opened_image)
base_detections = sv.Detections.from_sam(sam_result=base_result)
base_annotated_image = opened_image.copy()
base_annotated_image = base_annotator.annotate(base_annotated_image, detections=base_detections)

sv.plot_images_grid(images=[annotated_image, base_annotated_image], titles=["Fine-Tuned SAM-2.1", "Base SAM-2.1"], grid_size=(1, 2))
```

### Final
You need to move `best.pt` to the `anaconda3\envs\labelme\Lib\site-packages\labelme\checkpoints` directory.  
Then, update the `model path` in `app.py`,which is around line 1883.
 
## How to fine-tune MedSAM2
If the fine-tuned MedSAM2 can be used for automatic annotation and provides high accuracy, then this MedSAM2 can certainly be used for manual annotation.  
  
### At the beginning, select data
It is necessary to select some data with distinct features.  
Like this kind of data, where the tumor is relatively complete and clearly visible.  
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/IM-0023-0039.png" width=50%>  

### Dataset content
Your dataset must include both images with tumors and images without tumors (treated as background). Otherwise, MedSAM2 might mistakenly annotate the heart.  
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/medsam2-heart.png" width=50%>  

### Training
After the first round of training, the model will look like this.  
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/medsam2-fine-tune.png" width=50%>  
Then, you need to modify the dataset by adding some less obvious tumors and continue training using the fine-tuned MedSAM2.  

## Acknowledgement

This repo is the fork of [mpitid/pylabelme](https://github.com/mpitid/pylabelme).





