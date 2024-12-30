## MedSAM2 within Labelme  

We incorporated MedSAM2 into [Labelme](https://github.com/wkentaro/labelme) to perform image segmentation.  
MedSAM2 achieves better performance in medical image segmentation.  

<img src="https://github.com/worldstar/MedSAM2-labelme/blob/main/examples/medsam2/main%20windows.png" width=100%>

## Usage

Run `labelme --help` for detail.  

```bash
conda activate labelme
labelme  # just open gui
```

Open your image, and click medsam2 button.  
You don't need to modify the AI model.  
Then a red dot will appear at the location you click on the window.  
  
<img src="https://github.com/worldstar/MedSAM2-labelme/blob/main/examples/medsam2/medsam2-labelme-example.gif" width="100%" />

Click on the window , and press Enter after finishing.  
It will automatically save a JSON file, and the file will be named as `filename.json`.  
  
<img src="https://github.com/worldstar/MedSAM2-labelme/blob/main/examples/medsam2/screen%202024-12-23%20170840.png" width="100%" />

## Installation

There are options:

- Platform agnostic installation: [Anaconda](#anaconda)
- Platform specific installation: [Ubuntu](#ubuntu), [macOS](#macos), [Windows](#windows)
- Pre-build binaries from [the release section](https://github.com/labelmeai/labelme/releases)

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

## Setup MedSAM2 semi-automatic

Move the files inside MedSAM2-click to `anaconda3\envs\labelme\Lib\site-packages\labelme`, and replace the old `app.py` with the new `app.py` .

You can download MedSAM2_pretrain checkpoint from checkpoints folder`MedSAM2-click/checkpoints`:
```bash
bash download_ckpts.sh

#or you can open download_ckpt.sh then copy the url and download on your browser.
```
Then, move the downloaded checkpoint `MedSAM2_pretrain.pth` to the folder `anaconda3\envs\labelme\Lib\site-packages\labelme\checkpoints`.

### After downloading the checkpoint
You need to open `app.py`, and update the path to the checkpoint file, which is around line 1883.  
It is recommended to use an absolute path to ensure no errors occur.

### Download Pytoch
We need to use [Pytorch](https://pytorch.org/) gpu version
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## Some possible errors  
You might encounter this Qt error on Windows 11.  
<img src="https://github.com/worldstar/MedSAM2-labelme/blob/main/examples/medsam2/error1.png">  
You need to rename the `Qt5` folder to `Qt` in the `anaconda3\envs\labelme\Library\qml\` directory.

No such file or directory for your checkpoint.  
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'path/to/checkpoints/MedSAM2_pretrain.pth
```
You need to change it to an absolute path.

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

### After Traning
You need to move `best.pt` to the `anaconda3\envs\labelme\Lib\site-packages\labelme\checkpoints` directory.  
Then, update the `model path` in `app.py`,which is around line 1883.

## Acknowledgement

This repo is the fork of [mpitid/pylabelme](https://github.com/mpitid/pylabelme).





