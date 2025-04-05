## MedSAM2 within Labelme   
Our goal is to significantly reduce the time doctors spend annotating images and make the tool accessible for non-professional users as well.  
So we integrated MedSAM2 into [Labelme](https://github.com/wkentaro/labelme)  to perform segmentation on breast cancer images and cardiac ultrasound images.  
While MedSAM2 has shown good performance in medical image segmentation overall, our tests revealed that its performance on multimodal breast cancer images and cardiac ultrasound segmentation tasks was suboptimal.  
Therefore, we fine-tuned MedSAM2 based on these findings and we have now implemented fully automated segmentation of multimodal breast cancer images without any human intervention.  
The following interface includes button `Med-SAM2` that we have added.  

<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/main-windows.png" width=100%/>
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/demonstration.png" width=100%/>

## Usage

Run `labelme --help` for detail.  

```bash
conda activate labelme
labelme  # just open gui
```

## Mode Selection
We offer four mode options: two fully automated(`ENTIRE FOLDER`, `ONE IMAGE`) and two semi-automated(`BATCH TASK`, `GENERAL`). The fully automated modes are divided into single image and entire folder, while the semi-automated modes are divided into batch processing tasks and general mode.
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/mode-select.png" width=100%/>

### Semi-automatic mode : SEMI-AUTO(CLICK)
Click `SEMI-AUTO(CLICK)` button.
Then a red dot will appear at the location you click on the window.  
Click on the window , and press `Enter` after finishing. 
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/dce-semi.gif" width="100%" />  

### Click Noise Points for SEMI-AUTO(CLICK)
By selecting some noise points, we can further improve the accuracy. The default mode is `Select Subject` and you can switch between the `Select Subject` and `Select Background` modes by clicking the respective buttons.  
Before using `Select Background` ：  
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/dce-show-bg.png" width="100%">  
After using `Select Background` ：  
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/dce-show-bg.gif" width="100%">  

### Transfer Point for SEMI-AUTO(CLICK)
In semi-automatic segmentation for consecutive images, pressing `Enter` will allow the use of points from the previous image.  
Note that the `BATCH TASK` mode does not support point passing.  
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/infect-point.gif" width="100%">  

### Semi-automatic mode : BATCH-TASK(CLICK)
This mode provides a button labeled `BATCH TASK(CLICK)` or the `Esc` key on the keyboard as a shortcut, allowing multiple segmentations within a single image. The final segmentation results are collected into a single JSON file. This is useful, for example, when annotating ultrasound images of the heart.  
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/heart.gif" width="100%" />  

### Automatic mode : ONE IMAGE(AUTO) and ENTIRE FOLDER(AUTO)
Open your image folder, and click `ONE IMAGE(AUTO)` button or `ENTIRE FOLDER(AUTO)` button.  
`ONE IMAGE (AUTO)` is for segmenting the current image, while `ENTIRE FOLDER (AUTO)` is for segmenting all images in the current folder, with the segmentation progress displayed in the terminal.  
The fully automated segmentation supports ADC, DCE, DWI, and PET images, but currently, it does not support cardiac ultrasound images.  

### Demonstration of Automatic mode
ADC ( Apparent Diffusion Coefficient ) :  
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/auto-adc.gif" width=100%>  
DCE ( Dynamic Contrast-Enhanced ) :  
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/auto-dce.gif" width=100%>  
DWI ( Diffusion Weighted Imaging ) :  
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/auto-dwi.gif" width=100%>  
PET ( Positron Emission Tomography ) :  
<img src="https://github.com/worldstar/MedSAM2-Labelme/blob/main/examples/medsam2/auto-pet.gif" width=100%>  

### Save json file
It will automatically save a JSON file, and the file will be named as `filename.json`.  

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
You might encounter some issues because our development team has not used macOS.  
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
We need to use [Pytorch](https://pytorch.org/) ,select your version and download it.
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## Setup MedSAM2
Move the files inside `MedSAM2`(our folder) to `anaconda3/envs/labelme/Lib/site-packages/labelme`, and replace the old `app.py` with the new `app.py` .

You can download MedSAM2_pretrain checkpoint from checkpoints folder`MedSAM2/checkpoints`(This is the folder on our GitHub.):
```bash
bash download_ckpts.sh

#or you can open download_ckpt.sh then copy the url and download on your browser.
#Please note that this is the pre-trained MedSAM2 model, not our fine-tuned version.
```
Then, move the downloaded checkpoint `MedSAM2_pretrain.pth` to the folder `anaconda3/envs/labelme/Lib/site-packages/labelme/checkpoints`.  
If you can't find it, make good use of the file manager's search function.  
We will upload the fine-tuned multimodal segmentation model later.

### Set Icons
Download icons in our folder `MedSAM2/icons`, then place it into the folder`anaconda3/envs/labelme/Lib/site-packages/labelme/icons`.

### You need to change the path
Open `app.py`.
```bash
Update the path to the icons, which is around line 1904.
Update the path to the checkpoint file, which is around line 1958.  
Update the path to the global_data.json, which is around line 1981.
```
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
Your dataset must include both images with tumors and images without tumors (treated as background).   

### AFter First Round Training
After the first round of training, the model will look like this.  
Then, you can modify the dataset by adding some less obvious tumors and continue training using the fine-tuned MedSAM2.  

## Acknowledgement

This repo is the fork of [mpitid/pylabelme](https://github.com/mpitid/pylabelme).





