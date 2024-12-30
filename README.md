## MedSAM2 

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
Then a red dot will appear at the location you click on the window.  
  
<img src="https://github.com/worldstar/MedSAM2-labelme/blob/main/examples/medsam2/medsam2-labelme-example.gif" width="100%" />

Click on the window , and press Enter after finishing.  
It will automatically save a JSON file, and the file will be named as `filename.json`.  
  
<img src="https://github.com/worldstar/MedSAM2-labelme/blob/main/examples/medsam2/screen%202024-12-23%20170840.png" width="100%" />

## How to develop

```bash
git clone https://github.com/labelmeai/labelme.git
cd labelme

# Install anaconda3 and labelme
curl -L https://github.com/wkentaro/dotfiles/raw/main/local/bin/install_anaconda3.sh | bash -s .
source .anaconda3/bin/activate
pip install -e .
```


### How to build standalone executable

Below shows how to build the standalone executable on macOS, Linux and Windows.  

```bash
# Setup conda
conda create --name labelme python=3.9
conda activate labelme

# Build the standalone executable
pip install .
pip install 'matplotlib<3.3'
pip install pyinstaller
pyinstaller labelme.spec
dist/labelme --version
```


### How to contribute

Make sure below test passes on your environment.  
See `.github/workflows/ci.yml` for more detail.

```bash
pip install -r requirements-dev.txt

ruff format --check  # `ruff format` to auto-fix
ruff check  # `ruff check --fix` to auto-fix
MPLBACKEND='agg' pytest -vsx tests/
```

## How to setup MedSAM2 semi-automatic

Move the files inside MedSAM2-click to anaconda3\envs\labelme\Lib\site-packages\labelme, and replace the old app.py with the new app.py .

You can download MedSAM2_pretrain checkpoint from checkpoints folder:
```bash
bash download_ckpts.sh
```

### After downloading the checkpoint
You need to open app.py, and update the path to the checkpoint file, which is around line 1950.  
It is recommended to use an absolute path to ensure no errors occur.

### Download Pytoch
We need to use [Pytorch](https://pytorch.org/) gpu version
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### How to start

Open your image, and click medsam2 button.  
Then a red dot will appear at the location you click on the window.  
  
<img src="https://github.com/worldstar/MedSAM2-labelme/blob/main/examples/medsam2/medsam2-labelme-example.gif" width="100%" />

Click on the window , and press Enter after finishing.  
It will automatically save a JSON file, and the file will be named as `filename.json`.  
  
<img src="https://github.com/worldstar/MedSAM2-labelme/blob/main/examples/medsam2/screen%202024-12-23%20170840.png" width="100%" />
  

## Acknowledgement

This repo is the fork of [mpitid/pylabelme](https://github.com/mpitid/pylabelme).





