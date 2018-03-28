# DnC-RL-Tensorflow
A Tensorflow implementation of Divide-and-Conquer Reinforcement Learning.

## Usage

### Installation

1. Copy Mujoco and its license key to `~/.mujoco`
2. Add the following `PATH` into `~/.bashrc` or `~/.zshrc`

```bash
Add MUJOCO_PY_MJKEY_PATH=/home/lywoon/.mujoco/mjkey.txt
Add MUJOCO_PY_MJPRO_PATH=/home/lywoon/.mujoco/mjpro131
```

3. Make a virtual environment with Python 3.5.2 or 3.6.3 (do not use `anaconda` due to its compatibility with opencv and moviepy)
4. Install packages in the following `Dependencies` section


### Dependencies

```bash
# brew install open-mpi  # for mac OSX
$ sudo apt-get install python3-tk tmux

# install python packages
$ pip install tensorflow pillow moviepy gym mujoco-py==0.5.7 tqdm ipdb scipy opencv-python matplotlib mpi4py

# run imageio.plugins.ffmpeg.download() in python
$ python
>>> import imageio; imageio.plugins.ffmpeg.download()
```

### Virtual screen

```bash
# install packages
sudo apt-get install xserver-xorg libglu1-mesa-dev freeglut3-dev mesa-common-dev libxmu-dev libxi-dev
# configure nvidia-x
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
# run virtual screen
sudo /usr/bin/X :0
# run progrma with DISPLAY=:0
DISPLAY=:0 <program>
```
