# FEVER

```
python3 -m venv venv && . venv/bin/activate
# sudo apt-get install python3-opencv python3-numpy
sudo apt-get install libatlas-base-dev libjasper-dev
pip3 install --no-cache-dir tensorflow
pip3 install opencv-contrib-python
export LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1
pip3 install numpy cvlib
# pip3 install pylepton
git clone https://github.com/groupgets/pylepton.git && cd pylepton
python fever.py
```
