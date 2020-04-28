# FEVER

A contactless fever thermometer with auto-aim. Combines a thermal camera with face detection.

⚠️ ***This is an incomplete prototype. It is not a medical device. See [issues](https://github.com/maxbbraun/fever/issues) for remaining work.***

## Parts

- [FLIR Radiometric Lepton Dev Kit](https://www.sparkfun.com/products/retired/14654) (thermal camera)
- [BME680 Breakout Board](https://www.sparkfun.com/products/15743) (ambient temperature, pressure, and humidity sensor)
- [Raspberry Pi 4 Model B](https://www.sparkfun.com/products/15447) (or similar model)

## Assembly

| ![breadboard front](breadboard-front.jpg) | ![breadboard back](breadboard-back.jpg) |
| - | - |

## Setup

Image [Raspbian](https://www.raspberrypi.org/downloads/raspbian/)

`sudo raspi-config`
- `Network Options > Wi-fi`
- `Boot Options > Desktop / CLI > Console Autologin`
- `Interfacing Options > SSH`
- `Interfacing Options > SPI`
- `Interfacing Options > I2C`

## Install

```bash
git clone https://github.com/maxbbraun/fever.git && cd fever
scp fever.py pi@192.168.x.x:/home/pi/
```

```bash
ssh pi@192.168.x.x

sudo apt-get update
sudo apt-get install -y python3-venv python3-opencv libatlas-base-dev libjasper-dev libhdf5-dev libqt4-dev git
python3 -m venv venv && . venv/bin/activate
pip3 install --no-cache-dir tensorflow
pip3 install opencv-contrib-python
pip3 install numpy absl-py cvlib
pip3 install smbus2 bme680
git clone https://github.com/groupgets/pylepton.git
cd pylepton && python setup.py install && cd ..
```

```bash
. venv/bin/activate
export LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1
```

## Run

```bash
python fever.py --verbosity=1

I0410 18:11:23.993699 1995587312 fever.py:55] Ambient temperature: 24 °C
I0410 18:11:23.994379 1995587312 fever.py:57] Ambient pressure: 1013 hPa
I0410 18:11:23.994970 1995587312 fever.py:59] Ambient humidity: 42 %
I0410 18:11:23.953286 1995587312 fever.py:87] 0 people
I0410 18:11:23.993699 1995587312 fever.py:55] Ambient temperature: 24 °C
I0410 18:11:23.994379 1995587312 fever.py:57] Ambient pressure: 1013 hPa
I0410 18:11:23.994970 1995587312 fever.py:59] Ambient humidity: 42 %
I0410 18:11:25.208623 1995587312 fever.py:85] 1 person
I0410 18:11:25.210044 1995587312 fever.py:94] 34 °C
...
```

## Visualize

| ![Visualize and detect](visualize-detect.png) | ![Visualize](visualize.png) |
| :-: | :-: |
| `python fever.py --visualize` | `python fever.py --visualize --nodetect` |

## Flags

```bash
python fever.py --help

       USAGE: fever.py [flags]
flags:

fever.py:
  --[no]detect: Whether to run face detection.
    (default: 'true')
  --[no]display_metric: Whether to display metric units.
    (default: 'true')
  --face_confidence: The confidence threshold for face detection.
    (default: '0.5')
    (a number)
  --max_temperature: The maximum temperature in centikelvin (for enhancing image contrast).
    (default: '37315')
    (an integer)
  --min_temperature: The minimum temperature in centikelvin (for enhancing image contrast).
    (default: '23715')
    (an integer)
  --[no]visualize: Whether to visualize the thermal image.
    (default: 'false')

Try --helpfull to get a list of all flags.
```
