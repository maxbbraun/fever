from absl import app
from absl import flags
from absl import logging
import bme680
import cv2
from imutils import resize
from mrcnn.config import Config
from mrcnn import model as modellib
import numpy as np
from pylepton import Lepton
from smbus2 import SMBus
from time import time

FLAGS = flags.FLAGS
flags.DEFINE_integer('min_temperature', 23715, 'The minimum temperature in'
                     ' centikelvin (for enhancing image contrast).')
flags.DEFINE_integer('max_temperature', 37315, 'The maximum temperature in'
                     ' centikelvin (for enhancing image contrast).')
flags.DEFINE_float('face_confidence', 0.5,
                   'The confidence threshold for face detection.')
flags.DEFINE_bool('display_metric', True, 'Whether to display metric units.')

COCO_PERSON_CLASS_ID = 1


class MrcnnConfig(Config):
    NAME = 'coco_inference'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81

def format_temperature(temperature):
    # The raw temperature is in centikelvin.
    celsius = temperature / 100 - 273.15
    if FLAGS.display_metric:
        return '%.f °C' % celsius
    else:
        fahrenheit = celsius * 9 / 5 + 32
        return '%.f °F' % fahrenheit


def main(_):
    # Initialize ambient sensors.
    ambient = bme680.BME680(i2c_addr=bme680.I2C_ADDR_PRIMARY,
                            i2c_device=SMBus(1))
    # TODO: Tune settings.
    ambient.set_humidity_oversample(bme680.OS_2X)
    ambient.set_pressure_oversample(bme680.OS_4X)
    ambient.set_temperature_oversample(bme680.OS_8X)
    ambient.set_filter(bme680.FILTER_SIZE_3)
    ambient.set_gas_status(bme680.DISABLE_GAS_MEAS)

    # Initialize thermal image buffers.
    raw_buffer = np.ndarray((Lepton.ROWS, Lepton.COLS, 1), dtype=np.uint16)
    rgb_buffer = np.ndarray((Lepton.ROWS, Lepton.COLS, 3), dtype=np.uint8)
    scale_factor = (FLAGS.max_temperature - FLAGS.min_temperature) // 255

    # TODO
    mrcnn_config = MrcnnConfig()
    model = modellib.MaskRCNN(mode='inference', config=mrcnn_config,
                              model_dir='/tmp/')
    model.load_weights('mask_rcnn_coco.h5', by_name=True)

    # Start the data processing loop.
    with Lepton() as lepton:
        while True:
            start_time = time()

            # Acquire ambient sensor readings.
            if not ambient.get_sensor_data():
                logging.warning('Ambient sensor data not ready')
            ambient_data = ambient.data
            logging.debug('Ambient temperature: %.f °C'
                          % ambient_data.temperature)
            logging.debug('Ambient pressure: %.f hPa'
                          % ambient_data.pressure)
            logging.debug('Ambient humidity: %.f %%'
                          % ambient_data.humidity)

            # Get the latest frame from the thermal camera.
            lepton.capture(data_buffer=raw_buffer)

            # Prepare the raw temperature data for face detection: Map to a
            # normal range before reducing the bit depth and min/max normalize
            # for better contrast before converting to RGB.
            scaled_buffer = np.uint8((raw_buffer - FLAGS.min_temperature)
                                     // scale_factor)
            cv2.normalize(src=scaled_buffer, dst=scaled_buffer, alpha=0,
                          beta=255, norm_type=cv2.NORM_MINMAX)
            cv2.cvtColor(src=scaled_buffer, dst=rgb_buffer,
                         code=cv2.COLOR_GRAY2RGB)

            # TODO
            resized = resize(rgb_buffer, width=512) # TODO
            detections = model.detect([resized], verbose=1)
            logging.info(detections)

            # Detect any faces in the frame.
            # faces, _ = cv.detect_face(rgb_buffer,
            #                           threshold=FLAGS.face_confidence)

            # TODO: Estimate distance based on face size.

            # TODO: Model thermal attenuation based on distance and ambient
            #       temperature, pressure, and humidity.

            # Find the (highest) temperature of each face.
            # if len(faces) == 1:
            #     logging.info('1 person')
            # else:
            #     logging.info('%d people' % len(faces))
            # for face in faces:
            #     crop = raw_buffer[face[0]:face[2], face[1]:face[3]]
            #     if crop.size == 0:
            #         logging.warning('Empty crop')
            #         continue
            #     temperature = np.max(crop)
            #     logging.info(format_temperature(temperature))

            # Calculate timing stats.
            duration = time() - start_time
            logging.debug('Frame took %.f ms (%.2f Hz)' % (
                duration * 1000, 1 / duration))


if __name__ == '__main__':
    app.run(main)
