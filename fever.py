from absl import app
from absl import flags
import cv2
import cvlib as cv
import numpy as np
from pylepton import Lepton

FLAGS = flags.FLAGS
flags.DEFINE_integer('min_temperature', 23715, 'The minimum temperature in'
                     ' centikelvin (for enhancing image contrast).')
flags.DEFINE_integer('max_temperature', 37315, 'The maximum temperature in'
                     ' centikelvin (for enhancing image contrast).')
flags.DEFINE_float('face_confidence', 0.5,
                   'The confidence threshold for face detection.')
flags.DEFINE_bool('display_metric', False, 'Whether to display metric units.')


def format_temperature(temperature):
    # The raw temperature is in centikelvin.
    celsius = temperature / 100 - 273.15
    if FLAGS.display_metric:
        return '%.f °C' % celsius
    fahrenheit = celsius * 9 / 5 + 32
    return '%.f °F' % fahrenheit


def main(_):
    with Lepton() as lepton:
        raw_buffer = np.ndarray((Lepton.ROWS, Lepton.COLS, 1), dtype=np.uint16)
        rgb_buffer = np.ndarray((Lepton.ROWS, Lepton.COLS, 3), dtype=np.uint8)
        scale_factor = (FLAGS.max_temperature - FLAGS.min_temperature) // 255

        while True:
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

            # Detect any faces in the frame.
            faces, _ = cv.detect_face(rgb_buffer,
                                      threshold=FLAGS.face_confidence)

            # Find the (highest) temperature of each face.
            print('%d faces' % len(faces))
            for face in faces:
                left = face[0]
                top = face[1]
                right = face[2] + 1
                bottom = face[3] + 1
                crop = raw_buffer[left:right, top:bottom]
                if crop.size == 0:
                    continue
                temperature = np.max(crop)
                print(format_temperature(temperature))


if __name__ == '__main__':
    app.run(main)
