from absl import app
from absl import flags
from absl import logging
import bme680
import cv2
import cvlib as cv
import numpy as np
from pylepton import Lepton
from smbus2 import SMBus
from time import time

FLAGS = flags.FLAGS
flags.DEFINE_integer('min_temperature', 23715, 'The minimum temperature in '
                     'centikelvin (for enhancing image contrast).')
flags.DEFINE_integer('max_temperature', 37315, 'The maximum temperature in '
                     'centikelvin (for enhancing image contrast).')
flags.DEFINE_float('face_confidence', 0.5,
                   'The confidence threshold for face detection.')
flags.DEFINE_bool('display_metric', True, 'Whether to display metric units.')
flags.DEFINE_bool('detect', True, 'Whether to run face detection.')
flags.DEFINE_bool('visualize', False, 'Whether to visualize the thermal '
                  'image.')

TURBO_COLORMAP = np.array([
    [48, 18, 59], [50, 21, 67], [51, 24, 74], [52, 27, 81],
    [53, 30, 88], [54, 33, 95], [55, 36, 102], [56, 39, 109],
    [57, 42, 115], [58, 45, 121], [59, 47, 128], [60, 50, 134],
    [61, 53, 139], [62, 56, 145], [63, 59, 151], [63, 62, 156],
    [64, 64, 162], [65, 67, 167], [65, 70, 172], [66, 73, 177],
    [66, 75, 181], [67, 78, 186], [68, 81, 191], [68, 84, 195],
    [68, 86, 199], [69, 89, 203], [69, 92, 207], [69, 94, 211],
    [70, 97, 214], [70, 100, 218], [70, 102, 221], [70, 105, 224],
    [70, 107, 227], [71, 110, 230], [71, 113, 233], [71, 115, 235],
    [71, 118, 238], [71, 120, 240], [71, 123, 242], [70, 125, 244],
    [70, 128, 246], [70, 130, 248], [70, 133, 250], [70, 135, 251],
    [69, 138, 252], [69, 140, 253], [68, 143, 254], [67, 145, 254],
    [66, 148, 255], [65, 150, 255], [64, 153, 255], [62, 155, 254],
    [61, 158, 254], [59, 160, 253], [58, 163, 252], [56, 165, 251],
    [55, 168, 250], [53, 171, 248], [51, 173, 247], [49, 175, 245],
    [47, 178, 244], [46, 180, 242], [44, 183, 240], [42, 185, 238],
    [40, 188, 235], [39, 190, 233], [37, 192, 231], [35, 195, 228],
    [34, 197, 226], [32, 199, 223], [31, 201, 221], [30, 203, 218],
    [28, 205, 216], [27, 208, 213], [26, 210, 210], [26, 212, 208],
    [25, 213, 205], [24, 215, 202], [24, 217, 200], [24, 219, 197],
    [24, 221, 194], [24, 222, 192], [24, 224, 189], [25, 226, 187],
    [25, 227, 185], [26, 228, 182], [28, 230, 180], [29, 231, 178],
    [31, 233, 175], [32, 234, 172], [34, 235, 170], [37, 236, 167],
    [39, 238, 164], [42, 239, 161], [44, 240, 158], [47, 241, 155],
    [50, 242, 152], [53, 243, 148], [56, 244, 145], [60, 245, 142],
    [63, 246, 138], [67, 247, 135], [70, 248, 132], [74, 248, 128],
    [78, 249, 125], [82, 250, 122], [85, 250, 118], [89, 251, 115],
    [93, 252, 111], [97, 252, 108], [101, 253, 105], [105, 253, 102],
    [109, 254, 98], [113, 254, 95], [117, 254, 92], [121, 254, 89],
    [125, 255, 86], [128, 255, 83], [132, 255, 81], [136, 255, 78],
    [139, 255, 75], [143, 255, 73], [146, 255, 71], [150, 254, 68],
    [153, 254, 66], [156, 254, 64], [159, 253, 63], [161, 253, 61],
    [164, 252, 60], [167, 252, 58], [169, 251, 57], [172, 251, 56],
    [175, 250, 55], [177, 249, 54], [180, 248, 54], [183, 247, 53],
    [185, 246, 53], [188, 245, 52], [190, 244, 52], [193, 243, 52],
    [195, 241, 52], [198, 240, 52], [200, 239, 52], [203, 237, 52],
    [205, 236, 52], [208, 234, 52], [210, 233, 53], [212, 231, 53],
    [215, 229, 53], [217, 228, 54], [219, 226, 54], [221, 224, 55],
    [223, 223, 55], [225, 221, 55], [227, 219, 56], [229, 217, 56],
    [231, 215, 57], [233, 213, 57], [235, 211, 57], [236, 209, 58],
    [238, 207, 58], [239, 205, 58], [241, 203, 58], [242, 201, 58],
    [244, 199, 58], [245, 197, 58], [246, 195, 58], [247, 193, 58],
    [248, 190, 57], [249, 188, 57], [250, 186, 57], [251, 184, 56],
    [251, 182, 55], [252, 179, 54], [252, 177, 54], [253, 174, 53],
    [253, 172, 52], [254, 169, 51], [254, 167, 50], [254, 164, 49],
    [254, 161, 48], [254, 158, 47], [254, 155, 45], [254, 153, 44],
    [254, 150, 43], [254, 147, 42], [254, 144, 41], [253, 141, 39],
    [253, 138, 38], [252, 135, 37], [252, 132, 35], [251, 129, 34],
    [251, 126, 33], [250, 123, 31], [249, 120, 30], [249, 117, 29],
    [248, 114, 28], [247, 111, 26], [246, 108, 25], [245, 105, 24],
    [244, 102, 23], [243, 99, 21], [242, 96, 20], [241, 93, 19],
    [240, 91, 18], [239, 88, 17], [237, 85, 16], [236, 83, 15],
    [235, 80, 14], [234, 78, 13], [232, 75, 12], [231, 73, 12],
    [229, 71, 11], [228, 69, 10], [226, 67, 10], [225, 65, 9],
    [223, 63, 8], [221, 61, 8], [220, 59, 7], [218, 57, 7],
    [216, 55, 6], [214, 53, 6], [212, 51, 5], [210, 49, 5],
    [208, 47, 5], [206, 45, 4], [204, 43, 4], [202, 42, 4],
    [200, 40, 3], [197, 38, 3], [195, 37, 3], [193, 35, 2],
    [190, 33, 2], [188, 32, 2], [185, 30, 2], [183, 29, 2],
    [180, 27, 1], [178, 26, 1], [175, 24, 1], [172, 23, 1],
    [169, 22, 1], [167, 20, 1], [164, 19, 1], [161, 18, 1],
    [158, 16, 1], [155, 15, 1], [152, 14, 1], [149, 13, 1],
    [146, 11, 1], [142, 10, 1], [139, 9, 2], [136, 8, 2],
    [133, 7, 2], [129, 6, 2], [126, 5, 2], [122, 4, 3]], dtype=np.uint8)
WINDOW_NAME = 'window'
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
LINE_COLOR = (255, 255, 255)
LINE_THICKNESS = 2
LABEL_COLOR = LINE_COLOR
LABEL_FONT = cv2.FONT_HERSHEY_DUPLEX
LABEL_SCALE = 1
LABEL_THICKNESS = 2


def get_temperature(temperatures, face):
    # Consider the raw temperatures insides the face bounding box.
    crop = temperatures[face[1]:face[3], face[0]:face[2]]
    if crop.size == 0:
        return None

    # Use the maximum temperature across the face.
    return np.max(crop)


def format_temperature(temperature, add_unit=True):
    # The raw temperature is in centikelvin.
    celsius = temperature / 100 - 273.15
    if FLAGS.display_metric:
        if add_unit:
            return '%.f °C' % celsius
        else:
            return '%.f' % celsius
    else:
        fahrenheit = celsius * 9 / 5 + 32
        if add_unit:
            return '%.f °F' % fahrenheit
        else:
            return '%.f' % fahrenheit


def main(_):
    if FLAGS.detect:
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
    raw_buffer = np.zeros((Lepton.ROWS, Lepton.COLS, 1), dtype=np.int16)
    scaled_buffer = np.zeros((Lepton.ROWS, Lepton.COLS, 1), dtype=np.uint8)
    if FLAGS.detect:
        rgb_buffer = np.zeros((Lepton.ROWS, Lepton.COLS, 3), dtype=np.uint8)
    if FLAGS.visualize:
        window_buffer = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3),
                                 dtype=np.uint8)
    raw_scale_factor = (FLAGS.max_temperature - FLAGS.min_temperature) // 255
    window_scale_factor_x = WINDOW_WIDTH / Lepton.COLS
    window_scale_factor_y = WINDOW_HEIGHT / Lepton.ROWS

    if FLAGS.visualize:
        # Initialize the window.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    # Start the data processing loop.
    with Lepton() as lepton:
        while (not FLAGS.visualize or
               cv2.getWindowProperty(WINDOW_NAME, 0) != -1):
            try:
                start_time = time()

                if FLAGS.detect:
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

                # Map the raw temperature data to a normal range before
                # reducing the bit depth and min/max normalizing for better
                # contrast.
                np.clip((raw_buffer - FLAGS.min_temperature) //
                        raw_scale_factor, 0, 255, out=scaled_buffer)
                cv2.normalize(src=scaled_buffer, dst=scaled_buffer, alpha=0,
                              beta=255, norm_type=cv2.NORM_MINMAX)

                if FLAGS.detect:
                    # Convert to the expected RGB format.
                    cv2.cvtColor(src=scaled_buffer, dst=rgb_buffer,
                                 code=cv2.COLOR_GRAY2RGB)

                    # Detect any faces in the frame.
                    faces, _ = cv.detect_face(rgb_buffer,
                                              threshold=FLAGS.face_confidence)

                    # TODO: Estimate distance based on face size.

                    # TODO: Model thermal attenuation based on distance and
                    #       ambient temperature, pressure, and humidity.

                    # Find the (highest) temperature of each face.
                    if len(faces) == 1:
                        logging.info('1 person')
                    else:
                        logging.info('%d people' % len(faces))
                    for face in faces:
                        temperature = get_temperature(raw_buffer, face)
                        if not temperature:
                            logging.warning('Empty crop')
                            continue
                        logging.info(format_temperature(temperature))

                if FLAGS.visualize:
                    # Apply the colormap.
                    turbo_buffer = TURBO_COLORMAP[
                        scaled_buffer.reshape((Lepton.ROWS, Lepton.COLS))]

                    # Resize for the window.
                    cv2.cvtColor(src=turbo_buffer, dst=turbo_buffer,
                                 code=cv2.COLOR_RGB2BGR)
                    cv2.resize(src=turbo_buffer, dst=window_buffer,
                               dsize=(WINDOW_WIDTH, WINDOW_HEIGHT),
                               interpolation=cv2.INTER_CUBIC)

                    if FLAGS.detect:
                        # Draw the face bounding boxes and temperature.
                        for face in faces:
                            top_left = (
                                int(window_scale_factor_x * face[0]),
                                int(window_scale_factor_y * face[1]))
                            bottom_right = (
                                int(window_scale_factor_x * face[2]),
                                int(window_scale_factor_y * face[3]))
                            cv2.rectangle(window_buffer, top_left,
                                          bottom_right, LINE_COLOR,
                                          LINE_THICKNESS)

                            temperature = get_temperature(raw_buffer, face)
                            if not temperature:
                                continue
                            label = format_temperature(temperature,
                                                       add_unit=False)
                            label_size, _ = cv2.getTextSize(label, LABEL_FONT,
                                                            LABEL_SCALE,
                                                            LABEL_THICKNESS)
                            label_position = (
                                (top_left[0] + bottom_right[0]) // 2 -
                                label_size[0] // 2,
                                (top_left[1] + bottom_right[1]) // 2 +
                                label_size[1] // 2)
                            cv2.putText(window_buffer, label, label_position,
                                        LABEL_FONT, LABEL_SCALE, LABEL_COLOR,
                                        LABEL_THICKNESS, cv2.LINE_AA)

                    # Draw the frame.
                    cv2.imshow(WINDOW_NAME, window_buffer)
                    cv2.waitKey(1)

                # Calculate timing stats.
                duration = time() - start_time
                logging.debug('Frame took %.f ms (%.2f Hz)' % (
                    duration * 1000, 1 / duration))

            # Stop on SIGINT.
            except KeyboardInterrupt:
                break

    if FLAGS.visualize:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(main)
