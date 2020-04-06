import cv2
import cvlib as cv
import numpy as np
from pylepton import Lepton

# TODO: As command line arguments.
FACE_CONFIDENCE = 0.5
USE_IMPERIAL = True


def format_temp(temp):
    # Raw data is Kelvin * 100.
    celsius = temp / 100 - 273.15
    if not USE_IMPERIAL:
        return "%.2f °C" % celsius
    fahrenheit = celsius * 9 / 5 + 32
    return "%.2f °F" % fahrenheit


with Lepton() as lepton:
    raw_buffer = np.ndarray((Lepton.ROWS, Lepton.COLS, 1), dtype=np.uint16)
    norm_buffer = np.ndarray((Lepton.ROWS, Lepton.COLS, 1), dtype=np.uint16)
    rgb_buffer = np.ndarray((Lepton.ROWS, Lepton.COLS, 3), dtype=np.uint8)

    while True:
        # Get the latest frame from the thermal camera.
        lepton.capture(data_buffer=raw_buffer)

        # Detect any faces in the frame.
        # TODO: Normalize to room/body temperature range instead.
        np.right_shift(raw_buffer, 8, out=norm_buffer)  # 16 bit -> 8 bit
        cv2.cvtColor(src=np.uint8(norm_buffer), dst=rgb_buffer,
                     code=cv2.COLOR_GRAY2RGB)  # Grayscale -> RGB
        cv2.normalize(src=rgb_buffer, dst=rgb_buffer, alpha=0, beta=255,
                      norm_type=cv2.NORM_MINMAX)  # Enhance contrast
        faces, _ = cv.detect_face(rgb_buffer, threshold=FACE_CONFIDENCE)

        # Find the (highest) temperature of each face.
        print("%d faces" % len(faces))
        for face in faces:
            left = face[0]
            top = face[1]
            right = face[2] + 1
            bottom = face[3] + 1
            crop = raw_buffer[left:right, top:bottom]
            temp = np.max(crop)
            print(format_temp(temp))
