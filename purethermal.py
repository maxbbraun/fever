from ctypes import byref
from ctypes import cast
from ctypes import CFUNCTYPE
from ctypes import create_string_buffer
from ctypes import c_uint16
from ctypes import c_void_p
from ctypes import POINTER
from libuvc import LoadUvc
from libuvc import UvcContext
from libuvc import UvcDevice
from libuvc import UvcDeviceHandle
from libuvc import UvcFrame
from libuvc import UvcFormatDesc
from libuvc import UvcStreamCtrl
import numpy as np
from threading import Lock


# The USB Vendor and Product IDs used to find the UVC device.
USB_VENDOR_ID = 0x1e4e
USB_PRODUCT_ID = 0x0100

# The video stream format for 16-bit temperature values.
VIDEO_STREAM_FORMAT_GUID_Y16 = create_string_buffer(
    b'Y16 \x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71', 16)

# The frame format for 16-bit temperature values.
UVC_FRAME_FORMAT_Y16 = 13


# A simple thread-safe double buffer.
class FrameBuffer(object):
    def initialize(self, width, height, dtype):
        self._shape = (height, width)
        self._buffers = [np.zeros(self._shape, dtype=dtype),
                         np.zeros(self._shape, dtype=dtype)]
        self._write_index = 0
        self._read_index = 1
        self._lock = Lock()

    def write(self, data):
        source = data.reshape(self._shape)
        destination = self._buffers[self._write_index]
        np.copyto(dst=destination, src=source)
        self._swap_buffers()

    def read(self):
        assert self._lock.locked()
        return self._buffers[self._read_index]

    def read_lock(self):
        return self._lock

    def _swap_buffers(self):
        with self._lock:
            self._write_index, self._read_index = (self._read_index,
                                                   self._write_index)


# Dynamically load the UVC library.
libuvc = LoadUvc()

# Define the FrameBuffer instance outside the PureThermal class to avoid scope
# issues with C callbacks.
frame_buffer = FrameBuffer()


def uvc_frame_callback(function):
    # Turn the C callback signature into a Python decorator.
    return CFUNCTYPE(None, POINTER(UvcFrame), c_void_p)(function)


@uvc_frame_callback
def frame_callback(frame_ptr, user_ptr):
    frame = frame_ptr.contents
    width = frame.width
    height = frame.height

    # Write the data from the callback into the frame buffer.
    assert frame.data_bytes == 2 * width * height
    data_ptr = cast(frame.data, POINTER(c_uint16 * width * height))
    data = np.frombuffer(data_ptr.contents,
                         dtype=np.uint16).reshape(height, width)
    frame_buffer.write(data)


class PureThermal(object):
    def __enter__(self):
        self._uvc_context = POINTER(UvcContext)()
        self._uvc_device = POINTER(UvcDevice)()
        self._uvc_device_handle = POINTER(UvcDeviceHandle)()
        self._uvc_stream_ctrl = UvcStreamCtrl()

        uvc_error = libuvc.uvc_init(byref(self._uvc_context), 0)
        if uvc_error < 0:
            raise RuntimeError('Failed to initialize UVC context (error %d)'
                               % uvc_error)

        uvc_error = libuvc.uvc_find_device(self._uvc_context,
                                           byref(self._uvc_device),
                                           USB_VENDOR_ID, USB_PRODUCT_ID, 0)
        if uvc_error < 0:
            raise RuntimeError('Failed to find UVC device (error %d)'
                               % uvc_error)

        uvc_error = libuvc.uvc_open(self._uvc_device,
                                    byref(self._uvc_device_handle))
        if uvc_error < 0:
            raise RuntimeError('Failed to open UVC device (error %d)'
                               % uvc_error)

        frame_formats = self._frame_formats(VIDEO_STREAM_FORMAT_GUID_Y16)
        if not frame_formats:
            raise RuntimeError('Video stream format GUID Y16 not supported')

        frame_format = frame_formats[0]
        frame_format_fps = int(1e7 / frame_format.dwDefaultFrameInterval)
        uvc_error = libuvc.uvc_get_stream_ctrl_format_size(
            self._uvc_device_handle,
            byref(self._uvc_stream_ctrl),
            UVC_FRAME_FORMAT_Y16,
            frame_format.wWidth,
            frame_format.wHeight,
            frame_format_fps)
        if uvc_error < 0:
            raise RuntimeError('Failed to negotiate stream profile (error %d)'
                               % uvc_error)

        # Initialize the frame buffer, now that the size is known.
        self._frame_width = frame_format.wWidth
        self._frame_height = frame_format.wHeight
        frame_buffer.initialize(self._frame_width, self._frame_height,
                                np.uint16)

        uvc_error = libuvc.uvc_start_streaming(self._uvc_device_handle,
                                               byref(self._uvc_stream_ctrl),
                                               frame_callback, None, 0)
        if uvc_error < 0:
            raise RuntimeError('Failed to start streaming (error %d)'
                               % uvc_error)

        return self

    def __exit__(self, type, value, traceback):
        libuvc.uvc_stop_streaming(self._uvc_device_handle)
        libuvc.uvc_unref_device(self._uvc_device)
        libuvc.uvc_exit(self._uvc_context)

    def frame(self):
        return frame_buffer.read()

    def frame_lock(self):
        return frame_buffer.read_lock()

    def width(self):
        return self._frame_width

    def height(self):
        return self._frame_height

    def _as_iterator(self, pointer):
        while pointer:
            contents = pointer.contents
            yield contents
            pointer = contents.next

    def _frame_formats(self, video_stream_format_guid):
        frame_formats = []

        # Iterate over the available video stream formats.
        libuvc.uvc_get_format_descs.restype = POINTER(UvcFormatDesc)
        format_desc_ptr = libuvc.uvc_get_format_descs(self._uvc_device_handle)
        for format_desc in self._as_iterator(format_desc_ptr):
            # Ignore formats where the GUID doesn't match.
            if format_desc.guidFormat[0:4] != video_stream_format_guid[0:4]:
                continue

            # Iterate over the available frame formats.
            frame_desc_ptr = format_desc.frame_descs
            for frame_desc in self._as_iterator(frame_desc_ptr):
                frame_formats.append(frame_desc)

        return frame_formats
