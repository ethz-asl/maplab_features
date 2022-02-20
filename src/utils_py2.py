import os
import errno
import numpy as np

def open_fifo(file_name, mode):
    try:
        os.mkfifo(file_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return open(file_name, mode)

def read_bytes(file, num_bytes):
    bytes = b''
    num_read = 0
    while num_read < num_bytes:
        bytes += file.read(num_bytes - num_read)
        num_read = len(bytes)
    return bytes

def read_np(file, dtype):
    header = read_bytes(file, 4*3)
    num_bytes, rows, cols = np.frombuffer(header, dtype=np.uint32)
    bytes = read_bytes(file, num_bytes)
    arr = np.frombuffer(bytes, dtype=dtype)

    if cols != 0:
        arr = arr.reshape((rows, cols))

    return arr

def send_np(file, arr):
    if arr.ndim == 1:
        rows, cols = arr.size, 0
    elif arr.ndim == 2:
        rows, cols = arr.shape[0], arr.shape[1]
        arr = arr.flatten()
    else:
        raise ValueError("Sending more than 2d numpy arrays not implemented")

    bytes = arr.tobytes()
    header = np.array([len(bytes), rows, cols], dtype=np.uint32).tobytes()
    file.write(header)
    file.write(bytes)
    file.flush()
