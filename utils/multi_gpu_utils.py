from tensorflow.python.client import device_lib


def count_gpus(input_num):
    if input_num is None:
        num_device = 0
        for device in device_lib.list_local_devices():
            if device.device_type == 'GPU':
                num_device += 1
    else:
        num_device = int(input_num)
    return num_device

