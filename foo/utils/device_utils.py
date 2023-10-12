
import pynvml
import os

class DeviceProvider:
    def __init__(self, json_file):
        self.device_id_map = {}
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()  
        for i in range(num_gpus):
            map[str(i)] = []
            
    def get_valid_device_id(self):
        return 0
    