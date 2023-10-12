import subprocess
import time
import numpy as np
import multiprocessing
import threading


class GPUStatistics:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.gpu_usage_data = []
        self.memory_usage_data = []
        self.temperature_gpu_data = []
        self.stop_flag = multiprocessing.Event()

    def _get_gpu_usage(self):
        try:
            nvidia_smi_output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu', '--format=csv,noheader,nounits', f'--id={self.gpu_id}'], universal_newlines=True)
            gpu_usage, memory_used, temperature_gpu= map(
                int, nvidia_smi_output.strip().split(', '))
            return gpu_usage, memory_used,temperature_gpu
        except Exception as e:
            print(f"错误: {e}")
            return 0, 0

    def record_gpu_usage(self, duration_ms):
        start_time = time.time() * 1000
        while time.time() * 1000 - start_time < duration_ms and not self.stop_flag.is_set():
            gpu_usage, memory_used, temperature_gpu = self._get_gpu_usage()
            if gpu_usage > 0 and memory_used > 0:
                self.gpu_usage_data.append(gpu_usage)
                self.memory_usage_data.append(memory_used)
                self.temperature_gpu_data.append(temperature_gpu)
                # print(f">>> GPU使用率: {gpu_usage:.2f}%\t显存使用: {memory_used} MiB")
            time.sleep(0.01)

        # 统计结束后立即输出结果
        self.print_statistics()

    def get_average_gpu_temperature(self):
        return np.mean(self.temperature_gpu_data)

    def get_max_gpu_temperature(self):
        return np.max(self.temperature_gpu_data)

    def get_average_gpu_usage(self):
        return np.mean(self.gpu_usage_data)

    def get_max_gpu_usage(self):
        return np.max(self.gpu_usage_data)

    def get_gpu_usage_variance(self):
        return np.var(self.gpu_usage_data)

    def get_median_gpu_usage(self):
        return np.median(self.gpu_usage_data)

    def get_average_memory_usage(self):
        return np.mean(self.memory_usage_data)

    def get_max_memory_usage(self):
        return np.max(self.memory_usage_data)

    def get_memory_usage_variance(self):
        return np.var(self.memory_usage_data)

    def get_median_memory_usage(self):
        return np.median(self.memory_usage_data)

    def reset_data(self):
        self.gpu_usage_data = []
        self.memory_usage_data = []
        self.temperature_gpu_data = []

    def stop(self):
        self.stop_flag.set()

    def print_statistics(self):
        print(f"================================================================\n")
        print(f"平均GPU温度: {self.get_average_gpu_temperature():.2f}%")
        print(f"最大GPU温度: {self.get_max_gpu_temperature():.2f}%")
        print(f"平均GPU使用率: {self.get_average_gpu_usage():.2f}%")
        print(f"最高GPU使用率: {self.get_max_gpu_usage():.2f}%")
        print(f"GPU使用率方差: {self.get_gpu_usage_variance():.2f}")
        print(f"中位数GPU使用率: {self.get_median_gpu_usage():.2f}%")
        print(f"平均显存使用: {self.get_average_memory_usage():.2f} MiB")
        print(f"最高显存使用: {self.get_max_memory_usage():.2f} MiB")
        print(f"显存使用方差: {self.get_memory_usage_variance():.2f} MiB^2")
        print(f"中位数显存使用: {self.get_median_memory_usage():.2f} MiB")

# if __name__ == "__main__":
#     gpu_stats = GPUStatistics(gpu_id=0)

#     def run_gpu_stats():
#         gpu_stats.record_gpu_usage(30000)

#     # 创建并启动子线程
#     my_thread = threading.Thread(target=run_gpu_stats)
#     my_thread.start()

#     # 启动新进程运行另一个Python文件
#     subprocess.run(['python', 'run_with_pytorch.py'])

#     # 停止GPU统计
#     gpu_stats.stop()
