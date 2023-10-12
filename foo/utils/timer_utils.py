import time

from DINet.utils.logger import custom_logger as LOG

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        LOG.info('[*] Time taken by [{}]: {:.4f} seconds'.format(func.__name__, end_time - start_time))
        return result
    return wrapper