import enum

# 管道读写模式
class PipeMode(enum.Enum):
    # 阻塞式读取
    ReadOnly_Block = 1
    # 阻塞式写入
    WriteOnly_Block = 2
    # 非阻塞式读取
    ReadOnly_NonBlock = 3
    # 非阻塞式写入
    WriteOnly_NonBlock = 4

# std异常
class StdException(Exception):
    pass