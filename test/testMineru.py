"""
执行 MinerU 命令
import_file_path: 解析的文件路径
file_dir_path: 解析后的文件目录
"""
import os
import subprocess
import time

# 1. 设置环境变量
env = os.environ.copy()
env["MINERU_MODEL_SOURCE"] = "local"

# 1. 构建命令行
cmd = "mineru -p abc.pdf -o output --backend pipeline"

process_start_time = time.time()

# 2. 执行命令行(子进程执行命令行)
proc = subprocess.Popen(
    args=cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    errors="replace",       # 遇到乱码时替换
    text=True,              # 输出的内容是字符串 不是字节
    encoding="utf-8",       # 用指定的中文字符集进行编解码
    bufsize=1               # 按行缓冲，只要缓冲区一行满了就输出
)

# 3. 获取日志信息
for line in proc.stdout:
    print(f"执行MinerU产生的日志：{line}")

# 4. 等待子进程做完
processed_code = proc.wait()

process_end_time = time.time()
if processed_code == 0:
    print(f"MinerU成功解析PDF文件，耗时:{process_end_time - process_start_time:.2f}s")
else:
    print("MinerU解析PDF文件失败")