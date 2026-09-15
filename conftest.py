"""pytest 根配置：确保项目根目录在 sys.path 中，使测试可导入项目包"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
