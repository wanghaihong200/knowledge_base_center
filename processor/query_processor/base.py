"""检索流程节点基类（模板方法）：统一日志与任务进度追踪"""
from abc import ABC, abstractmethod

from tool.logger import logger
from utils.task_utils import add_running_task


class NodeBase(ABC):
    """
    检索侧节点基类（与导入侧 BaseNode 的差异：无 config 注入、异常不包装、用全局 logger）

    任务进度约定（笔记20）：
    - 基类统一在 process 前记录「运行中」；
    - 「已完成」由各节点在自身 process 结束前调用 add_done_task
      （并发分支节点拿不到统一收口，故不放在基类）。
    """

    name: str = "base_node"  # 子类必须覆盖

    def __call__(self, state: dict) -> dict:
        logger.info(f"--- {self.name} 开始 ---")
        add_running_task(
            state.get("session_id"), self.name, bool(state.get("is_stream"))
        )
        result = self.process(state)
        logger.info(f"--- {self.name} 完成 ---")
        return result

    @abstractmethod
    def process(self, state: dict) -> dict:
        """节点核心处理逻辑，子类必须实现"""
        pass

    def log_step(self, step_name: str, message: str = "") -> None:
        """记录阶段日志：[步骤] 附加信息"""
        msg = f"[{step_name}]"
        if message:
            msg += f" {message}"
        logger.info(f"[{self.name}] {msg}")
