from abc import ABC, abstractmethod
from typing import List, Tuple
from ..environment.job_shop_environment import JobShopEnvironment, Action

class BaseScheduler(ABC):
    """Base class for job shop scheduling algorithms."""

    @abstractmethod
    def solve(self, env: JobShopEnvironment, max_steps: int = 1000) -> Tuple[List[Action], int]:
        """Solve the job shop scheduling problem."""
        pass
