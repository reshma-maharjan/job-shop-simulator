import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationParams:
    """Parameters for generating job shop problems."""
    num_jobs: int
    """Number of jobs to generate"""

    num_machines: int
    """Number of machines available"""

    min_duration: int
    """Minimum duration for operations"""

    max_duration: int
    """Maximum duration for operations"""

    dependency_density: float
    """Probability of creating dependencies between jobs (0.0 to 1.0)"""

    max_dependencies_per_job: int
    """Maximum number of dependencies a job can have"""

    long_job_rate: float = 0.1
    """Probability of generating a long job"""

    long_job_factor: float = 3.0
    """Multiplier for duration of long jobs"""

    output_file: Optional[str] = None
    """Optional file path to save the generated instance"""

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.num_jobs <= 0:
            raise ValueError("Number of jobs must be positive")

        if self.num_machines <= 0:
            raise ValueError("Number of machines must be positive")

        if self.min_duration <= 0:
            raise ValueError("Minimum duration must be positive")

        if self.max_duration < self.min_duration:
            raise ValueError("Maximum duration must be greater than or equal to minimum duration")

        if not 0.0 <= self.dependency_density <= 1.0:
            raise ValueError("Dependency density must be between 0.0 and 1.0")

        if self.max_dependencies_per_job < 0:
            raise ValueError("Maximum dependencies per job must be non-negative")

        if not 0.0 <= self.long_job_rate <= 1.0:
            raise ValueError("Long job rate must be between 0.0 and 1.0")

        if self.long_job_factor < 1.0:
            raise ValueError("Long job factor must be greater than or equal to 1.0")

    @classmethod
    def from_json(cls, json_file: str) -> 'GenerationParams':
        """Create GenerationParams from a JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)
            return cls(**data)

    def to_json(self, json_file: str):
        """Save GenerationParams to a JSON file."""
        with open(json_file, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def default(cls) -> 'GenerationParams':
        """Create GenerationParams with default values."""
        return cls(
            num_jobs=10,
            num_machines=5,
            min_duration=1,
            max_duration=100,
            dependency_density=0.3,
            max_dependencies_per_job=3,
            long_job_rate=0.1,
            long_job_factor=2.0
        )

    @classmethod
    def simple(cls) -> 'GenerationParams':
        """Create GenerationParams for a simple problem."""
        return cls(
            num_jobs=5,
            num_machines=3,
            min_duration=1,
            max_duration=10,
            dependency_density=0.2,
            max_dependencies_per_job=2,
            long_job_rate=0.0,
            long_job_factor=1.0
        )

    @classmethod
    def complex(cls) -> 'GenerationParams':
        """Create GenerationParams for a complex problem."""
        return cls(
            num_jobs=20,
            num_machines=10,
            min_duration=5,
            max_duration=200,
            dependency_density=0.4,
            max_dependencies_per_job=5,
            long_job_rate=0.15,
            long_job_factor=4.0
        )
