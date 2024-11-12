# Performant Job Shop Scheduling (PER-JSP)

A high-performance Job Shop Scheduling Problem (JSSP) solver with C++ core and Python bindings. The project provides both a fast C++ library and intuitive Python interface for solving JSSP using various algorithms including Q-Learning and Actor-Critic methods.

## Features

- 🚀 High-performance C++ core with Python bindings
- 🐍 Pure Python fallback implementation
- 🔧 Flexible environment configuration
- 📊 Built-in visualization
- 📈 Support for standard benchmark problems (Taillard)
- 🧮 Multiple solver algorithms

### Implemented Algorithms
| Algorithm | Status | Implementation |
|-----------|:------:|----------------|
| Q-Learning | ✅ | C++/Python |
| Actor-Critic | ✅ | C++/Python |
| SARSA | ❌ | Planned |
| DQN | ❌ | Planned |
| PPO | ❌ | Planned |
| DDPG | ❌ | Planned |

### Environment Features
| Feature | Status | Notes |
|---------|:------:|-------|
| Jobs/Operations | ✅ | Full support |
| Taillard Benchmarks | ✅ | Built-in |
| Custom Environments | ✅ | JSON format |
| Machine Breakdowns | 🚧 | In progress |
| Tool Management | 🚧 | In progress |
| Priority Scheduling | 🚧 | Planned |

## Installation

There are two ways to install PER-JSP:

### 1. Python-Only Installation (Fast Install)
For users who only need the Python implementation without C++ optimizations:

```bash
PYTHON_ONLY=1 pip install .
```

This installation:
- ✅ No C++ compiler needed
- ✅ No system dependencies required
- ✅ Quick installation
- ❌ Lower performance compared to C++ version

### 2. Full Installation (With C++ Extensions)
For users who want maximum performance:

First, install system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libgl-dev \
    libglu1-mesa-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    python3-dev

# macOS
brew install cmake ninja pkg-config

# Windows (with Visual Studio installed)
# No additional dependencies needed
```

Then install the package:
```bash
pip install .
```

This installation:
- ✅ Maximum performance
- ✅ All features available
- ❓ Requires system dependencies
- ❓ Longer installation time

## Quick Start

```python
from per_jsp import Environment, QLearning

# Create environment
env = Environment.from_taillard(1)  # Load Taillard instance 1

# Create solver
solver = QLearning(
    env,
    learning_rate=0.1,
    discount_factor=0.9,
    exploration_rate=0.1
)

# Train
solver.train(episodes=1000)

# Get solution
schedule = solver.get_best_schedule()
schedule.visualize()
```

## Advanced Usage

### Custom Problem Instance

```python
from per_jsp import Environment

# Define your problem
problem = {
    "jobs": [
        {"operations": [
            {"machine": 0, "processing_time": 10},
            {"machine": 1, "processing_time": 20}
        ]},
        {"operations": [
            {"machine": 1, "processing_time": 15},
            {"machine": 0, "processing_time": 25}
        ]}
    ]
}

# Create environment
env = Environment.from_dict(problem)
```

### Using Different Solvers

```python
from per_jsp import Environment, ActorCritic

env = Environment.from_taillard(1)

# Actor-Critic solver
solver = ActorCritic(
    env,
    actor_lr=0.001,
    critic_lr=0.001,
    discount_factor=0.99
)

# Train with specific settings
solver.train(
    episodes=1000,
    max_steps=10000,
    verbose=True
)
```

## Performance Comparison

| Problem Size | Python-Only | With C++ | Speedup |
|-------------|-------------|----------|---------|
| 6x6         | 1.00x       | 8.45x    | 8.45x   |
| 10x10       | 1.00x       | 12.3x    | 12.3x   |
| 20x20       | 1.00x       | 15.7x    | 15.7x   |

## Contributing

Contributions are welcome! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/cair/per-jsp
cd per-jsp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{andersen2024perjsp,
  author = {Andersen, Per-Arne},
  title = {PER-JSP: A Performant Job Shop Scheduling Framework},
  year = {2024},
  url = {https://github.com/cair/per-jsp}
}
```

## Support

- 📖 [Documentation](https://github.com/cair/per-jsp/wiki)
- 🐛 [Issue Tracker](https://github.com/cair/per-jsp/issues)
- 💬 [Discussions](https://github.com/cair/per-jsp/discussions)