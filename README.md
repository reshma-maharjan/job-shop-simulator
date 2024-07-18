# Job Shop Scheduling Algorithms

This project implements various algorithms for solving the Job Shop Scheduling Problem (JSSP), including Q-Learning and Actor-Critic methods. It provides both a C++ library and Python bindings for easy integration into different environments.

## Features

- Job Shop Environment simulation
- Q-Learning algorithm implementation
- Actor-Critic algorithm implementation
- Taillard problem instance generator
- Visualization of scheduling results
- Python bindings for easy integration

## Dependencies

This project uses vcpkg to manage C++ dependencies. The dependencies are specified in the `vcpkg.json` manifest file in the root directory of the project.

Before building the project, ensure you have the following installed:

### Build Tools
```bash
sudo apt-get install build-essential pkg-config cmake git curl zip unzip tar autoconf autoconf-archive libtool
```

### GLEW
```bash
sudo apt-get install libxmu-dev libxi-dev libgl-dev libglu1-mesa-dev
```

### GLFW3
```bash
sudo apt-get install libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

### Python3
```bash
sudo apt-get install python3 python3-dev
```

### Performance Tools (for WSL2)
```bash
sudo apt-get install linux-tools-common linux-tools-generic
sudo /usr/lib/linux-tools-6.8.0-36/perf
```

### vcpkg Setup

To install vcpkg:

```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
```

Set the `VCPKG_ROOT` environment variable to the vcpkg installation directory:

```bash
export VCPKG_ROOT=/path/to/vcpkg
```

## Building the Project

This project uses CMakePresets.json to manage build configurations. There are two main configurations: debug and release.

1. Clone the repository:
   ```bash
   git clone https://github.com/cair/job-shop-simulator.git
   cd job-shop-simulator
   ```

2. To configure and build the project in debug mode:
   ```bash
   cmake --preset=debug
   cmake --build --preset=debug
   ```

3. To configure and build the project in release mode:
   ```bash
   cmake --preset=release
   cmake --build --preset=release
   ```

## Installing the Python Package

To install the Python package, first ensure that the `VCPKG_ROOT` environment variable is set, then run:

```bash
pip install .
```

This will automatically use CMakePresets.json to build the project and install the Python package.

## Usage

### C++ Library

To use the C++ library in your project, include the necessary headers and link against the library:

```cpp
#include <jobshop/job_shop_environment.h>
#include <jobshop/job_shop_qlearning.h>

// Your code here
```

### Python Module

After installing the Python package, you can use it in your Python code:

```python
import jobshop

# Create a job shop environment
env = jobshop.JobShopEnvironment(jobs)

# Create a Q-Learning agent
agent = jobshop.JobShopQLearning(env, alpha=0.1, gamma=0.9, epsilon=0.3)

# Train the agent
agent.train(num_episodes=1000)

# Print the best schedule
agent.printBestSchedule()
```

## Project Structure

- `src/`: Contains the C++ source files
- `include/`: Contains the header files
- `bindings/`: Contains the Python bindings
- `CMakeLists.txt`: The main CMake configuration file
- `CMakePresets.json`: Defines the build presets
- `vcpkg.json`: Specifies the project dependencies for vcpkg

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

If you have any questions or feedback, please open an issue on the GitHub repository.