# GPU Acceleration for QAOA on Windows

## Your Setup
- **GPU**: NVIDIA RTX 5060 Ti (16GB VRAM)
- **OS**: Windows
- **Problem**: `qiskit-aer-gpu` is Linux-only
- **Goal**: Increase batch size, simulate more qubits simultaneously

---

## Option 1: WSL2 (Windows Subsystem for Linux) - RECOMMENDED

This is the **best option** - you get full Linux GPU support on Windows.

### Setup Steps

#### 1. Enable WSL2
```powershell
# Run in PowerShell as Administrator
wsl --install
# Restart your computer
```

#### 2. Install Ubuntu in WSL2
```powershell
# After restart
wsl --install -d Ubuntu-22.04
# Set up username and password when prompted
```

#### 3. Install NVIDIA Drivers for WSL2
- Download from: https://developer.nvidia.com/cuda/wsl
- Install the Windows driver (it automatically enables WSL2 GPU support)
- **Important**: Install the **Windows** driver, NOT a Linux driver

#### 4. Verify GPU Access in WSL2
```bash
# Inside WSL2 Ubuntu terminal
nvidia-smi
# Should show your RTX 5060 Ti!
```

#### 5. Install CUDA Toolkit in WSL2
```bash
# Update package list
sudo apt update
sudo apt upgrade -y

# Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 6. Install Python Environment in WSL2
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, restart terminal

# Create environment
conda create -n qaoa python=3.10
conda activate qaoa

# Install dependencies
pip install numpy scipy matplotlib networkx qiskit qiskit-aer-gpu
```

#### 7. Access Your Windows Files
```bash
# Your Windows C: drive is at /mnt/c/
cd /mnt/c/Users/joeco/OneDrive/Documents/projects/quantum_network_optimization/

# Run your code
python main.py
```

### Advantages
✅ Full GPU support (CUDA, cuQuantum)
✅ Access Windows files easily
✅ Best performance
✅ Can use qiskit-aer-gpu directly
✅ Easy file sharing between Windows and Linux

### Disadvantages
⚠️ Initial setup time (~30-60 minutes)
⚠️ Need to work in terminal (but can use VS Code WSL extension)

---

## Option 2: Docker with GPU Support - ALTERNATIVE

If you prefer containerized environments.

### Setup Steps

#### 1. Install Docker Desktop for Windows
- Download from: https://www.docker.com/products/docker-desktop/
- Enable WSL2 backend during installation
- Enable GPU support in settings

#### 2. Install NVIDIA Container Toolkit
```powershell
# This happens automatically with newer Docker Desktop versions
# Verify with:
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```

#### 3. Create Dockerfile
```dockerfile
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip git

# Install Qiskit with GPU support
RUN pip3 install numpy scipy matplotlib networkx qiskit qiskit-aer-gpu

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
```

#### 4. Build and Run
```powershell
# Build image
docker build -t qaoa-gpu .

# Run with GPU and mount your project folder
docker run --gpus all -v C:\Users\joeco\OneDrive\Documents\projects\quantum_network_optimization:/workspace -it qaoa-gpu

# Inside container
python3 main.py
```

### Advantages
✅ Isolated environment
✅ Full GPU support
✅ Easy to reproduce

### Disadvantages
⚠️ Requires Docker knowledge
⚠️ File permissions can be tricky
⚠️ Slightly more overhead

---

## Option 3: Dual Boot / Dedicated Linux - OVERKILL

Install Ubuntu alongside Windows. This is overkill for your use case.

---

## Option 4: Google Colab with GPU - QUICK TEST

For quick testing without local setup.

### Steps

1. Go to https://colab.research.google.com
2. Create new notebook
3. Runtime → Change runtime type → GPU (T4)
4. Install dependencies:
```python
!pip install qiskit qiskit-aer-gpu
```

### Advantages
✅ Zero setup
✅ Free GPU access
✅ Good for testing

### Disadvantages
⚠️ Limited to T4 GPU (16GB like yours, but shared)
⚠️ Session timeouts
⚠️ Need to upload files
⚠️ Not for production

---

## Recommended Approach: WSL2

I **strongly recommend WSL2** because:

1. **Full Windows integration** - Access your files at `/mnt/c/`
2. **Full GPU support** - CUDA, cuQuantum, everything works
3. **VS Code integration** - Install "Remote - WSL" extension in VS Code
4. **Low overhead** - Near-native performance
5. **One-time setup** - Then works forever

### After WSL2 Setup

Your workflow becomes:
```bash
# Open WSL2 terminal
wsl

# Navigate to project
cd /mnt/c/Users/joeco/OneDrive/Documents/projects/quantum_network_optimization/

# Activate environment
conda activate qaoa

# Run code (now using GPU!)
python main.py
```

Or even better, use VS Code:
1. Install "Remote - WSL" extension
2. Open folder in WSL2
3. Edit and run code as normal, but GPU-accelerated!

---

## Modifying Your Code for GPU

Once you have qiskit-aer-gpu installed (in WSL2 or Docker), modify your code:

### Update `quantum_helpers.py`

```python
# At the top of the file
from qiskit_aer import AerSimulator

def simulate_split_circuits(sub_circuits, shots=1024, sim_method='statevector', use_gpu=False):
    """
    Simulate split sub-circuits and combine results into full bitstrings.
    
    Parameters:
    -----------
    sub_circuits : list
        List of (sub_circuit, qubit_indices) tuples from split_circuit_for_simulation
    shots : int, optional
        Number of shots for simulation
    use_gpu : bool, optional
        If True, use GPU acceleration. Requires qiskit-aer-gpu package. Default: False.
    
    Returns:
    --------
    dict: Dictionary mapping full bitstrings to counts
    """
    
    # Simulate each sub-circuit
    sub_results = []
    
    for sub_circuit, qubit_indices in sub_circuits:
        if use_gpu:
            # GPU-accelerated simulation
            simulator = AerSimulator(method=sim_method, device='GPU')
        else:
            # CPU simulation (default)
            simulator = AerSimulator(method=sim_method)
        
        transpiled_circuit = transpile(sub_circuit, simulator)
        
        job = simulator.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        sub_results.append((counts, qubit_indices))
    
    # Rest of function unchanged...
```

### Update `quantum_engines.py`

```python
def QAOA_approx(..., use_gpu=False):
    '''
    ...
    use_gpu : bool
        If True, use GPU acceleration for circuit simulation.
        Requires qiskit-aer-gpu package (Linux/WSL2 only). Default: False.
    '''
    
    # In run_QAOA function:
    def run_QAOA(..., use_gpu=False):
        counts = simulate_large_circuit_in_batches(
            bound_circuit, batch_size, shots, sim_method, use_gpu=use_gpu
        )
```

---

## Expected Performance Gains

### Current Setup (CPU)
- Batch size: 8 qubits
- Simulation time: ~X seconds per iteration
- Memory limited by RAM

### With RTX 5060 Ti (16GB VRAM)
- **Batch size: 20-25 qubits** (huge increase!)
- **Simulation time: ~X/10 seconds** (10x speedup estimate)
- Memory limited by VRAM (16GB is excellent)

### Why This Matters

**Current**: 
- 90 qubits → 12 batches of 8 → 12 simulations per iteration
- Each simulation: 2^8 = 256 dimensional state vector

**With GPU**:
- 90 qubits → 4 batches of 22-23 → 4 simulations per iteration
- Each simulation: 2^22 = 4.2M dimensional state vector
- **3x fewer batches = 3x faster!**

---

## Detailed WSL2 Setup Guide

Since WSL2 is the best option, here's a complete step-by-step:

### Phase 1: Install WSL2 (5 minutes)

```powershell
# In PowerShell (Admin)
wsl --install
# Restart computer
```

### Phase 2: Setup Ubuntu (10 minutes)

After restart:
```powershell
# Check WSL version
wsl -l -v
# Should show VERSION 2

# If not, convert:
wsl --set-version Ubuntu-22.04 2
```

Launch Ubuntu from Start Menu, create username/password.

### Phase 3: GPU Drivers (15 minutes)

1. Download Windows NVIDIA driver: https://developer.nvidia.com/cuda/wsl
2. Install it (on Windows, not in WSL!)
3. Verify in WSL:
```bash
nvidia-smi
# Should work!
```

### Phase 4: CUDA Toolkit (20 minutes)

```bash
# In WSL2 Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3
```

### Phase 5: Python Environment (10 minutes)

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Press Enter, yes, yes, restart terminal

# Create environment
conda create -n qaoa python=3.10 -y
conda activate qaoa

# Install packages
pip install numpy scipy matplotlib networkx pandas
pip install qiskit qiskit-aer-gpu
```

### Phase 6: Test GPU (2 minutes)

```python
# test_gpu.py
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Create simple circuit
qc = QuantumCircuit(15)
qc.h(range(15))
qc.measure_all()

# Test GPU
simulator = AerSimulator(method='statevector', device='GPU')
job = simulator.run(qc, shots=1000)
result = job.result()
print("GPU simulation successful!")
print(f"Result: {result.get_counts()}")
```

```bash
python test_gpu.py
# Should work!
```

### Phase 7: Run Your Project (1 minute)

```bash
cd /mnt/c/Users/joeco/OneDrive/Documents/projects/quantum_network_optimization/
python main.py  # Now GPU-accelerated!
```

---

## VS Code Integration (Bonus)

Make development seamless:

1. Install "Remote - WSL" extension in VS Code
2. Click green button (bottom left) → "New WSL Window"
3. Open your project folder
4. Edit files normally, they're saved to Windows
5. Run with GPU acceleration automatically!

---

## Troubleshooting

### "CUDA not found"
```bash
echo $PATH  # Should include /usr/local/cuda/bin
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### "nvidia-smi not working in WSL"
- Update Windows NVIDIA driver
- Make sure WSL2 (not WSL1): `wsl -l -v`

### "qiskit-aer-gpu import error"
```bash
pip uninstall qiskit-aer qiskit-aer-gpu
pip install qiskit-aer-gpu
```

### "Permission denied on /mnt/c"
```bash
# Files are Windows files, access them normally
# No special permissions needed
```

---

## Cost-Benefit Analysis

| Approach | Setup Time | GPU Access | Ease of Use | Recommended |
|----------|-----------|------------|-------------|-------------|
| **WSL2** | 60 min | ✅ Full | ⭐⭐⭐⭐⭐ | **YES** |
| Docker | 45 min | ✅ Full | ⭐⭐⭐ | Maybe |
| Dual Boot | 2-3 hours | ✅ Full | ⭐⭐ | No |
| Colab | 0 min | ⚠️ Shared | ⭐⭐⭐ | Testing only |
| Stay on CPU | 0 min | ❌ None | ⭐⭐⭐⭐⭐ | No (you have GPU!) |

---

## My Recommendation

**Go with WSL2**:
1. One-hour setup
2. Full GPU access
3. Easy Windows integration
4. Works with VS Code
5. 3-10x performance improvement
6. Bigger batch sizes (8 → 20+ qubits)

With your RTX 5060 Ti (16GB), you'll be able to:
- Simulate 20-25 qubit batches (vs current 8)
- 10x speedup on simulations
- Run more experiments faster
- Higher fidelity pretraining

---

## Next Steps

1. **Install WSL2** (tonight, 5 minutes)
2. **Install Ubuntu 22.04** (tomorrow, 10 minutes)
3. **Setup CUDA** (weekend, 45 minutes)
4. **Test GPU** (weekend, 5 minutes)
5. **Run your code** (profit!)

Want help with any specific step? I can provide more detailed commands or troubleshooting!
