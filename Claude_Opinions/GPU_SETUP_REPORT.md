# Qiskit GPU Setup - Issues and Fixes Report

## Quantum Network Optimization Project

\---

## Executive Summary

Your code has **3 critical bugs** preventing proper GPU usage with qiskit-aer-gpu.
The main issue is a hardcoded 'GPU' device string that ignores the device parameter you're passing.

\---

## Issues Found

### Issue 1: Hardcoded GPU Device (CRITICAL)

**Location**: `quantum\_helpers.py`, line \~363 in `simulate\_split\_circuits()`

**Current Code**:

```python
simulator = AerSimulator(method=sim\_method, device='GPU')  # WRONG!
```

**Problem**:

* Device is hardcoded to 'GPU' instead of using the `device` parameter
* This means your code ALWAYS tries to use GPU, even when you set `device='CPU'`
* If GPU isn't available, this causes crashes

**Fix**:

```python
simulator = AerSimulator(method=sim\_method, device=device)
```

\---

### Issue 2: No GPU Availability Checking

**Location**: Throughout the codebase

**Problem**:

* No verification that GPU is actually available before attempting to use it
* No graceful fallback to CPU if GPU fails
* No informative error messages

**Fix**: Add helper function

```python
def get\_simulator(method='statevector', device='CPU'):
    """Safely create simulator with GPU fallback"""
    from qiskit\_aer import AerSimulator
    
    if device.upper() == 'GPU':
        try:
            simulator = AerSimulator(method=method, device='GPU')
            print("✓ GPU simulation enabled")
            return simulator
        except Exception as e:
            print(f"⚠ GPU unavailable: {e}")
            print("  Falling back to CPU")
            return AerSimulator(method=method, device='CPU')
    else:
        return AerSimulator(method=method, device='CPU')
```

\---

### Issue 3: Function Signature Inconsistency

**Location**: `quantum\_helpers.py`, `simulate\_split\_circuits()`

**Current Signature**:

```python
def simulate\_split\_circuits(sub\_circuits, shots=1024, sim\_method='statevector', device='CPU'):
```

**Problem**:

* Function accepts `device` parameter but doesn't use it properly (see Issue 1)
* Documentation says it accepts device but implementation ignores it

\---

## Required Changes

### Step 1: Add Helper Functions to quantum\_helpers.py

Add these two functions near the top of `quantum\_helpers.py` (after imports):

```python
def check\_gpu\_available():
    """
    Check if GPU is available for qiskit-aer simulation.
    
    Returns:
    --------
    bool: True if GPU is available, False otherwise
    """
    try:
        from qiskit\_aer import AerSimulator
        simulator = AerSimulator(method='statevector', device='GPU')
        return True
    except Exception as e:
        print(f"GPU check failed: {e}")
        return False


def get\_simulator(method='statevector', device='CPU'):
    """
    Safely create an AerSimulator with proper device selection.
    
    Parameters:
    -----------
    method : str
        Simulation method ('statevector' or 'density\_matrix')
    device : str
        Requested device ('CPU' or 'GPU')
    
    Returns:
    --------
    AerSimulator: Configured simulator
    """
    from qiskit\_aer import AerSimulator
    
    if device.upper() == 'GPU':
        try:
            simulator = AerSimulator(method=method, device='GPU')
            print("✓ Using GPU for simulation")
            return simulator
        except Exception as e:
            print(f"⚠ GPU requested but not available: {e}")
            print("  Falling back to CPU")
            return AerSimulator(method=method, device='CPU')
    else:
        return AerSimulator(method=method, device='CPU')
```

### Step 2: Fix simulate\_split\_circuits Function

In `simulate\_split\_circuits()` function (around line 363), change:

**FROM**:

```python
for sub\_circuit, qubit\_indices in sub\_circuits:
    simulator = AerSimulator(method=sim\_method, device='GPU')  # HARDCODED!
    transpiled\_circuit = transpile(sub\_circuit, simulator)
```

**TO**:

```python
for sub\_circuit, qubit\_indices in sub\_circuits:
    simulator = get\_simulator(method=sim\_method, device=device)  # NOW USES PARAMETER!
    transpiled\_circuit = transpile(sub\_circuit, simulator)
```

\---

## Verification Steps

### 1\. Check qiskit-aer-gpu Installation

In WSL2, run:

```bash
python3 -c "from qiskit\_aer import AerSimulator; print(AerSimulator.available\_devices())"
```

**Expected output if GPU works**:

```
\['CPU', 'GPU']
```

**If you only see `\['CPU']`**:
You need to install qiskit-aer-gpu:

```bash
pip uninstall qiskit-aer
pip install qiskit-aer-gpu
```

### 2\. Test CUDA in WSL2

```bash
nvidia-smi
```

This should show your GPU. If not, you need to:

1. Install NVIDIA drivers in Windows (not WSL)
2. Ensure WSL2 (not WSL1): `wsl --list --verbose`
3. Install CUDA toolkit in WSL2

### 3\. Test GPU in Python

Create a test script:

```python
from qiskit\_aer import AerSimulator

# Test GPU availability
print("Available devices:", AerSimulator.available\_devices())

# Try creating GPU simulator
try:
    sim\_gpu = AerSimulator(method='statevector', device='GPU')
    print("✓ GPU simulator created successfully")
except Exception as e:
    print(f"✗ GPU simulator failed: {e}")

# Try creating CPU simulator (should always work)
try:
    sim\_cpu = AerSimulator(method='statevector', device='CPU')
    print("✓ CPU simulator created successfully")
except Exception as e:
    print(f"✗ CPU simulator failed: {e}")
```

\---

## Installation Guide for qiskit-aer-gpu in WSL2

### Prerequisites

1. **Windows**: NVIDIA GPU drivers installed (latest)
2. **WSL2**: Must be WSL2, not WSL1

   * Check: `wsl --list --verbose`
   * Should show "VERSION 2"

### Install CUDA in WSL2

```bash
# Remove old CUDA if present
sudo apt remove --purge cuda\*
sudo apt autoremove
sudo apt autoclean

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86\_64/cuda-keyring\_1.1-1\_all.deb
sudo dpkg -i cuda-keyring\_1.1-1\_all.deb
sudo apt update

# Install CUDA toolkit
sudo apt install cuda-toolkit-12-6

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> \~/.bashrc
echo 'export LD\_LIBRARY\_PATH=/usr/local/cuda/lib64:$LD\_LIBRARY\_PATH' >> \~/.bashrc
source \~/.bashrc

# Verify
nvcc --version
nvidia-smi
```

### Install qiskit-aer-gpu

```bash
# Remove CPU-only version
pip uninstall qiskit-aer

# Install GPU version
pip install qiskit-aer-gpu

# Verify
python3 -c "from qiskit\_aer import AerSimulator; print(AerSimulator.available\_devices())"
```

\---

## Testing the Fixes

After applying all fixes, test with this modified section in `main.py`:

```python
# Test GPU availability first
from quantum\_helpers import check\_gpu\_available

if check\_gpu\_available():
    print("\\n✓ GPU is available for qiskit-aer")
    device = 'GPU'
else:
    print("\\n⚠ GPU not available, using CPU")
    device = 'CPU'

# Then run your QAOA with the detected device
graphs\_dict, runtime\_data, labelled\_tt\_data, qaoa\_progress = QAOA\_approx(
    graph, graphs\_dict, runtime\_data, 
    labelled\_tt\_data, qaoa\_progress, 
    shots=shots, 
    inv\_penalty\_m=inv\_penalty\_m,
    layers=layers,
    exploration\_strength=exploration\_strength,
    warm\_start=None,
    label='QAOA-Pretrained-L3-2Q',
    custom\_initial\_params=pretrained\_params,
    use\_soft\_validity=True,
    device=device  # This will now work correctly!
)
```

\---

## Summary of All Required Changes

### quantum\_helpers.py:

1. **Add** `check\_gpu\_available()` function (after imports)
2. **Add** `get\_simulator()` function (after imports)
3. **Fix** line \~363: Change `device='GPU'` to `device=device`
4. **Fix** line \~363: Change `AerSimulator(method=sim\_method, device='GPU')`
to `get\_simulator(method=sim\_method, device=device)`

### main.py (optional but recommended):

1. **Add** GPU availability check before running QAOA
2. **Use** the check result to set device variable

\---

## Expected Performance

With GPU properly configured:

* **Speedup**: 5-50x faster than CPU (depending on circuit size)
* **Memory**: Can handle larger circuits
* **Stability**: Graceful CPU fallback if GPU unavailable

Without these fixes:

* **Current state**: Code crashes if GPU unavailable
* **Confusion**: Device parameter is ignored

\---

## Additional Notes

### Why the Bug Happened

The function signature accepts `device` parameter but the implementation
hardcodes 'GPU'. This is a classic copy-paste error where someone added
the parameter but forgot to use it.

### Best Practice

Always verify device availability before use and provide fallback options.
The `get\_simulator()` function follows this pattern.

### Testing

After fixes, your code should:

1. ✓ Work on systems without GPU (CPU fallback)
2. ✓ Work on systems with GPU (uses GPU)
3. ✓ Inform user which device is being used
4. ✓ Not crash due to GPU unavailability

\---

## Questions to Investigate

1. **Do you have qiskit-aer or qiskit-aer-gpu installed?**

   * Run: `pip list | grep qiskit-aer`
2. **Does `nvidia-smi` work in WSL2?**

   * Run: `nvidia-smi` in WSL2 terminal
3. **What does this show?**

```python
   from qiskit\_aer import AerSimulator
   print(AerSimulator.available\_devices())
   ```

Send me the output of these three checks and I can provide more specific guidance!

