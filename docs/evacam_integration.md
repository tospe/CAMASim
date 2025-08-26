# EVACAM Integration Guide

## Overview

CAMASim integrates with EVACAM (Eva-CAM) to provide circuit-level hardware cost analysis for Content Addressable Memory designs. EVACAM offers detailed modeling of area, timing, dynamic energy, and leakage power for NVM-based CAMs.

## What is EVACAM?

EVACAM is a circuit/architecture-level modeling and evaluation tool that can project the area, timing, dynamic energy, and leakage power of NVM-based CAMs. It supports:

- **NVM Technologies**: FeFET, RRAM, STT-MRAM, PCM
- **CAM Types**: Ternary CAM (TCAM), Analog CAM (ACAM), Multi-bit CAM (MCAM)
- **Match Types**: Exact match, threshold match, best match
- **Detailed Analysis**: Circuit-level performance metrics

## Enabling EVACAM in CAMASim

### Method 1: Using CAMConfig Builder

```python
from CAMASim.config import CAMConfig

# Enable EVACAM on any configuration
config = (CAMConfig.preset("decision_tree")
          .use_evacam_cost(True)
          .build())

# Or use the dedicated EVACAM preset
config = CAMConfig.preset("decision_tree_evacam").build()

# Custom configuration with EVACAM
config = (CAMConfig()
          .distance("hamming")
          .match_type("exact")
          .array_size(128, 64)
          .cell_type("TCAM")
          .device_type("FeFET")  # FeFET is currently supported
          .use_evacam_cost(True)
          .build())
```

### Method 2: Manual JSON Configuration

```json
{
  "query": {
    "distance": "hamming",
    "searchScheme": "exact",
    "bit": 3
  },
  "array": {
    "row": 128,
    "col": 128,
    "cell": "ACAM",
    "bit": 3,
    "useEVACAMCost": true
  },
  "cell": {
    "device": "FeFET"
  }
}
```

## Complete Example

Here's a complete example demonstrating EVACAM integration:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from CAMASim import CAMASim, CAMConfig
from DecisionTree.dt2cam import DT2Array

def run_evacam_example():
    """Complete example using EVACAM for hardware cost analysis."""
    
    # 1. Prepare data
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train decision tree
    clf = DecisionTreeClassifier(random_state=42, max_depth=10)
    clf.fit(X_train, y_train)
    
    # Convert to CAM format
    threshold_array, col2featureID, row2classID, threshold_min, threshold_max = DT2Array(clf)
    X_feature = X_test[:, col2featureID]
    query_array = np.clip(X_feature, threshold_min, threshold_max)
    
    print(f"CAM Array shape: {threshold_array.shape}")
    print(f"CAM Query shape: {query_array.shape}")
    
    # 2. Configuration with EVACAM enabled
    print("\n=== Running with EVACAM Cost Analysis ===")
    config_evacam = (CAMConfig.preset("decision_tree")
                     .device_type("FeFET")  # Use FeFET for EVACAM
                     .use_evacam_cost(True)
                     .build())
    
    # 3. Run CAMASim with EVACAM
    cam = CAMASim(config_evacam)
    
    # Write phase
    print("Writing data to CAM...")
    write_latency, write_energy = cam.write(threshold_array)
    print(f"Write - Latency: {write_latency:.2e} ns, Energy: {write_energy:.2e} J")
    
    # Query phase  
    print("Querying CAM...")
    query_results, query_latency, query_energy = cam.query(query_array)
    print(f"Query - Latency: {query_latency:.2e} ps, Energy: {query_energy:.2e} J")
    
    # 4. Display detailed EVACAM metrics
    print("\n=== EVACAM Detailed Analysis ===")
    print("Circuit-level metrics have been calculated by EVACAM:")
    print("- Total Area, Mat Area, Subarray Area")
    print("- Search Latency, RESET Latency")
    print("- Read/Write Dynamic Energy")
    print("- Leakage Power")
    print("\nSee EVACAM_cost.json for detailed breakdown")
    
    return query_results

if __name__ == "__main__":
    results = run_evacam_example()
```

## EVACAM Configuration Options

### Supported Device Types
- **FeFET**: Currently the primary supported device type
- **RRAM**: Support may vary depending on EVACAM version

### Configuration Parameters

The EVACAM integration automatically handles:
- **Capacity**: Calculated from array dimensions
- **Word Width**: Set from array column size
- **Process Node**: Default 22nm
- **Temperature**: Default 350K
- **Optimization Target**: Default ReadEDP

### Performance Metrics

EVACAM provides detailed circuit-level analysis:

```
Area:
 - Total Area = 77.603um x 158.121um = 12270.614um^2
 - Mat Area = 77.603um x 158.121um = 12270.614um^2 (81.115%)
 - Subarray Area = 83.443um x 144.987um = 12098.116um^2 (82.271%)

Timing:
 - Search Latency = 635.059ps
 - RESET Latency = 10.220ns

Power:
 - Read Dynamic Energy = 7.965pJ
 - Write Dynamic Energy = 6.336pJ  
 - Leakage Power = 2.390mW
```

## Troubleshooting

### Prerequisites
1. **EVACAM Submodule**: Ensure EVACAM submodule is initialized
   ```bash
   git submodule update --init --recursive
   ```

2. **Build EVACAM**: CAMASim automatically builds EVACAM when needed
   ```bash
   make -C CAMASim/performance/module/EVACAM
   ```

### Common Issues

**Issue**: `ERROR: EvaCAM submodule not loaded`
- **Solution**: Run `git submodule update --init --recursive`

**Issue**: `Failed to build EvaCAM`  
- **Solution**: Ensure g++ compiler is available and check compilation errors

**Issue**: `ERROR: EvaCAM returned non-zero value`
- **Solution**: Check EVACAM configuration parameters and device compatibility

**Issue**: Device type not supported
- **Solution**: Currently use `"FeFET"` as device type for best compatibility

### Fallback to Predefined Costs

If EVACAM fails, CAMASim automatically falls back to predefined costs:

```python
# Disable EVACAM, use predefined costs
config = (CAMConfig.preset("decision_tree")
          .use_evacam_cost(False)  # Explicitly disable
          .build())
```

## Advanced Usage

### Custom EVACAM Configuration Files

For advanced users, EVACAM configuration files can be customized in:
- `CAMASim/performance/module/EVACAM/2FeFET_TCAM.cfg`
- `CAMASim/performance/module/EVACAM/2FeFET_TCAM.cell`

### Comparing EVACAM vs Predefined Costs

```python
# Run with both approaches for comparison
configs = [
    ("Predefined", CAMConfig.preset("decision_tree").use_evacam_cost(False).build()),
    ("EVACAM", CAMConfig.preset("decision_tree").use_evacam_cost(True).build())
]

for name, config in configs:
    cam = CAMASim(config)
    write_lat, write_eng = cam.write(data)
    query_lat, query_eng = cam.query(queries)
    print(f"{name} - Write: {write_lat:.2e}ns, Query: {query_lat:.2e}ps")
```

## References

- [EVACAM Paper](https://ieeexplore.ieee.org/document/9474110): "Eva-CAM: A Circuit/Architecture-Level Evaluation Tool for General Content Addressable Memories", DATE 2022
- [EVACAM GitHub](https://github.com/tospe/EvaCAM): Updated EVACAM repository
- [CAMASim Paper](https://arxiv.org/abs/2403.03442): "CAMASim: A Comprehensive Simulation Framework for Content-Addressable Memory based Accelerators"