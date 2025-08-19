# CAMASim Documentation

CAMASim is a comprehensive simulation framework for Content-Addressable Memory (CAM) based accelerators.

## Quick Start

```python
from CAMASim import CAMASim, CAMConfig

# Create configuration using preset
config = CAMConfig.preset("decision_tree").build()

# Initialize simulator
cam = CAMASim(config)

# Use the simulator
cam.write(data)
results, latency, energy = cam.query(query_data)
```

## Configuration

### Easy Configuration with CAMConfig

```python
# Using presets
config = CAMConfig.preset("decision_tree")  # or "image_search", "database", "small_test"

# Custom configuration with method chaining
config = (CAMConfig()
          .distance("hamming")
          .match_type("exact")
          .array_size(128, 128)
          .cell_type("ACAM")
          .device_type("RRAM")
          .variation("bitflip", flip_rate=0.05)
          .build())
```

### Configuration Options

**Distance Functions:**
- `"hamming"` - Hamming distance
- `"manhattan"` - L1/Manhattan distance  
- `"euclidean"` - L2/Euclidean distance
- `"innerproduct"` - Inner product
- `"rangequery"` - Range query (for ACAM)

**Match Types:**
- `"exact"` - Exact matching
- `"knn"` - K-nearest neighbors
- `"threshold"` - Threshold matching

**Cell Types:**
- `"ACAM"` - Analog CAM (fully supported)
- `"TCAM"` - Ternary CAM (basic support)
- `"MCAM"` - Multi-bit CAM (basic support)

**Device Types:**
- `"RRAM"` - Resistive RAM
- `"FeFET"` - Ferroelectric FET

**Noise Types:**
- `"gaussian"` - Gaussian noise
- `"bitflip"` - Bit-flip errors
- `"g-dependent"` - Conductance-dependent variation (RRAM only)

## Data Formats

### ACAM (Analog CAM)
- **Storage data**: 3D array `(samples, features, 2)` with `[min, max]` thresholds
- **Query data**: 2D array `(queries, features)` with single values

### TCAM/MCAM
- **Storage data**: 2D array `(samples, features)`
- **Query data**: 2D array `(queries, features)`

## Examples

See `/example/` folder for complete examples:
- `example_config_usage.py` - Configuration system examples
- `DecisionTree/example.py` - Decision tree inference on CAM

## Performance Evaluation

CAMASim provides detailed performance metrics:
- **Latency**: Array, peripheral, and interconnect latency
- **Energy**: Breakdown of energy consumption
- **Architecture**: Configurable hierarchy (subarrays, arrays, mats, banks)

## Save/Load Configuration

```python
# Save configuration
config.save("my_config.json")

# Load configuration
config = CAMConfig.load("my_config.json")
```