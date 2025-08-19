# CAMASim Configuration Guide

## Overview

CAMASim uses the `CAMConfig` class for easy, validated configuration. This replaces the complex nested JSON approach with a fluent, builder-pattern API.

## Basic Usage

### Presets

```python
from CAMASim import CAMConfig

# Use a preset as starting point
config = CAMConfig.preset("decision_tree")      # For ML applications
config = CAMConfig.preset("image_search")       # For image processing
config = CAMConfig.preset("database")           # For database operations  
config = CAMConfig.preset("small_test")         # For testing

# Build the final configuration
final_config = config.build()
```

### Method Chaining

```python
config = (CAMConfig()
          .distance("hamming")              # Set distance function
          .match_type("exact")              # Set matching type
          .array_size(256, 128)             # Set array dimensions
          .cell_type("TCAM")                # Set cell type
          .device_type("RRAM")              # Set device type
          .variation("bitflip", flip_rate=0.1)  # Add noise
          .architecture(4, 4, 2)            # Set hierarchy
          .build())
```

## Configuration Parameters

### Functional Configuration
- **Distance**: `distance("hamming" | "manhattan" | "euclidean" | "innerproduct" | "rangequery")`
- **Match Type**: `match_type("exact" | "knn" | "threshold")`
- **Match Parameter**: `match_parameter(int)` - for knn or threshold
- **Data Bits**: `data_bits(int)` - precision level

### Architecture Configuration
- **Array Size**: `array_size(rows, cols)`
- **Architecture**: `architecture(subarrays, arrays, mats)` - hierarchy levels
- **Merge Type**: `merge_type("voting" | "and" | "comparator" | "gather")`

### Circuit Configuration  
- **Cell Type**: `cell_type("ACAM" | "TCAM" | "MCAM")`
- **Sensing Circuit**: `sensing_circuit("exact" | "best" | "threshold")`

### Device Configuration
- **Device Type**: `device_type("RRAM" | "FeFET")`
- **Variation/Noise**: 
  - `variation("gaussian", std_dev=0.1)`
  - `variation("bitflip", flip_rate=0.05)` 
  - `variation("g-dependent")` (RRAM only)
- **No Noise**: `no_variation()`
- **Quantization**: `quantization(enabled=True, bits=8)`

### Performance Options
- **Function Simulation**: `enable_function_simulation(True/False)`
- **Performance Evaluation**: `enable_performance_evaluation(True/False)`

## Validation

CAMConfig provides comprehensive validation:

```python
try:
    config = CAMConfig().distance("invalid_distance")
except CAMConfigError as e:
    print(e)  # "Invalid distance function 'invalid_distance'. Valid options: [...]"
```

## Save/Load

```python
# Save configuration
config.save("my_config.json")

# Load configuration  
loaded_config = CAMConfig.load("my_config.json")

# Copy and modify
new_config = CAMConfig.load("my_config.json").variation("bitflip", flip_rate=0.2)
```

## Cell Type Considerations

### ACAM (Analog CAM)
- **Best for**: Range queries, continuous data
- **Data format**: 3D storage arrays, 2D query arrays
- **Distance**: Usually `"rangequery"`
- **Features**: Full voltage conversion, noise support

### TCAM (Ternary CAM)  
- **Best for**: Exact matching, binary/ternary data
- **Data format**: 2D arrays
- **Distance**: Usually `"hamming"`
- **Features**: Basic support, numerical values

### MCAM (Multi-bit CAM)
- **Best for**: Multi-level data
- **Data format**: 2D arrays  
- **Features**: Basic support, configurable bits

## Common Patterns

### Decision Tree Inference
```python
config = (CAMConfig.preset("decision_tree")
          .array_size(128, 128)
          .variation("bitflip", flip_rate=0.02)
          .build())
```

### Image Search
```python  
config = (CAMConfig.preset("image_search")
          .distance("euclidean")
          .match_type("knn")
          .match_parameter(5)
          .build())
```

### High-Performance Database
```python
config = (CAMConfig()
          .distance("hamming")
          .match_type("exact")
          .cell_type("TCAM")
          .array_size(512, 64)
          .architecture(8, 4, 2)
          .no_variation()
          .build())
```