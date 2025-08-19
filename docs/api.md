# CAMASim API Reference

## CAMASim Class

Main simulator class for CAM-based accelerators.

### Constructor

```python
CAMASim(config: dict)
```

**Parameters:**
- `config`: Configuration dictionary (use `CAMConfig.build()` to create)

### Methods

#### write(data)
Store data in the CAM array.

**Parameters:**
- `data`: numpy array
  - ACAM: 3D array `(samples, features, 2)` with `[min, max]` ranges
  - TCAM/MCAM: 2D array `(samples, features)`

**Returns:** 
- `(latency, energy)`: Write operation performance metrics

#### query(data)  
Query the CAM array with input data.

**Parameters:**
- `data`: numpy array
  - ACAM: 2D array `(queries, features)` with single values
  - TCAM/MCAM: 2D array `(queries, features)`

**Returns:**
- `results`: List of matching indices for each query
- `latency`: Query latency in picoseconds
- `energy`: Query energy in Joules

#### from_preset(preset_name, **overrides) [Class Method]
Create CAMASim instance from preset with optional overrides.

**Parameters:**
- `preset_name`: Preset name ("decision_tree", "image_search", etc.)
- `**overrides`: Configuration method overrides

**Returns:** CAMASim instance

---

## CAMConfig Class

Configuration builder with validation and presets.

### Class Methods

#### preset(name)
Create configuration from preset.

**Parameters:**
- `name`: "decision_tree", "image_search", "database", or "small_test"

**Returns:** CAMConfig instance

#### load(filename)
Load configuration from JSON file.

**Parameters:**
- `filename`: Path to JSON configuration file

**Returns:** CAMConfig instance

### Configuration Methods

#### Functional Configuration
- `distance(func)`: Set distance function
  - Options: "hamming", "manhattan", "euclidean", "innerproduct", "rangequery"
- `match_type(match)`: Set matching type  
  - Options: "exact", "knn", "threshold"
- `match_parameter(param)`: Set match parameter (int)
- `data_bits(bits)`: Set data precision (int)

#### Architecture Configuration  
- `array_size(rows, cols)`: Set array dimensions
- `architecture(subarrays, arrays, mats)`: Set hierarchy levels
- `merge_type(merge)`: Set merge strategy

#### Circuit Configuration
- `cell_type(cell)`: Set CAM cell type
  - Options: "ACAM", "TCAM", "MCAM"
- `sensing_circuit(sensing, limit=None)`: Set sensing type
  - Options: "exact", "best", "threshold"

#### Device Configuration
- `device_type(device)`: Set device technology
  - Options: "RRAM", "FeFET" 
- `variation(type, std_dev=None, flip_rate=None)`: Set noise model
  - Types: "gaussian", "bitflip", "g-dependent"
- `no_variation()`: Disable noise
- `quantization(enabled=True, bits=8)`: Set quantization

#### Utility Methods
- `enable_function_simulation(enabled=True)`: Toggle function simulation
- `enable_performance_evaluation(enabled=True)`: Toggle performance evaluation  
- `validate()`: Explicit configuration validation
- `build()`: Build final configuration dictionary
- `save(filename)`: Save configuration to JSON file

### Example Usage

```python
# Method chaining
config = (CAMConfig()
          .distance("hamming")
          .match_type("exact")  
          .cell_type("TCAM")
          .device_type("RRAM")
          .variation("bitflip", flip_rate=0.05)
          .build())

# Using presets
config = CAMConfig.preset("decision_tree").variation("gaussian", std_dev=0.1).build()
```

---

## CAMConfigError Exception

Raised for configuration validation errors.

**Attributes:**
- `message`: Descriptive error message with valid options

**Example:**
```python
try:
    config = CAMConfig().distance("invalid")
except CAMConfigError as e:
    print(e)  # "Invalid distance function 'invalid'. Valid options: [...]"
```

---

## Configuration Schema

### Complete Configuration Structure

```python
{
    "query": {
        "distance": str,        # Distance function
        "searchScheme": str,    # Match type  
        "parameter": int,       # Match parameter
        "ifAddWriteNoise": int, # Enable write noise (0/1)
        "FuncSim": int,        # Enable function simulation (0/1)
        "PerfEval": int,       # Enable performance evaluation (0/1)
        "bit": int             # Data bits
    },
    "arch": {
        "SubarraysPerArray": int, # Hierarchy: subarrays per array
        "ArraysPerMat": int,      # Hierarchy: arrays per mat
        "MatsPerBank": int,       # Hierarchy: mats per bank
        "Merge": str              # Merge type
    },
    "array": {
        "row": int,      # Array rows
        "col": int,      # Array columns  
        "sensing": str,  # Sensing circuit type
        "cell": str,     # Cell type
        "bit": int       # Data bits (matches query.bit)
    },
    "cell": {
        "type": str,         # Cell type (matches array.cell)
        "device": str,       # Device technology
        "design": str,       # Cell design (auto-set)
        "representation": str, # Physical representation (auto-set)
        "writeNoise": {
            "quantization": {
                "hasQuantNoise": int, # Enable quantization noise (0/1)
                "quantBits": int      # Quantization bits
            },
            "variation": {
                "hasVariation": int,  # Enable variation (0/1)
                "type": str,          # Variation type
                "stdDev": float,      # Standard deviation (gaussian/g-dependent)
                "value": float        # Flip rate (bitflip)
            }
        }
    }
}
```

### Preset Configurations

#### decision_tree
- Distance: rangequery, Match: exact
- Cell: ACAM, Device: RRAM
- Array: 128×128, Architecture: 4×4×4
- Noise: Gaussian variation enabled

#### image_search  
- Distance: euclidean, Match: knn (k=5)
- Cell: ACAM, Device: RRAM
- Array: 256×256, Architecture: 8×4×2
- Bits: 8-bit data

#### database
- Distance: hamming, Match: threshold (≤5)
- Cell: TCAM, Device: RRAM  
- Array: 512×64, Architecture: 2×8×4
- Bits: 4-bit data

#### small_test
- Distance: hamming, Match: exact
- Cell: ACAM, Device: RRAM
- Array: 32×32, Architecture: 1×1×1  
- Bits: 1-bit data