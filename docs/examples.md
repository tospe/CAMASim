# CAMASim Examples

## Basic Usage

### Simple ACAM Example

```python
import numpy as np
from CAMASim import CAMASim, CAMConfig

# Create configuration
config = CAMConfig.preset("decision_tree").build()

# Initialize simulator
cam = CAMASim(config)

# Prepare data for ACAM (3D storage, 2D query)
# Storage: threshold ranges [min, max]
storage_data = np.random.uniform(0, 10, (50, 8, 2))
storage_data[:, :, 1] = np.maximum(storage_data[:, :, 0], storage_data[:, :, 1])

# Query: single values to test against ranges
query_data = np.random.uniform(0, 10, (10, 8))

# Run simulation
cam.write(storage_data)
results, latency, energy = cam.query(query_data)

print(f"Found {len(results)} result sets")
print(f"Latency: {latency:.2f} ps")
print(f"Energy: {energy:.2e} J")
```

### Simple TCAM Example

```python
import numpy as np
from CAMASim import CAMASim, CAMConfig

# Configure for TCAM
config = (CAMConfig()
          .distance("hamming")
          .match_type("exact")
          .cell_type("TCAM")
          .array_size(64, 32)
          .data_bits(1)
          .device_type("RRAM")
          .build())

# Initialize simulator
cam = CAMASim(config)

# Binary data for TCAM
storage_data = np.random.randint(0, 2, (20, 16))
query_data = np.random.randint(0, 2, (5, 16))

# Run simulation
cam.write(storage_data)
results, latency, energy = cam.query(query_data)

print(f"Query results: {len(results)} matches")
```

## Noise Testing

### Gaussian Noise

```python
# Test different noise levels
noise_levels = [0.01, 0.05, 0.1, 0.2]
results = []

for noise in noise_levels:
    config = (CAMConfig.preset("decision_tree")
              .variation("gaussian", std_dev=noise)
              .build())
    
    cam = CAMASim(config)
    cam.write(data)
    result, _, _ = cam.query(query)
    results.append(len(result))
    
print(f"Results vs noise: {list(zip(noise_levels, results))}")
```

### Bitflip Noise

```python
# Test bitflip error rates
flip_rates = [0.01, 0.05, 0.1, 0.15]

for rate in flip_rates:
    config = (CAMConfig.preset("decision_tree")
              .variation("bitflip", flip_rate=rate)
              .build())
    
    cam = CAMASim(config)
    # Test accuracy degradation...
```

## Performance Comparison

### Array Size Impact

```python
sizes = [(64, 64), (128, 128), (256, 256)]
performance = []

for rows, cols in sizes:
    config = (CAMConfig.preset("decision_tree")
              .array_size(rows, cols)
              .build())
    
    cam = CAMASim(config)
    cam.write(data)
    _, latency, energy = cam.query(query)
    
    performance.append({
        'size': f"{rows}x{cols}",
        'latency': latency,
        'energy': energy
    })

for p in performance:
    print(f"Size {p['size']}: {p['latency']:.1f}ps, {p['energy']:.2e}J")
```

### Distance Function Comparison

```python
distances = ["hamming", "manhattan", "euclidean"]

for dist in distances:
    config = (CAMConfig()
              .distance(dist)
              .match_type("exact")
              .cell_type("TCAM")
              .build())
    
    cam = CAMASim(config)
    # Compare accuracy and performance...
```

## Advanced Usage

### Custom Architecture

```python
# Large-scale configuration
config = (CAMConfig()
          .distance("hamming")
          .match_type("threshold")
          .match_parameter(3)          # Allow 3 bit differences
          .array_size(512, 128)
          .architecture(               # 8 subarrays, 4 arrays, 2 mats
              subarrays=8,
              arrays=4, 
              mats=2
          )
          .cell_type("TCAM")
          .device_type("RRAM")
          .variation("g-dependent")    # RRAM-specific noise
          .quantization(enabled=True, bits=6)
          .build())
```

### Configuration Management

```python
# Save base configurations
base_config = CAMConfig.preset("decision_tree")
base_config.save("base_dt.json")

# Create variants
for rate in [0.01, 0.05, 0.1]:
    variant = (CAMConfig.load("base_dt.json")
               .variation("bitflip", flip_rate=rate))
    variant.save(f"dt_bitflip_{rate:.2f}.json")

# Load and use
config = CAMConfig.load("dt_bitflip_0.05.json")
cam = CAMASim(config.build())
```

## Error Handling

```python
from CAMASim.config import CAMConfigError

try:
    config = (CAMConfig()
              .distance("invalid_distance")  # This will fail
              .build())
except CAMConfigError as e:
    print(f"Configuration error: {e}")
    # Use fallback configuration
    config = CAMConfig.preset("small_test").build()

# Validate before use
try:
    config = build_my_config()
    config.validate()  # Explicit validation
    cam = CAMASim(config)
except CAMConfigError as e:
    print(f"Invalid configuration: {e}")
```

## Factory Methods

```python
# Quick initialization with overrides
cam = CAMASim.from_preset("decision_tree", 
                         array_size=(64, 64),
                         variation=("bitflip", 0.05))

# Equivalent to:
config = (CAMConfig.preset("decision_tree")
          .array_size(64, 64)
          .variation("bitflip", flip_rate=0.05)
          .build())
cam = CAMASim(config)
```