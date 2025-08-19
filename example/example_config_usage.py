#!/usr/bin/env python3
"""
CAMASim Configuration Builder Examples

This script demonstrates the new easy-to-use configuration system for CAMASim.
"""

import numpy as np
from CAMASim.config import CAMConfig
from CAMASim.main import CAMASim


def generate_test_data(cell_type, num_samples=10, num_features=16, data_range=(0, 7), for_query=False):
    """Generate appropriate test data for different CAM cell types."""
    if cell_type == "ACAM":
        if for_query:
            # Query data for ACAM is 2D - single values to test against ranges
            return np.random.uniform(data_range[0], data_range[1], (num_samples, num_features))
        else:
            # ACAM storage data needs 3D array with [min, max] thresholds
            data = np.random.uniform(data_range[0], data_range[1], (num_samples, num_features, 2))
            # Ensure min <= max for each threshold pair
            data[:, :, 1] = np.maximum(data[:, :, 0], data[:, :, 1])
            return data
    else:
        # TCAM, BCAM, MCAM use 2D arrays for both storage and query
        return np.random.randint(data_range[0], data_range[1] + 1, (num_samples, num_features))


def example_simple_usage():
    """Example 1: Simple configuration using presets"""
    print("=== Example 1: Simple Preset Usage ===")
    
    # Create configuration using preset
    config = CAMConfig.preset("decision_tree").build()
    print("Decision tree preset config created")
    
    # Initialize CAMASim
    cam = CAMASim(config)
    
    # Generate appropriate test data for the cell type
    data = generate_test_data("ACAM", num_samples=10, num_features=16, for_query=False)
    query = generate_test_data("ACAM", num_samples=5, num_features=16, for_query=True)
    
    print(f"Writing {data.shape[0]} samples to CAM...")
    cam.write(data)
    
    print(f"Querying with {query.shape[0]} samples...")
    results, latency, energy = cam.query(query)
    if results is not None:
        print(f"Query completed. Results: {len(results)} result sets")
    else:
        print("Query completed. No results returned.")
    print()


def example_custom_configuration():
    """Example 2: Custom configuration with method chaining"""
    print("=== Example 2: Custom Configuration ===")
    
    # Build custom configuration using fluent interface
    # Use TCAM with proper bit configuration that matches the cost_config.json
    config = (CAMConfig()
              .distance("hamming")  # Use hamming for simpler data
              .match_type("exact")  # Use "exact" instead of "best"
              .array_size(128, 128)  # Use supported size
              .cell_type("TCAM")  # Use TCAM for simpler 2D array format
              .data_bits(1)  # Use 1-bit for TCAM (matches cost_config.json)
              .architecture(subarrays=2, arrays=2, mats=2)
              .device_type("RRAM")
              .no_variation()  # Disable noise for this example
              .build())
    
    print("Custom TCAM config created with method chaining")
    print(f"  - Array: {config['array']['row']}x{config['array']['col']}")
    print(f"  - Cell type: {config['array']['cell']}")
    print(f"  - Data bits: {config['array']['bit']}")
    
    # Initialize and test
    cam = CAMASim(config)
    print("CAMASim initialized with custom config")
    
    # Test with appropriate data for TCAM (binary data for 1-bit)
    data = np.random.randint(0, 2, (5, 8))  # Binary data for TCAM
    query = np.random.randint(0, 2, (2, 8))  # Binary query
    
    print(f"Testing with {data.shape} binary data array for TCAM")
    cam.write(data)
    results, _, _ = cam.query(query)
    if results is not None:
        print(f"Query successful, results: {len(results)} result sets")
    else:
        print("Query successful, no results returned")
    print()


def example_noise_configuration():
    """Example 3: Configuration with different noise types"""
    print("=== Example 3: Noise Configuration ===")
    
    # Use ACAM-based config for reliable noise testing
    base_config = CAMConfig.preset("decision_tree").distance("rangequery")
    
    # Configuration with Gaussian noise
    gaussian_config = base_config.variation("gaussian", std_dev=0.05).build()
    print("Config with Gaussian noise created")
    
    # Configuration with bitflip noise  
    bitflip_config = (CAMConfig.preset("decision_tree")
                     .distance("rangequery")
                     .variation("bitflip", flip_rate=0.1)
                     .build())
    print("Config with bitflip noise created")
    
    # Test both configurations with proper ACAM data
    test_data = generate_test_data("ACAM", num_samples=5, num_features=8, for_query=False)
    test_query = generate_test_data("ACAM", num_samples=2, num_features=8, for_query=True)
    
    for name, config in [("Gaussian", gaussian_config), ("Bitflip", bitflip_config)]:
        print(f"\nTesting {name} noise with ACAM:")
        try:
            cam = CAMASim(config)
            cam.write(test_data)
            results, _, _ = cam.query(test_query)
            if results is not None:
                print(f"  Results obtained: {len(results)} result sets")
            else:
                print("  No results returned")
        except Exception as e:
            print(f"  Error: {e}")
    print()


def example_advanced_configuration():
    """Example 4: Advanced configuration with all options"""
    print("=== Example 4: Advanced Configuration ===")
    
    config = (CAMConfig()
              .distance("hamming")
              .match_type("threshold")
              .match_parameter(3)  # Allow up to 3 bit differences
              .array_size(512, 64)
              .cell_type("TCAM") 
              .data_bits(2)
              .architecture(subarrays=8, arrays=4, mats=2)
              .device_type("FeFET")
              .variation("d2d", std_dev=0.02)
              .quantization(enabled=True, bits=6)
              .sensing_circuit("threshold")
              .enable_performance_evaluation(True)
              .build())
    
    print("Advanced config created with:")
    print(f"  - Hamming distance with threshold matching (≤3 differences)")
    print(f"  - 512x64 TCAM array with 2-bit data")
    print(f"  - 8x4x2 architecture hierarchy") 
    print(f"  - FeFET device with D2D variation")
    print(f"  - 6-bit quantization enabled")
    print()


def example_config_validation():
    """Example 5: Configuration validation and error handling"""
    print("=== Example 5: Configuration Validation ===")
    
    try:
        # This will fail - invalid distance function
        config = CAMConfig().distance("invalid_distance")
        print("This should not print")
    except Exception as e:
        print(f"✓ Caught expected error: {e}")
    
    try:
        # This will fail - negative array size
        config = CAMConfig().array_size(-10, 20) 
        print("This should not print")
    except Exception as e:
        print(f"✓ Caught expected error: {e}")
    
    try:
        # This will fail - bitflip without flip rate
        config = CAMConfig().variation("bitflip").build()
        print("This should not print") 
    except Exception as e:
        print(f"✓ Caught expected error: {e}")
    
    # Valid configuration
    config = (CAMConfig.preset("small_test")
              .variation("bitflip", flip_rate=0.05)
              .build())
    print("✓ Valid configuration created successfully")
    print()


def example_save_load_config():
    """Example 6: Saving and loading configurations"""
    print("=== Example 6: Save/Load Configuration ===")
    
    # Create and save configuration
    config = (CAMConfig.preset("decision_tree")
              .variation("bitflip", flip_rate=0.02)
              .architecture(subarrays=2, arrays=2, mats=2))
    
    config.save("my_cam_config.json")
    print("Configuration saved to my_cam_config.json")
    
    # Load configuration
    loaded_config = CAMConfig.load("my_cam_config.json")
    print("Configuration loaded from file")
    print(f"Loaded config: {loaded_config}")
    print()


if __name__ == "__main__":
    print("CAMASim Configuration Builder Examples\n")
    print("This demonstrates the new easy-to-use configuration system")
    print("based on the configuration parameters from the paper.\n")
    
    try:
        example_simple_usage()
        example_custom_configuration() 
        example_noise_configuration()
        example_config_validation()
        example_save_load_config()
        
        print("All examples completed successfully!")
        print("\nNext steps:")
        print("1. Try creating your own configurations using CAMConfig")
        print("2. Use presets as starting points: 'decision_tree', 'image_search', 'database', 'small_test'")
        print("3. Chain methods to customize: .distance().match_type().array_size().variation()")
        print("4. Save configurations for reuse: config.save('my_config.json')")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure CAMASim is properly installed and in your Python path")
    except Exception as e:
        print(f"Error running examples: {e}")
        print("This might be due to missing dependencies or CAMASim setup issues")