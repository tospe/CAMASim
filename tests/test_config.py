#!/usr/bin/env python3
"""
Quick test of the new CAMConfig system
"""

import numpy as np
from CAMASim.config import CAMConfig, CAMConfigError

def test_config_system():
    """Test the new configuration system"""
    print("Testing CAMConfig system...")
    
    # Test preset creation
    config = CAMConfig.preset("small_test")
    print(f"✓ Preset created: {config}")
    
    # Test method chaining
    config = (CAMConfig()
              .distance("hamming")
              .array_size(64, 32)
              .cell_type("TCAM")
              .variation("bitflip", flip_rate=0.05)
              .data_bits(2))
    
    print(f"✓ Method chaining works: {config}")
    
    # Test validation
    try:
        bad_config = CAMConfig().distance("invalid")
        print("✗ Should have failed")
    except CAMConfigError as e:
        print(f"✓ Validation works: {e}")
    
    # Test build
    built_config = config.build()
    print(f"✓ Config built successfully")
    print(f"  Distance: {built_config['query']['distance']}")
    print(f"  Array size: {built_config['array']['row']}x{built_config['array']['col']}")
    print(f"  Variation: {built_config['cell']['writeNoise']['variation']['type']}")
    
    # Test factory method
    try:
        from CAMASim.main import CAMASim
        cam = CAMASim.from_preset("small_test")
        print("✓ Factory method works")
    except Exception as e:
        print(f"! Factory method issue (may be due to missing deps): {e}")
    
    print("Configuration system test completed!")

if __name__ == "__main__":
    test_config_system()