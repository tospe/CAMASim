#!/usr/bin/env python3
"""
Example demonstrating parameter flow from CAMASim to EvaCAM.

This example shows exactly which CAMASim configuration parameters are 
translated to EvaCAM configuration settings and how they impact the circuit-level analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from CAMASim import CAMASim, CAMConfig
from CAMASim.performance.cost import get_EVACAM_cost
import json


def demonstrate_parameter_mapping():
    """Show detailed parameter mapping from CAMASim to EvaCAM."""
    
    print("=" * 80)
    print("CAMASIM TO EVACAM PARAMETER MAPPING EXAMPLE")
    print("=" * 80)
    
    # Example 1: Basic parameter mapping
    print("\n1. BASIC PARAMETER MAPPING")
    print("-" * 50)
    
    # Create CAMASim configuration
    config = (CAMConfig()
              .distance("hamming")
              .match_type("exact")
              .array_size(128, 64)  # This will be sent to EvaCAM
              .cell_type("TCAM")
              .device_type("FeFET")  # This determines EvaCAM config file
              .use_evacam_cost(True)
              .build())
    
    print("CAMASim Configuration:")
    print(f"  Array Size: {config['array']['row']} × {config['array']['col']}")
    print(f"  Cell Type: {config['array']['cell']}")
    print(f"  Device Type: {config['cell']['device']}")
    print(f"  Use EvaCAM: {config['array'].get('useEVACAMCost', False)}")
    
    # Show parameters that will be sent to EvaCAM
    print("\nParameters sent to EvaCAM:")
    print("  1. CAPACITY (B): calculated from array dimensions")
    print("  2. WORDWIDTH (bit): set from array column count")
    print("  3. Device-specific config: 2FeFET_TCAM.cfg")
    
    # Calculate what EvaCAM will receive
    row = config['array']['row']
    col = config['array']['col']
    
    # EvaCAM expects power-of-2 dimensions
    evacam_col = 2 ** (int(np.log2(col - 1)) + 1)
    evacam_row = 2 ** (int(np.log2(row - 1)) + 1)
    evacam_capacity = evacam_col * evacam_row / 8
    
    print(f"\nEvaCAM calculated parameters:")
    print(f"  Capacity: {evacam_capacity} bytes (from {evacam_row}×{evacam_col} array)")
    print(f"  WordWidth: {col} bits")
    print(f"  Config file: 2FeFET_TCAM.cfg")


def demonstrate_different_configurations():
    """Show parameter mapping for different CAMASim configurations."""
    
    print("\n\n2. MULTIPLE CONFIGURATION EXAMPLES")
    print("-" * 50)
    
    configurations = [
        {
            'name': 'Small ACAM',
            'config': (CAMConfig()
                      .array_size(32, 16)
                      .cell_type("ACAM")
                      .device_type("FeFET")
                      .use_evacam_cost(True)
                      .build()),
        },
        {
            'name': 'Large TCAM',
            'config': (CAMConfig()
                      .array_size(512, 256)
                      .cell_type("TCAM")
                      .device_type("FeFET")
                      .use_evacam_cost(True)
                      .build()),
        },
        {
            'name': 'Decision Tree Preset',
            'config': CAMConfig.preset("decision_tree_evacam").build(),
        }
    ]
    
    for example in configurations:
        print(f"\n{example['name']}:")
        config = example['config']
        
        # Extract key parameters
        row = config['array']['row']
        col = config['array']['col']
        cell_type = config['array']['cell']
        device = config['cell']['device']
        
        # Calculate EvaCAM parameters
        try:
            evacam_col = 2 ** (int(np.log2(col - 1)) + 1)
            evacam_row = 2 ** (int(np.log2(row - 1)) + 1)
            evacam_capacity = evacam_col * evacam_row / 8
        except:
            print(f"  Skip: Invalid dimensions for EvaCAM")
            continue
        
        print(f"  CAMASim: {cell_type} {device} {row}×{col}")
        print(f"  EvaCAM: Capacity={evacam_capacity}B, WordWidth={col}bit")
        print(f"  Config: {'2FeFET_TCAM.cfg' if device == 'FeFET' else 'Unknown'}")


def show_evacam_config_modification():
    """Show actual EvaCAM config file modification."""
    
    print("\n\n3. EVACAM CONFIG FILE MODIFICATION")
    print("-" * 50)
    
    # Create a specific configuration
    config = (CAMConfig()
              .array_size(256, 128)
              .cell_type("TCAM")
              .device_type("FeFET")
              .use_evacam_cost(True)
              .build())
    
    print("CAMASim Configuration:")
    print(f"  Array: {config['array']['row']} × {config['array']['col']}")
    print(f"  Device: {config['cell']['device']}")
    
    print("\nEvaCAM config file modifications:")
    print("  File: CAMASim/performance/module/EVACAM/2FeFET_TCAM.cfg")
    
    # Show what gets modified
    row = config['array']['row']
    col = config['array']['col']
    evacam_col = 2 ** (int(np.log2(col - 1)) + 1)
    evacam_row = 2 ** (int(np.log2(row - 1)) + 1)
    evacam_capacity = int(evacam_col * evacam_row / 8)
    
    print(f"\nBefore modification:")
    print(f"  -Capacity (B): 2048")
    print(f"  -WordWidth (bit): 128")
    
    print(f"\nAfter modification:")
    print(f"  -Capacity (B): {evacam_capacity}")
    print(f"  -WordWidth (bit): {col}")
    
    print(f"\nCalculations:")
    print(f"  Nearest power of 2 for columns: {col} → {evacam_col}")
    print(f"  Nearest power of 2 for rows: {row} → {evacam_row}")
    print(f"  Capacity: {evacam_col} × {evacam_row} / 8 = {evacam_capacity} bytes")


def demonstrate_actual_evacam_call():
    """Show actual EvaCAM execution with parameter passing."""
    
    print("\n\n4. ACTUAL EVACAM EXECUTION FLOW")
    print("-" * 50)
    
    try:
        # Create configuration
        config = (CAMConfig.preset("decision_tree_evacam")
                  .array_size(128, 64)
                  .device_type("FeFET")
                  .use_evacam_cost(True)
                  .build())
        
        print("Step 1: Create CAMASim configuration")
        print(f"  Array: {config['array']['row']}×{config['array']['col']}")
        print(f"  Device: {config['cell']['device']}")
        
        print("\nStep 2: Extract EvaCAM parameters")
        array_config = config['array']
        cell_config = config['cell']
        
        print(f"  Array config sent to EvaCAM: {array_config}")
        print(f"  Cell config sent to EvaCAM: {cell_config}")
        
        print("\nStep 3: Call EvaCAM cost evaluation")
        print("  Function: get_EVACAM_cost(array_config, cell_config)")
        
        # This would actually call EvaCAM (commented out for safety)
        # cost = get_EVACAM_cost(array_config, cell_config)
        print("  → Modifies 2FeFET_TCAM.cfg file")
        print("  → Runs: ./Eva-CAM 2FeFET_TCAM.cfg")
        print("  → Parses run_output.log")
        print("  → Returns cost dictionary to CAMASim")
        
        print("\nStep 4: EvaCAM returns parsed results")
        example_cost = {
            "subarray": {"latency": 635.059, "energy": 7.965e-12},
            "interconnect": {"latency": 0, "energy": 0},
            "peripheral": {"latency": 0, "energy": 0},
            "write": {"latency": 10220.0, "energy": 6.336e-12}
        }
        print(f"  Example return: {example_cost}")
        
    except Exception as e:
        print(f"EvaCAM execution would fail: {e}")
        print("  Common issues:")
        print("    - EVACAM submodule not initialized")
        print("    - g++ compiler not available")
        print("    - Invalid configuration parameters")


def show_complete_parameter_flow():
    """Show complete parameter flow from user to EvaCAM and back."""
    
    print("\n\n5. COMPLETE PARAMETER FLOW")
    print("-" * 50)
    
    print("User Input → CAMASim Config → EvaCAM Parameters → EvaCAM Results → CAMASim Performance")
    
    print("\nDetailed Flow:")
    print("1. User sets: array_size(256, 128) + device_type('FeFET')")
    print("2. CAMASim creates config dict:")
    print("   {'array': {'row': 256, 'col': 128, 'useEVACAMCost': True},")
    print("    'cell': {'device': 'FeFET', 'type': 'TCAM'}}")
    print("3. EvaCAM extraction:")
    print("   - Config file: 2FeFET_TCAM.cfg (based on device='FeFET')")
    print("   - Capacity: calculated (256×128 → 32768 bytes)")
    print("   - WordWidth: set (128 bits)")
    print("4. EvaCAM execution:")
    print("   - Modifies config file")
    print("   - Runs circuit simulation")
    print("   - Generates area/timing/power results")
    print("5. Result parsing:")
    print("   - Extracts metrics from log file")
    print("   - Converts units (ps↔ns, pJ↔J)")
    print("   - Returns to CAMASim cost format")


def main():
    """Main execution function."""
    demonstrate_parameter_mapping()
    demonstrate_different_configurations()
    show_evacam_config_modification()
    demonstrate_actual_evacam_call()
    show_complete_parameter_flow()
    
    print("\n" + "=" * 80)
    print("SUMMARY: CAMASIM TO EVACAM PARAMETER MAPPING")
    print("=" * 80)
    print("Key Parameters Sent to EvaCAM:")
    print("  • Array dimensions (row, col) → Capacity, WordWidth")
    print("  • Device type → Config file selection")
    print("  • Cell type → Circuit model selection")
    print("\nEvaCAM Returns to CAMASim:")
    print("  • Area metrics (total, mat, subarray)")
    print("  • Timing metrics (search latency, write latency)")
    print("  • Energy metrics (read/write dynamic energy)")
    print("  • Power metrics (leakage power)")
    print("\nIntegration Benefits:")
    print("  • Seamless parameter translation")
    print("  • Automatic configuration modification")
    print("  • Robust error handling and fallback")


if __name__ == "__main__":
    main()