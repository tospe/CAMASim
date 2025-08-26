#!/usr/bin/env python3
"""
Example demonstrating actual CAMASim execution with EVACAM cost evaluation.

This example runs a complete CAMASim simulation using EVACAM for circuit-level
hardware cost analysis, showing the difference between predefined costs and 
EVACAM-evaluated costs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from CAMASim import CAMASim, CAMConfig
from DecisionTree.dt2cam import DT2Array

def load_and_prepare_data():
    """Load iris dataset and train a decision tree."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train decision tree
    clf = DecisionTreeClassifier(random_state=42, max_depth=10)
    clf.fit(X_train, y_train)
    
    return clf, X_test, y_test

def convert_to_cam_format(clf, X_test):
    """Convert decision tree and test data to CAM format."""
    threshold_array, col2featureID, row2classID, threshold_min, threshold_max = DT2Array(clf)
    
    X_feature = X_test[:, col2featureID]
    query_array = np.clip(X_feature, threshold_min, threshold_max)
    
    return threshold_array, query_array, row2classID

def run_evacam_comparison():
    """Compare CAMASim performance with and without EVACAM cost evaluation."""
    
    print("=== CAMASim EVACAM Cost Evaluation Example ===\n")
    
    # Prepare data
    clf, X_test, y_test = load_and_prepare_data()
    cam_data, cam_query, row2classID = convert_to_cam_format(clf, X_test)
    
    print(f"CAM Array shape: {cam_data.shape}")
    print(f"CAM Query shape: {cam_query.shape}\n")
    
    # Configuration 1: Using predefined costs (default)
    print("1. Running CAMASim with predefined costs:")
    print("-" * 45)
    
    config_predefined = (CAMConfig.preset("decision_tree")
                        .use_evacam_cost(False)  # Explicitly disable EVACAM
                        .build())
    
    cam_predefined = CAMASim(config_predefined)
    
    # Write phase with predefined costs
    print("Writing data to CAM...")
    write_latency_pred, write_energy_pred = cam_predefined.write(cam_data)
    print(f"Write - Latency: {write_latency_pred}, Energy: {write_energy_pred}")
    
    # Query phase with predefined costs
    print("Querying CAM...")
    query_results_pred, query_latency_pred, query_energy_pred = cam_predefined.query(cam_query)
    print(f"Query - Latency: {query_latency_pred}, Energy: {query_energy_pred}\n")
    
    # Configuration 2: Using EVACAM cost evaluation
    print("2. Running CAMASim with EVACAM cost evaluation:")
    print("-" * 50)
    
    try:
        config_evacam = (CAMConfig.preset("decision_tree_evacam")
                        .use_evacam_cost(True)  # Enable EVACAM
                        .build())
        
        cam_evacam = CAMASim(config_evacam)
        
        # Write phase with EVACAM costs
        print("Writing data to CAM (with EVACAM cost evaluation)...")
        write_latency_evacam, write_energy_evacam = cam_evacam.write(cam_data)
        print(f"Write - Latency: {write_latency_evacam}, Energy: {write_energy_evacam}")
        
        # Query phase with EVACAM costs
        print("Querying CAM (with EVACAM cost evaluation)...")
        query_results_evacam, query_latency_evacam, query_energy_evacam = cam_evacam.query(cam_query)
        print(f"Query - Latency: {query_latency_evacam}, Energy: {query_energy_evacam}")
        
        # Compare results
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON:")
        print("="*60)
        print(f"{'Metric':<20} {'Predefined':<15} {'EVACAM':<15} {'Difference':<15}")
        print("-" * 65)
        
        if write_latency_pred and write_latency_evacam:
            latency_diff = ((write_latency_evacam - write_latency_pred) / write_latency_pred) * 100
            print(f"{'Write Latency':<20} {write_latency_pred:<15.2e} {write_latency_evacam:<15.2e} {latency_diff:<15.2f}%")
        
        if write_energy_pred and write_energy_evacam:
            energy_diff = ((write_energy_evacam - write_energy_pred) / write_energy_pred) * 100
            print(f"{'Write Energy':<20} {write_energy_pred:<15.2e} {write_energy_evacam:<15.2e} {energy_diff:<15.2f}%")
        
        if query_latency_pred and query_latency_evacam:
            latency_diff = ((query_latency_evacam - query_latency_pred) / query_latency_pred) * 100
            print(f"{'Query Latency':<20} {query_latency_pred:<15.2e} {query_latency_evacam:<15.2e} {latency_diff:<15.2f}%")
        
        if query_energy_pred and query_energy_evacam:
            energy_diff = ((query_energy_evacam - query_energy_pred) / query_energy_pred) * 100
            print(f"{'Query Energy':<20} {query_energy_pred:<15.2e} {query_energy_evacam:<15.2e} {energy_diff:<15.2f}%")
        
        # Verify functional results are the same
        if query_results_pred is not None and query_results_evacam is not None:
            results_match = np.array_equal(query_results_pred, query_results_evacam)
            print(f"\nFunctional results match: {results_match}")
            if not results_match:
                print("Warning: EVACAM and predefined cost models produced different functional results!")
        
    except Exception as e:
        print(f"EVACAM evaluation failed: {e}")
        print("\nThis could be due to:")
        print("1. EVACAM submodule not initialized (run: git submodule init && git submodule update)")
        print("2. g++ compiler not available")
        print("3. EVACAM compilation issues")
        print("4. Configuration not supported by EVACAM")

def demonstrate_evacam_config_options():
    """Show different EVACAM configuration options."""
    
    print("\n" + "="*60)
    print("EVACAM CONFIGURATION OPTIONS:")
    print("="*60)
    
    # Option 1: Enable EVACAM on any preset
    config1 = (CAMConfig.preset("small_test")
               .use_evacam_cost(True)
               .build())
    print("1. Small test configuration with EVACAM enabled:")
    print(f"   useEVACAMCost: {config1['array'].get('useEVACAMCost', False)}")
    
    # Option 2: Use dedicated EVACAM preset  
    config2 = CAMConfig.preset("decision_tree_evacam").build()
    print("2. Decision tree EVACAM preset:")
    print(f"   useEVACAMCost: {config2['array'].get('useEVACAMCost', False)}")
    
    # Option 3: Custom configuration with EVACAM
    config3 = (CAMConfig()
               .distance("hamming")
               .match_type("exact")
               .array_size(128, 64)
               .cell_type("TCAM")
               .device_type("RRAM")
               .use_evacam_cost(True)
               .build())
    print("3. Custom TCAM configuration with EVACAM:")
    print(f"   useEVACAMCost: {config3['array'].get('useEVACAMCost', False)}")
    print(f"   Cell type: {config3['array']['cell']}")
    print(f"   Array size: {config3['array']['row']}x{config3['array']['col']}")

def main():
    """Main execution function."""
    # Run the complete EVACAM comparison
    run_evacam_comparison()
    
    # Show configuration options
    demonstrate_evacam_config_options()
    
    print(f"\n{'='*60}")
    print("EVACAM Integration Summary:")
    print("="*60)
    print("- EVACAM provides circuit-level hardware cost modeling")
    print("- Replaces predefined cost configurations with detailed analysis")
    print("- Requires EVACAM submodule and g++ compiler")
    print("- Can be enabled on any CAMConfig with .use_evacam_cost(True)")
    print("- Use 'decision_tree_evacam' preset for quick EVACAM setup")

if __name__ == "__main__":
    main()