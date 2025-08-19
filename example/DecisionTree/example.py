import random

import numpy as np
from dt2cam import DT2Array
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from CAMASim import CAMASim, CAMConfig


def create_cam_config(enable_noise=True, noise_type="gaussian", noise_param=None):
    """
    Create CAM configuration using the new CAMConfig builder system.
    
    Args:
        enable_noise: Whether to enable write noise
        noise_type: Type of noise ("gaussian", "bitflip", etc.)
        noise_param: Noise parameter (std_dev for gaussian, flip_rate for bitflip)
    """
    config_builder = CAMConfig.preset("decision_tree")
    
    if enable_noise:
        if noise_type == "bitflip":
            if noise_param is None:
                noise_param = 0.05  # Default 5% bitflip rate
            config_builder = config_builder.variation("bitflip", flip_rate=noise_param)
        elif noise_type == "gaussian":
            if noise_param is None:
                noise_param = 0.1  # Default std dev
            config_builder = config_builder.variation("gaussian", std_dev=noise_param)
        else:
            # Use preset default (gaussian with hasVariation=1)
            pass
    else:
        config_builder = config_builder.no_variation()
    
    return config_builder.build()

def simCAM(CAM_Data, CAM_Query, config=None):
    """
    Run CAM simulation with given data and configuration.
    
    Args:
        CAM_Data: Data array to write to CAM
        CAM_Query: Query array for CAM search
        config: Configuration dictionary (if None, uses default decision_tree preset)
    """
    if config is None:
        config = create_cam_config()
    
    cam = CAMASim(config)
    cam.write(CAM_Data)
    CAM_pred_ids, _, _ = cam.query(CAM_Query)  # Accuracy Evaluation
    print('CAM Simulation Done')
    return CAM_pred_ids

def load_dataset():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Target variable (species)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def evaluate_accuray(y_test, y_pred):
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def convert2cam(clf, X_test):
    (
        thresholdArray,
        col2featureID,
        row2classID,
        thresholdMin,
        thresholdMax,
    ) = DT2Array(clf)

    X_feature = X_test[:, col2featureID]

    queryArray = np.clip(X_feature, thresholdMin, thresholdMax)

    return thresholdArray, queryArray, row2classID


def process_cam_predictions(cam_pred_row, row2classID):
    """Process CAM prediction results to handle multiple matches."""
    processed_predictions = []
    
    for i in range(len(cam_pred_row)):
        if len(cam_pred_row[i]) > 1:
            # Multiple matches - randomly choose one
            processed_predictions.append([random.choice(cam_pred_row[i])])
        elif len(cam_pred_row[i]) == 0:
            # No matches - use default class 0
            processed_predictions.append([0])
        else:
            # Single match
            processed_predictions.append(cam_pred_row[i])
    
    pred_row = np.array(processed_predictions).ravel()
    pred_class = np.take(row2classID, pred_row)
    return pred_class

def compare_configurations():
    """Compare different CAM configurations."""
    print("\n=== Comparing Different CAM Configurations ===")
    
    X_train, X_test, y_train, y_test = load_dataset()
    
    # Train decision tree
    clf = DecisionTreeClassifier(random_state=42, max_depth=10)
    clf.fit(X_train, y_train)
    
    # Original prediction
    y_pred = clf.predict(X_test)
    accuracy_original = evaluate_accuray(y_test, y_pred)
    print(f'DT Accuracy (original): {accuracy_original:.4f}')
    
    # Convert to CAM format
    CAM_Array, CAM_Query, row2classID = convert2cam(clf, X_test)
    
    # Test different configurations
    configurations = [
        ("No Noise", create_cam_config(enable_noise=False)),
        ("Gaussian Noise (σ=0.05)", create_cam_config(noise_type="gaussian", noise_param=0.05)),
        ("Gaussian Noise (σ=0.1)", create_cam_config(noise_type="gaussian", noise_param=0.1)),
        ("Bitflip Noise (5%)", create_cam_config(noise_type="bitflip", noise_param=0.05)),
        ("Bitflip Noise (10%)", create_cam_config(noise_type="bitflip", noise_param=0.1)),
    ]
    
    for config_name, config in configurations:
        print(f"\nTesting: {config_name}")
        cam_pred_row = simCAM(CAM_Array, CAM_Query, config)
        cam_pred_class = process_cam_predictions(cam_pred_row, row2classID)
        accuracy_cam = evaluate_accuray(y_test, cam_pred_class)
        print(f'  CAM Accuracy: {accuracy_cam:.4f} (vs original: {accuracy_original:.4f})')

def main():
    """Main function demonstrating the refactored example with new config system."""
    print("CAMASim Decision Tree Example - Using New Configuration System")
    print("=" * 65)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_dataset()
    
    # Create a Decision Tree classifier
    # (Todo): Support random forest
    clf = DecisionTreeClassifier(random_state=42, max_depth=10)
    clf.fit(X_train, y_train)

    # Original prediction
    y_pred = clf.predict(X_test)
    accuracy_original = evaluate_accuray(y_test, y_pred)
    print(f'DT Accuracy (original): {accuracy_original:.4f}')

    # CAM Prediction with default configuration
    CAM_Array, CAM_Query, row2classID = convert2cam(clf, X_test)
    
    print(f'\nCAM Array shape: {CAM_Array.shape}')
    print(f'CAM Query shape: {CAM_Query.shape}')
    
    # Using new configuration system - default decision tree preset
    print("\nUsing new CAMConfig system with decision_tree preset:")
    config = CAMConfig.preset("decision_tree").build()
    
    cam_pred_row = simCAM(CAM_Array, CAM_Query, config)
    cam_pred_class = process_cam_predictions(cam_pred_row, row2classID)
    accuracy_cam = evaluate_accuray(y_test, cam_pred_class)
    print(f'DT Accuracy (CAM): {accuracy_cam:.4f}')
    
    # Demonstrate easy configuration changes
    print("\n" + "="*40)
    print("Easy Configuration Examples:")
    print("="*40)
    
    # Example 1: Different noise types
    print("\n1. Testing different noise configurations:")
    bitflip_config = CAMConfig.preset("decision_tree").variation("bitflip", flip_rate=0.05).build()
    print("   - Created bitflip noise config with 5% flip rate")
    
    no_noise_config = CAMConfig.preset("decision_tree").no_variation().build()
    print("   - Created no-noise config")
    
    # Example 2: Different array sizes
    print("\n2. Testing different array sizes:")
    small_config = (CAMConfig.preset("decision_tree")
                   .array_size(64, 64)
                   .architecture(2, 2, 2)
                   .build())
    print("   - Created smaller array configuration (64x64)")
    
    large_config = (CAMConfig.preset("decision_tree")
                   .array_size(256, 256)
                   .architecture(8, 4, 2) 
                   .build())
    print("   - Created larger array configuration (256x256)")
    
    # Example 3: Save configuration for reuse
    print("\n3. Saving configuration for reuse:")
    CAMConfig.preset("decision_tree").variation("bitflip", flip_rate=0.02).save("dt_bitflip_config.json")
    print("   - Saved bitflip configuration to 'dt_bitflip_config.json'")
    
    # Run comprehensive comparison
    compare_configurations()

if __name__ == '__main__':
    main()
