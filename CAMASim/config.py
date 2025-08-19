"""
CAMASim Configuration Builder

Provides an easy-to-use configuration builder for CAMASim with validation
and presets based on the configuration parameters from the paper.
"""

from typing import Dict, Any, Optional, Literal
import json
import copy


class CAMConfigError(Exception):
    """Exception raised for CAM configuration errors"""
    pass


class CAMConfig:
    """Configuration builder for CAMASim with validation and presets."""
    
    # Valid options based on actual implementation (not paper table)
    DISTANCE_FUNCTIONS = ["hamming", "manhattan", "euclidean", "innerproduct", "rangequery"]
    MATCH_TYPES = ["exact", "knn", "threshold"]  # Verified in search.py
    CELL_TYPES = ["TCAM", "ACAM", "MCAM"]  # BCAM not implemented, removed 
    SENSING_TYPES = ["exact", "best", "threshold"]  # From paper, need to verify
    DEVICE_TYPES = ["FeFET", "RRAM"]  # Only these have actual implementations
    # Variation types depend on device type:
    # RRAM device: ["gaussian", "g-dependent"] 
    # Numerical device: ["gaussian", "bitflip"]
    VARIATION_TYPES = ["gaussian", "bitflip", "g-dependent"]  # Actually implemented types
    VARIATION_MODELS = ["statistical", "experimental"]  # From paper - may not be used
    MERGE_TYPES = ["voting", "and", "comparator", "gather"]  # From paper - need verification
    
    def __init__(self):
        """Initialize with default configuration."""
        self._config = {
            "query": {
                "distance": "hamming",
                "searchScheme": "exact", 
                "parameter": 1,
                "ifAddWriteNoise": 1,
                "FuncSim": 1,
                "PerfEval": 1,
                "bit": 1
            },
            "arch": {
                "SubarraysPerArray": 1,
                "ArraysPerMat": 1, 
                "MatsPerBank": 1,
                "Merge": "exact"
            },
            "array": {
                "row": 128,
                "col": 128,
                "sensing": "exact",
                "cell": "ACAM",
                "bit": 1
            },
            "cell": {
                "type": "ACAM",
                "device": "RRAM",
                "design": "6T2M",
                "representation": "voltage",
                "writeNoise": {
                    "quantization": {
                        "hasQuantNoise": 0,
                        "quantBits": 8
                    },
                    "variation": {
                        "hasVariation": 0,
                        "type": "gaussian",
                        "stdDev": 0.1
                    }
                }
            }
        }
    
    @classmethod
    def preset(cls, name: str) -> "CAMConfig":
        """Create configuration from preset."""
        config = cls()
        
        presets = {
            "decision_tree": {
                "query": {"distance": "rangequery", "searchScheme": "exact", "parameter": 20, "ifAddWriteNoise": 1, "bit": 3},
                "array": {"row": 128, "col": 128, "cell": "ACAM", "bit": 3},
                "arch": {"SubarraysPerArray": 4, "ArraysPerMat": 4, "MatsPerBank": 4},
                "cell": {"device": "RRAM", "writeNoise": {"variation": {"hasVariation": 1, "type": "gaussian"}}}
            },
            "image_search": {
                "query": {"distance": "euclidean", "searchScheme": "knn", "parameter": 5, "bit": 8}, 
                "array": {"row": 256, "col": 256, "cell": "ACAM", "bit": 8},
                "arch": {"SubarraysPerArray": 8, "ArraysPerMat": 4, "MatsPerBank": 2}
            },
            "database": {
                "query": {"distance": "hamming", "searchScheme": "threshold", "parameter": 5, "bit": 4},
                "array": {"row": 512, "col": 64, "cell": "TCAM", "bit": 4}, 
                "arch": {"SubarraysPerArray": 2, "ArraysPerMat": 8, "MatsPerBank": 4}
            },
            "small_test": {
                "query": {"distance": "hamming", "searchScheme": "exact", "bit": 1},
                "array": {"row": 32, "col": 32, "cell": "ACAM", "bit": 1},
                "arch": {"SubarraysPerArray": 1, "ArraysPerMat": 1, "MatsPerBank": 1}
            }
        }
        
        if name not in presets:
            raise CAMConfigError(f"Unknown preset '{name}'. Available presets: {list(presets.keys())}")
        
        config._apply_preset(presets[name])
        return config
    
    def _apply_preset(self, preset_config: Dict[str, Any]):
        """Apply preset configuration by deep merging."""
        for section, values in preset_config.items():
            if section in self._config:
                self._config[section].update(values)
    
    # Functional Configuration Methods
    def distance(self, func: Literal["hamming", "manhattan", "euclidean", "innerproduct", "rangequery"]) -> "CAMConfig":
        """Set distance function."""
        if func not in self.DISTANCE_FUNCTIONS:
            raise CAMConfigError(f"Invalid distance function '{func}'. Valid options: {self.DISTANCE_FUNCTIONS}")
        self._config["query"]["distance"] = func
        return self
    
    def match_type(self, match: Literal["exact", "knn", "threshold"]) -> "CAMConfig":
        """Set match type."""
        if match not in self.MATCH_TYPES:
            raise CAMConfigError(f"Invalid match type '{match}'. Valid options: {self.MATCH_TYPES}")
        self._config["query"]["searchScheme"] = match
        return self
    
    def match_parameter(self, param: int) -> "CAMConfig":
        """Set match degree/parameter."""
        if not isinstance(param, int) or param < 0:
            raise CAMConfigError("Match parameter must be a non-negative integer")
        self._config["query"]["parameter"] = param
        return self
    
    def data_bits(self, bits: int) -> "CAMConfig":
        """Set data type (number of bits)."""
        if not isinstance(bits, int) or bits < 1:
            raise CAMConfigError("Data bits must be a positive integer")
        self._config["query"]["bit"] = bits
        self._config["array"]["bit"] = bits
        return self
    
    # Architecture Configuration Methods
    def architecture(self, subarrays: int, arrays: int, mats: int) -> "CAMConfig":
        """Set architecture hierarchy."""
        if any(not isinstance(x, int) or x < 1 for x in [subarrays, arrays, mats]):
            raise CAMConfigError("Architecture parameters must be positive integers")
        
        self._config["arch"]["SubarraysPerArray"] = subarrays
        self._config["arch"]["ArraysPerMat"] = arrays
        self._config["arch"]["MatsPerBank"] = mats
        return self
    
    def merge_type(self, merge: Literal["voting", "and", "comparator", "gather", "exact"]) -> "CAMConfig":
        """Set merge type."""
        valid_merges = self.MERGE_TYPES + ["exact"]  # "exact" for backward compatibility
        if merge not in valid_merges:
            raise CAMConfigError(f"Invalid merge type '{merge}'. Valid options: {valid_merges}")
        self._config["arch"]["Merge"] = merge
        return self
    
    # Circuit Configuration Methods  
    def array_size(self, rows: int, cols: int) -> "CAMConfig":
        """Set array dimensions."""
        if any(not isinstance(x, int) or x < 1 for x in [rows, cols]):
            raise CAMConfigError("Array dimensions must be positive integers")
        
        self._config["array"]["row"] = rows
        self._config["array"]["col"] = cols
        return self
    
    def cell_type(self, cell: Literal["TCAM", "ACAM", "MCAM"]) -> "CAMConfig":
        """Set cell type."""
        if cell not in self.CELL_TYPES:
            raise CAMConfigError(f"Invalid cell type '{cell}'. Valid options: {self.CELL_TYPES}")
        
        self._config["array"]["cell"] = cell
        self._config["cell"]["type"] = cell
        return self
    
    def sensing_circuit(self, sensing: Literal["exact", "best", "threshold"], limit: Optional[float] = None) -> "CAMConfig":
        """Set sensing circuit type and optional limit."""
        if sensing not in self.SENSING_TYPES:
            raise CAMConfigError(f"Invalid sensing type '{sensing}'. Valid options: {self.SENSING_TYPES}")
        
        self._config["array"]["sensing"] = sensing
        if limit is not None:
            if not isinstance(limit, (int, float)) or limit < 0:
                raise CAMConfigError("Sensing limit must be a non-negative number")
            self._config["array"]["sensing_limit"] = float(limit)
        return self
    
    # Device Configuration Methods
    def device_type(self, device: Literal["FeFET", "RRAM"]) -> "CAMConfig":
        """Set device type."""
        if device not in self.DEVICE_TYPES:
            raise CAMConfigError(f"Invalid device type '{device}'. Valid options: {self.DEVICE_TYPES}")
        self._config["cell"]["device"] = device
        return self
    
    def variation(self, var_type: Literal["gaussian", "bitflip", "g-dependent"], 
                  std_dev: Optional[float] = None, flip_rate: Optional[float] = None) -> "CAMConfig":
        """Set variation type and parameters."""
        if var_type not in self.VARIATION_TYPES:
            raise CAMConfigError(f"Invalid variation type '{var_type}'. Valid options: {self.VARIATION_TYPES}")
        
        self._config["cell"]["writeNoise"]["variation"]["hasVariation"] = 1
        self._config["cell"]["writeNoise"]["variation"]["type"] = var_type
        
        if var_type == "bitflip":
            if flip_rate is None:
                raise CAMConfigError("Bitflip variation requires flip_rate parameter")
            if not isinstance(flip_rate, (int, float)) or not 0 <= flip_rate <= 1:
                raise CAMConfigError("Flip rate must be between 0 and 1")
            self._config["cell"]["writeNoise"]["variation"]["value"] = float(flip_rate)
        elif var_type in ["gaussian", "g-dependent"]:
            if std_dev is None:
                std_dev = 0.1  # Default
            if not isinstance(std_dev, (int, float)) or std_dev < 0:
                raise CAMConfigError("Standard deviation must be non-negative")
            self._config["cell"]["writeNoise"]["variation"]["stdDev"] = float(std_dev)
        
        return self
    
    def no_variation(self) -> "CAMConfig":
        """Disable variation/noise."""
        self._config["cell"]["writeNoise"]["variation"]["hasVariation"] = 0
        return self
    
    def quantization(self, enabled: bool = True, bits: int = 8) -> "CAMConfig":
        """Set quantization noise parameters."""
        self._config["cell"]["writeNoise"]["quantization"]["hasQuantNoise"] = 1 if enabled else 0
        if enabled:
            if not isinstance(bits, int) or bits < 1:
                raise CAMConfigError("Quantization bits must be a positive integer")
            self._config["cell"]["writeNoise"]["quantization"]["quantBits"] = bits
        return self
    
    # Utility Methods
    def enable_performance_evaluation(self, enabled: bool = True) -> "CAMConfig":
        """Enable or disable performance evaluation."""
        self._config["query"]["PerfEval"] = 1 if enabled else 0
        return self
    
    def enable_function_simulation(self, enabled: bool = True) -> "CAMConfig":
        """Enable or disable function simulation."""  
        self._config["query"]["FuncSim"] = 1 if enabled else 0
        return self
    
    def validate(self) -> None:
        """Validate the current configuration."""
        # Check required fields exist
        required_sections = ["query", "arch", "array", "cell"]
        for section in required_sections:
            if section not in self._config:
                raise CAMConfigError(f"Missing required configuration section: {section}")
        
        # Validate bit consistency
        query_bits = self._config["query"].get("bit", 1)
        array_bits = self._config["array"].get("bit", 1) 
        if query_bits != array_bits:
            raise CAMConfigError(f"Bit mismatch: query.bit={query_bits}, array.bit={array_bits}")
        
        # Validate threshold match has parameter
        if (self._config["query"]["searchScheme"] == "threshold" and 
            "parameter" not in self._config["query"]):
            raise CAMConfigError("Threshold search scheme requires parameter to be set")
    
    def build(self) -> Dict[str, Any]:
        """Build and return the final configuration dictionary."""
        self.validate()
        return copy.deepcopy(self._config)
    
    def save(self, filename: str) -> None:
        """Save configuration to JSON file."""
        config = self.build()
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod 
    def load(cls, filename: str) -> "CAMConfig":
        """Load configuration from JSON file."""
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        
        instance = cls()
        instance._config = config_dict
        return instance
    
    def __repr__(self) -> str:
        """String representation showing key configuration parameters."""
        cfg = self._config
        return (f"CAMConfig("
                f"distance={cfg['query']['distance']}, "
                f"match={cfg['query']['searchScheme']}, "
                f"array={cfg['array']['row']}x{cfg['array']['col']}, "
                f"cell={cfg['array']['cell']}, "
                f"device={cfg['cell']['device']}, "
                f"bits={cfg['query']['bit']})")