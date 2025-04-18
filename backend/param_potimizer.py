# param_optimizer.py
import yaml
from itertools import product
import numpy as np
from tracking_evaluator import TrackingEvaluator

def evaluate_params(video_path, param_config):
    """Evaluate a specific parameter configuration"""
    # Update botsort config
    with open('custom-botsort.yaml', 'w') as f:
        yaml.dump(param_config, f)
    
    # Initialize evaluator
    evaluator = TrackingEvaluator()
    
    # Process video with current parameters
    # ... (your existing video processing code)
    
    # Return metrics
    return evaluator.get_metrics()

def optimize_parameters(video_path):
    """Find optimal parameters using grid search with early pruning"""
    # Parameter ranges to test
    param_ranges = {
        'track_high_thresh': [0.3, 0.4, 0.5],
        'track_low_thresh': [0.1, 0.15, 0.2],
        'new_track_thresh': [0.6, 0.7, 0.8],
        'track_buffer': [30, 45, 60],
        'match_thresh': [0.8, 0.85, 0.9],
        'proximity_thresh': [0.5, 0.6, 0.7],
        'appearance_thresh': [0.25, 0.35, 0.45]
    }
    
    best_score = 0
    best_params = None
    
    # Generate parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(product(*param_ranges.values()))
    
    # Test each combination
    for values in param_values:
        config = dict(zip(param_names, values))
        
        # Add fixed parameters
        config.update({
            'tracker_type': 'botsort',
            'fuse_score': True,
            'gmc_method': 'sparseOptFlow',
            'with_reid': True
        })
        
        # Evaluate current configuration
        metrics = evaluate_params(video_path, config)
        stability_score = metrics['stability_score']
        
        if stability_score > best_score:
            best_score = stability_score
            best_params = config
            print(f"New best configuration found:")
            print(f"Score: {best_score}")
            print(f"Parameters: {best_params}")
            print(f"Metrics: {metrics}")
    
    return best_params, best_score