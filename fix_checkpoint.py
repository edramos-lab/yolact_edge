#!/usr/bin/env python3
"""
Script to fix ResNet50 checkpoint device conflicts.
"""

import torch
import os

def fix_checkpoint(input_path, output_path):
    """Convert checkpoint to clean format without device conflicts."""
    print(f"Loading checkpoint from {input_path}...")
    
    # Try to load with different strategies
    checkpoint = None
    
    # Strategy 1: Try with weights_only=True
    try:
        checkpoint = torch.load(input_path, weights_only=True, map_location='cpu')
        print("Successfully loaded with weights_only=True")
    except Exception as e:
        print(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Try with custom map_location
        try:
            def map_location(storage, loc):
                return storage.cpu()
            checkpoint = torch.load(input_path, map_location=map_location)
            print("Successfully loaded with custom map_location")
        except Exception as e:
            print(f"Strategy 2 failed: {e}")
            
            # Strategy 3: Try loading on CPU first, then convert
            try:
                checkpoint = torch.load(input_path, map_location='cpu')
                print("Successfully loaded with map_location='cpu'")
            except Exception as e:
                print(f"Strategy 3 failed: {e}")
                raise RuntimeError("All loading strategies failed")
    
    # Convert all tensors to CPU and create clean state dict
    clean_state_dict = {}
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            clean_state_dict[key] = value.cpu().clone()
        else:
            clean_state_dict[key] = value
    
    # Save clean checkpoint
    print(f"Saving clean checkpoint to {output_path}...")
    torch.save(clean_state_dict, output_path)
    print("Checkpoint fixed successfully!")

if __name__ == "__main__":
    input_path = "weights/resnet50-19c8e357.pth"
    output_path = "weights/resnet50-19c8e357-fixed.pth"
    
    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found!")
        exit(1)
    
    fix_checkpoint(input_path, output_path) 