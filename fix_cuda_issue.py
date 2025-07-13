#!/usr/bin/env python3
"""
Fix for CUDA issue in yolact_edge.
This script modifies the yolact.py file to handle CPU-only environments.
"""

import os

def fix_cuda_issue():
    """Fix the CUDA issue by modifying the yolact.py file."""
    
    yolact_file = "yolact_edge/yolact.py"
    
    if not os.path.exists(yolact_file):
        print(f"Error: {yolact_file} not found.")
        return False
    
    # Read the file
    with open(yolact_file, 'r') as f:
        content = f.read()
    
    # Find and replace the problematic CUDA call
    old_line = "torch.cuda.current_device()"
    new_line = "# torch.cuda.current_device()  # Commented out for CPU-only environments"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        print("✅ Fixed CUDA issue in yolact.py")
    else:
        print("⚠️ CUDA line not found or already fixed")
    
    # Write back
    with open(yolact_file, 'w') as f:
        f.write(content)
    
    return True

def main():
    """Run the CUDA fix."""
    print("🔧 Fixing CUDA issue in yolact_edge...")
    
    # Check if we're in the right directory
    if not os.path.exists("yolact_edge"):
        print("❌ Error: yolact_edge directory not found!")
        print("Make sure you're in the correct directory after cloning.")
        return
    
    # Fix CUDA issue
    cuda_fixed = fix_cuda_issue()
    
    if cuda_fixed:
        print("\n✅ CUDA issue fixed successfully!")
        print("You should now be able to run the training script on CPU.")
    else:
        print("\n⚠️ CUDA fix may not have been applied.")

if __name__ == "__main__":
    main() 