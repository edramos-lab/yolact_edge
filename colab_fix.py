#!/usr/bin/env python3
"""
Comprehensive fix for yolact_edge issues in Colab environment.
This script fixes both the import issue and Cython compilation issue
in the cloned repository.
"""

import os
import re

def fix_import_issue():
    """Fix the import issue in the cloned yolact_edge repository."""
    
    config_file = "yolact_edge/data/config.py"
    
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found. Make sure you're in the yolact_edge directory.")
        return False
    
    # Read the file
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Check if the fix is already applied
    if "from yolact_edge.backbone import" in content:
        print("Import fix already applied.")
        return True
    
    # Apply the fix - handle both cases
    old_imports = [
        "from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone, MobileNetV2Backbone",
        "from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone"
    ]
    
    new_imports = [
        "from yolact_edge.backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone, MobileNetV2Backbone",
        "from yolact_edge.backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone"
    ]
    
    fixed = False
    for old_import, new_import in zip(old_imports, new_imports):
        if old_import in content:
            content = content.replace(old_import, new_import)
            print(f"Fixed import: {old_import[:50]}...")
            fixed = True
            break
    
    if not fixed:
        print("Could not find the problematic import line.")
        return False
    
    # Write the fixed content back
    with open(config_file, 'w') as f:
        f.write(content)
    
    print("Import issue fixed successfully!")
    return True

def fix_cython_issue():
    """Fix the Cython compilation issue in cython_nms.pyx."""
    
    cython_file = "yolact_edge/utils/cython_nms.pyx"
    
    if not os.path.exists(cython_file):
        print(f"Error: {cython_file} not found. Make sure you're in the yolact_edge directory.")
        return False
    
    # Read the file
    with open(cython_file, 'r') as f:
        content = f.read()
    
    # Check if language level directive is already present
    if "# cython: language_level=3" in content:
        print("Cython language level directive already present.")
    else:
        # Add language level directive at the top
        content = "# cython: language_level=3\n" + content
        print("Added Cython language level directive.")
    
    # Fix the deprecated np.int_t type - handle multiple variations
    replacements = [
        ("np.int_t", "np.int32_t"),
        ("dtype=np.int", "dtype=np.int32"),
        ("dtype=np.int64", "dtype=np.int32"),  # In case it was already changed
    ]
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"Fixed: {old} -> {new}")
    
    # Write the fixed content back
    with open(cython_file, 'w') as f:
        f.write(content)
    
    print("Cython compilation issue fixed successfully!")
    return True

def main():
    """Run all fixes."""
    print("🔧 Starting yolact_edge fixes for Colab environment...")
    
    # Check if we're in the right directory
    if not os.path.exists("yolact_edge"):
        print("❌ Error: yolact_edge directory not found!")
        print("Make sure you're in the correct directory after cloning.")
        return
    
    # Fix import issue
    print("\n1. Fixing import issue...")
    import_fixed = fix_import_issue()
    
    # Fix Cython issue
    print("\n2. Fixing Cython compilation issue...")
    cython_fixed = fix_cython_issue()
    
    if import_fixed and cython_fixed:
        print("\n✅ All fixes applied successfully!")
        print("You should now be able to run the training script without errors.")
    else:
        print("\n⚠️ Some fixes may not have been applied. Check the output above.")

if __name__ == "__main__":
    main() 