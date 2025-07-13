#!/usr/bin/env python3
"""
Comprehensive fix script for yolact_edge issues:
1. Fixes the import issue in config.py
2. Fixes the Cython compilation issue in cython_nms.pyx
Run this in your Colab notebook after cloning the repository.
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
    
    # Apply the fix
    old_import = "from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone"
    new_import = "from yolact_edge.backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone"
    
    # Handle the case with MobileNetV2Backbone
    old_import_mobile = "from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone, MobileNetV2Backbone"
    new_import_mobile = "from yolact_edge.backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone, MobileNetV2Backbone"
    
    if old_import_mobile in content:
        content = content.replace(old_import_mobile, new_import_mobile)
        print("Fixed import with MobileNetV2Backbone")
    elif old_import in content:
        content = content.replace(old_import, new_import)
        print("Fixed import without MobileNetV2Backbone")
    else:
        print("Could not find the problematic import line. The file might already be fixed or have a different structure.")
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
    
    # Fix the deprecated np.int_t type
    if "np.int_t" in content:
        content = content.replace("np.int_t", "np.int32_t")
        content = content.replace("dtype=np.int", "dtype=np.int32")
        print("Fixed deprecated np.int_t type declarations.")
    else:
        print("No deprecated np.int_t found.")
    
    # Write the fixed content back
    with open(cython_file, 'w') as f:
        f.write(content)
    
    print("Cython compilation issue fixed successfully!")
    return True

def main():
    """Run all fixes."""
    print("Starting yolact_edge fixes...")
    
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