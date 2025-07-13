#!/usr/bin/env python3
"""
Script to fix the import issue in yolact_edge config.py
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

if __name__ == "__main__":
    fix_import_issue() 