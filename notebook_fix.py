# Add this cell to your notebook after cloning the repository to fix both issues:

import os

def fix_yolact_edge_issues():
    """Fix both import and Cython issues in yolact_edge."""
    
    # Fix 1: Import issue in config.py
    config_file = "yolact_edge/data/config.py"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Replace incorrect import
        content = content.replace(
            "from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone, MobileNetV2Backbone",
            "from yolact_edge.backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone, MobileNetV2Backbone"
        )
        
        with open(config_file, 'w') as f:
            f.write(content)
        print("✅ Fixed import issue in config.py")
    
    # Fix 2: Cython compilation issue in cython_nms.pyx
    cython_file = "yolact_edge/utils/cython_nms.pyx"
    if os.path.exists(cython_file):
        with open(cython_file, 'r') as f:
            content = f.read()
        
        # Add language level directive if not present
        if "# cython: language_level=3" not in content:
            content = "# cython: language_level=3\n" + content
        
        # Fix deprecated np.int_t type
        content = content.replace("np.int_t", "np.int32_t")
        content = content.replace("dtype=np.int", "dtype=np.int32")
        
        with open(cython_file, 'w') as f:
            f.write(content)
        print("✅ Fixed Cython compilation issue in cython_nms.pyx")
    
    print("🎉 All fixes applied! You can now run the training script.")

# Run the fix
fix_yolact_edge_issues() 