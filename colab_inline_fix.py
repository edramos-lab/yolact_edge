# Copy this entire cell and paste it into your Colab notebook after cloning the repository:

import os

def fix_yolact_edge_colab():
    """Fix both import and Cython issues in the cloned yolact_edge repository."""
    
    print("🔧 Fixing yolact_edge issues in Colab...")
    
    # Fix 1: Import issue in config.py
    config_file = "yolact_edge/data/config.py"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Handle both import variations
        old_imports = [
            "from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone, MobileNetV2Backbone",
            "from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone"
        ]
        
        new_imports = [
            "from yolact_edge.backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone, MobileNetV2Backbone",
            "from yolact_edge.backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone"
        ]
        
        fixed = False
        for old, new in zip(old_imports, new_imports):
            if old in content:
                content = content.replace(old, new)
                print(f"✅ Fixed import issue")
                fixed = True
                break
        
        if not fixed:
            print("⚠️ Import already fixed or not found")
        
        with open(config_file, 'w') as f:
            f.write(content)
    
    # Fix 2: Cython compilation issue in cython_nms.pyx
    cython_file = "yolact_edge/utils/cython_nms.pyx"
    if os.path.exists(cython_file):
        with open(cython_file, 'r') as f:
            content = f.read()
        
        # Add language level directive if not present
        if "# cython: language_level=3" not in content:
            content = "# cython: language_level=3\n" + content
            print("✅ Added Cython language level directive")
        
        # Fix deprecated np.int_t type
        if "np.int_t" in content:
            content = content.replace("np.int_t", "np.int32_t")
            print("✅ Fixed np.int_t -> np.int32_t")
        
        if "dtype=np.int" in content:
            content = content.replace("dtype=np.int", "dtype=np.int32")
            print("✅ Fixed dtype=np.int -> dtype=np.int32")
        
        with open(cython_file, 'w') as f:
            f.write(content)
    
    print("🎉 All fixes applied! You can now run the training script.")

# Run the fix
fix_yolact_edge_colab() 