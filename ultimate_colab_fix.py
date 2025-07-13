# Copy this entire cell and paste it into your Colab notebook after cloning the repository:

import os
import subprocess
import sys

def install_cython():
    """Install Cython if not already installed."""
    try:
        import cython
        print("✅ Cython already installed")
    except ImportError:
        print("📦 Installing Cython...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cython"])
        print("✅ Cython installed successfully")

def ultimate_fix_yolact_edge():
    """Ultimate comprehensive fix for all yolact_edge issues."""
    
    print("🔧 Applying ultimate comprehensive fixes to yolact_edge...")
    
    # Step 1: Install Cython
    install_cython()
    
    # Fix 1: Completely rewrite the Cython file
    cython_file = "yolact_edge/utils/cython_nms.pyx"
    if os.path.exists(cython_file):
        fixed_cython_content = '''# cython: language_level=3
## Note: Figure out the license details later.
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np

cdef inline np.float32_t max(np.float32_t a, np.float32_t b) nogil:
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b) nogil:
    return a if a <= b else b

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def nms(np.ndarray[np.float32_t, ndim=2] dets, np.float32_t thresh):
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    cdef np.ndarray[np.int64_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] suppressed = \\
            np.zeros((ndets), dtype=np.int32)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr

    with nogil:
      for _i in range(ndets):
          i = order[_i]
          if suppressed[i] == 1:
              continue
          ix1 = x1[i]
          iy1 = y1[i]
          ix2 = x2[i]
          iy2 = y2[i]
          iarea = areas[i]
          for _j in range(_i + 1, ndets):
              j = order[_j]
              if suppressed[j] == 1:
                  continue
              xx1 = max(ix1, x1[j])
              yy1 = max(iy1, y1[j])
              xx2 = min(ix2, x2[j])
              yy2 = min(iy2, y2[j])
              w = max(0.0, xx2 - xx1 + 1)
              h = max(0.0, yy2 - yy1 + 1)
              inter = w * h
              ovr = inter / (iarea + areas[j] - inter)
              if ovr >= thresh:
                  suppressed[j] = 1

    return np.where(suppressed == 0)[0]
'''
        
        with open(cython_file, 'w') as f:
            f.write(fixed_cython_content)
        print("✅ Cython file completely rewritten")
    
    # Fix 2: Fix the import in config.py
    config_file = "yolact_edge/data/config.py"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Replace the import line
        old_import = "from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone, MobileNetV2Backbone"
        new_import = "from yolact_edge.backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone, MobileNetV2Backbone"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            print("✅ Fixed import in config.py")
        else:
            print("⚠️ Import line not found or already fixed")
        
        with open(config_file, 'w') as f:
            f.write(content)
    
    # Fix 3: Fix the CUDA issue in yolact.py
    yolact_file = "yolact_edge/yolact.py"
    if os.path.exists(yolact_file):
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
        
        with open(yolact_file, 'w') as f:
            f.write(content)
    
    # Fix 4: Fix the min_size configuration issue in augmentations.py
    augmentations_file = "yolact_edge/utils/augmentations.py"
    if os.path.exists(augmentations_file):
        with open(augmentations_file, 'r') as f:
            content = f.read()
        
        # Replace the min_size assignment to use getattr for backward compatibility
        old_line = "        self.min_size = cfg.min_size"
        new_line = "        # Use max_size as min_size if min_size is not defined (for backward compatibility)\n        self.min_size = getattr(cfg, 'min_size', cfg.max_size)"
        
        if old_line in content:
            content = content.replace(old_line, new_line)
            print("✅ Fixed min_size configuration issue in augmentations.py")
        else:
            print("⚠️ min_size line not found or already fixed")
        
        with open(augmentations_file, 'w') as f:
            f.write(content)
    
    # Fix 5: Fix the min_size configuration issue in output_utils.py
    output_utils_file = "yolact_edge/layers/output_utils.py"
    if os.path.exists(output_utils_file):
        with open(output_utils_file, 'r') as f:
            content = f.read()
        
        # Replace both instances of cfg.min_size with getattr
        old_line1 = "        r_w, r_h = Resize.faster_rcnn_scale(w, h, cfg.min_size, cfg.max_size)"
        new_line1 = "        # Use max_size as min_size if min_size is not defined (for backward compatibility)\n        min_size = getattr(cfg, 'min_size', cfg.max_size)\n        r_w, r_h = Resize.faster_rcnn_scale(w, h, min_size, cfg.max_size)"
        
        old_line2 = "        r_w, r_h = Resize.faster_rcnn_scale(w, h, cfg.min_size, cfg.max_size)"
        new_line2 = "        # Use max_size as min_size if min_size is not defined (for backward compatibility)\n        min_size = getattr(cfg, 'min_size', cfg.max_size)\n        r_w, r_h = Resize.faster_rcnn_scale(w, h, min_size, cfg.max_size)"
        
        if old_line1 in content:
            content = content.replace(old_line1, new_line1)
            print("✅ Fixed first min_size issue in output_utils.py")
        
        if old_line2 in content:
            content = content.replace(old_line2, new_line2)
            print("✅ Fixed second min_size issue in output_utils.py")
        
        with open(output_utils_file, 'w') as f:
            f.write(content)
    
    print("🎉 All ultimate fixes applied! You can now run the training script.")

# Run the ultimate fix
ultimate_fix_yolact_edge() 