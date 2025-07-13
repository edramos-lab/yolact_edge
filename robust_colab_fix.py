#!/usr/bin/env python3
"""
Robust fix for yolact_edge issues in Colab environment.
This script directly manipulates the files to ensure the fixes are applied.
"""

import os
import re

def fix_cython_file_directly():
    """Directly fix the Cython file by rewriting it completely."""
    
    cython_file = "yolact_edge/utils/cython_nms.pyx"
    
    if not os.path.exists(cython_file):
        print(f"Error: {cython_file} not found.")
        return False
    
    # Create the fixed Cython content
    fixed_content = '''# cython: language_level=3
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
    
    # Write the fixed content
    with open(cython_file, 'w') as f:
        f.write(fixed_content)
    
    print("✅ Cython file completely rewritten with fixes")
    return True

def fix_config_file_directly():
    """Directly fix the config file by rewriting the import line."""
    
    config_file = "yolact_edge/data/config.py"
    
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found.")
        return False
    
    # Read the file
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
    
    # Write back
    with open(config_file, 'w') as f:
        f.write(content)
    
    return True

def main():
    """Run the robust fixes."""
    print("🔧 Applying robust fixes to yolact_edge...")
    
    # Check if we're in the right directory
    if not os.path.exists("yolact_edge"):
        print("❌ Error: yolact_edge directory not found!")
        print("Make sure you're in the correct directory after cloning.")
        return
    
    # Fix config file
    print("\n1. Fixing config.py...")
    config_fixed = fix_config_file_directly()
    
    # Fix Cython file
    print("\n2. Fixing cython_nms.pyx...")
    cython_fixed = fix_cython_file_directly()
    
    if config_fixed and cython_fixed:
        print("\n✅ All fixes applied successfully!")
        print("You should now be able to run the training script without errors.")
    else:
        print("\n⚠️ Some fixes may not have been applied.")

if __name__ == "__main__":
    main() 