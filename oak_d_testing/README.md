# OAK-D stereo camera with Python

## Depth Map

`test_depth_map.py`, press `q` key to quit.

`test_save_depth_map.py`, press `s` key to save.

### Capture File Format

**RGB PNG (rgb_*.png)**
- PNG, 8-bit per channel
- 3 channels (BGR in OpenCV; most viewers show it as normal color)

**Depth PNG (depth_mm_*.png)**
- PNG, 16-bit, 1 channel
- dtype: uint16
- values: depth in millimeters
- 0 = invalid/no depth
