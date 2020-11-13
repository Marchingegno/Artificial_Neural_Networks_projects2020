from tensorflow.python.client import device_lib

# What version of Python do you have?
import sys

import tensorflow.keras
import pandas as pd
import tensorflow as tf

print(device_lib.list_local_devices())
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
print(tf.config.list_physical_devices('GPU'))