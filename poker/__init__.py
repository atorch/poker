import os

# Note: hide TensorFlow's C++ startup logging (e.g. CUDA/cuDNN not found on CPU-only machines).
#  Must be set before tensorflow is first imported. Override by setting the variable yourself.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
