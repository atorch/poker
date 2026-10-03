# GPU note (Oct 2026, not pursued yet)

This machine has an **NVIDIA Quadro P2000 Mobile** (Pascal, 4 GB, driver 470.256, so CUDA ≤ 11.4), but nothing here uses it:
- PyTorch is installed from the CPU-only wheel index (`pyproject.toml`).
- TensorFlow's wheel has no CUDA libraries. Hence "Could not find cuda drivers" when the old Keras model is loaded.

**Expected payoff: small for now.** The network is a tiny MLP (172 → 256 → 256 → 4). A training step costs about 1.5 ms on CPU for a batch of 256, and the Python game engine plus feature encoding take about half of the wall time, so a GPU would give at most about 2× overall. It becomes worth it with larger networks, much larger batches, or search.

**If we try it:**
- Current PyTorch CUDA wheels need a newer driver than 470 (CUDA 11.8 needs 520+, 12.x needs 525+).
- Check whether recent PyTorch CUDA builds still support Pascal (sm_61).
- Measure the speedup on the training step alone before switching.
