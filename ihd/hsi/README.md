# HSI Depth Models

This package contains hyperspectral-native depth experiments for IH-Depth.

The first method adapts Depth Anything V2 by replacing the DINOv2 patch
embedding projection:

- Original: `Conv2d(3, hidden_dim, patch, patch)`
- HSI: `Conv2d(B, hidden_dim, patch, patch)`, where `B` is the number of LWHSI bands.

The new projection is initialized from the mean RGB kernel repeated across HSI
channels and scaled by `3 / B`. This makes inference and future fine-tuning
possible without changing the rest of the transformer/depth head.

Inputs are full LWHSI cubes, not pseudo-broadband RGB. The current default
normalization is per-band spatial standardization.
