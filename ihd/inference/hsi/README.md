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

The UniK3D HSI variant uses the same patch-embedding adaptation on
`model.pixel_encoder.patch_embed.proj`, while keeping UniK3D's panoramic
padding/resizing behavior. This is the preferred HSI backbone when prioritizing
panoramic geometry.
