# Vendored: residual U-Net encoder

Source: `gardening_tools` by Sebastian Llambias et al.
<https://github.com/Sllambias/gardening_tools>, commit `808c6462a0c4b6a8738b2696844da9b3322ccb5d`,
licensed Apache-2.0 (see `LICENSE` in this directory).

Files copied from `gardening_tools/modules/networks/components/`:

- `blocks.py` — unchanged.
- `encoders.py` — one change: the `from gardening_tools...blocks import`
  line was rewritten to the relative `from .blocks import`.

Only the encoder is vendored; the decoder and task heads are not needed,
because MedTabX uses the network as a frozen feature extractor.

The encoder configuration (`resenc_unet_b`: features (32, 64, 128, 256, 320,
320), strides 2, kernel 3, blocks (1, 3, 4, 6, 6, 6), InstanceNorm3d,
LeakyReLU) is taken from `asparagus/modules/networks/resenc_unet.py` in
<https://github.com/Sllambias/asparagus>, commit `16e00d7989915dc23633b307f1990e6dc543cec1`, Apache-2.0.
