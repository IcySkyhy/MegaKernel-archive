"""M39 CPU reference harness for one official Qwen4Exp decoder layer.

Modules:
  ckpt      per-layer checkpoint loader (safetensors header index, MXFP4 dequant)
  mxfp4     E2M1 + E8M0 group-32 dequantization
  hc        hyper-connection mixer (bit-faithful to nvidia/ops/hc.py)
  gdn       GDN ("linear_attention") segment
  qsa       QSA ("full_attention") segment + indexer
  moe       MoE segment (router + routed + shared expert)
  layer     the decoder layer skeleton (delayed combine, 3-tensor boundary)
  official  direct-import probe of the official implementation
"""
