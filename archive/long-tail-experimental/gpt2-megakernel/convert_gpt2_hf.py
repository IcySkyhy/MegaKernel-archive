#!/usr/bin/env python3
"""
Download GPT-2 weights from HuggingFace and convert to binary format for CUDA inference.

Usage:
    python convert_gpt2_hf.py                    # Downloads gpt2 (124M)
    python convert_gpt2_hf.py --model gpt2-medium # Downloads gpt2-medium (355M)
    python convert_gpt2_hf.py --output my_weights.bin
"""

import os
import sys
import argparse
import numpy as np
import torch
from transformers import GPT2LMHeadModel


def get_model_config(model_name):
    """Return model configuration parameters."""
    configs = {
        'gpt2': {
            'n_layer': 12,
            'n_head': 12,
            'n_embd': 768,
            'n_positions': 1024,
            'vocab_size': 50257
        },
        'gpt2-medium': {
            'n_layer': 24,
            'n_head': 16,
            'n_embd': 1024,
            'n_positions': 1024,
            'vocab_size': 50257
        },
        'gpt2-large': {
            'n_layer': 36,
            'n_head': 20,
            'n_embd': 1280,
            'n_positions': 1024,
            'vocab_size': 50257
        },
        'gpt2-xl': {
            'n_layer': 48,
            'n_head': 25,
            'n_embd': 1600,
            'n_positions': 1024,
            'vocab_size': 50257
        }
    }
    return configs.get(model_name, configs['gpt2'])


def download_gpt2_from_hf(model_name='gpt2'):
    """Download GPT-2 model from HuggingFace."""
    print(f"Downloading {model_name} from HuggingFace...")
    print("This may take a few minutes on first run...")
    
    try:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        print(f"✓ Successfully loaded {model_name}")
        return model
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nMake sure you have transformers installed:")
        print("  pip install transformers torch")
        sys.exit(1)


def convert_to_binary(model, output_path, model_name='gpt2'):
    """Convert PyTorch GPT-2 model to binary format."""
    
    config = get_model_config(model_name)
    n_layer = config['n_layer']
    n_embd = config['n_embd']
    vocab_size = config['vocab_size']
    n_positions = config['n_positions']
    
    print(f"\nModel Configuration:")
    print(f"  Layers: {n_layer}")
    print(f"  Embedding dim: {n_embd}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Max positions: {n_positions}")
    
    state_dict = model.state_dict()
    
    # Calculate total size
    total_params = 0
    for name, param in state_dict.items():
        total_params += param.numel()
    
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"\nConverting to binary format: {output_path}")
    
    with open(output_path, 'wb') as f:
        # 1. Token embeddings (wte): [vocab_size, n_embd]
        print("  Writing token embeddings...")
        wte = state_dict['transformer.wte.weight'].cpu().float().numpy()
        assert wte.shape == (vocab_size, n_embd), f"Expected {(vocab_size, n_embd)}, got {wte.shape}"
        f.write(wte.tobytes())
        
        # 2. Position embeddings (wpe): [n_positions, n_embd]
        print("  Writing position embeddings...")
        wpe = state_dict['transformer.wpe.weight'].cpu().float().numpy()
        assert wpe.shape == (n_positions, n_embd), f"Expected {(n_positions, n_embd)}, got {wpe.shape}"
        f.write(wpe.tobytes())
        
        # 3. Transformer layers
        for i in range(n_layer):
            print(f"  Writing layer {i+1}/{n_layer}...")
            
            # Layer Norm 1
            ln1_weight = state_dict[f'transformer.h.{i}.ln_1.weight'].cpu().float().numpy()
            ln1_bias = state_dict[f'transformer.h.{i}.ln_1.bias'].cpu().float().numpy()
            assert ln1_weight.shape == (n_embd,), f"ln1_weight shape mismatch"
            assert ln1_bias.shape == (n_embd,), f"ln1_bias shape mismatch"
            f.write(ln1_weight.tobytes())
            f.write(ln1_bias.tobytes())
            
            # Attention QKV projection
            # Note: HuggingFace stores as [n_embd, 3*n_embd], we want [3*n_embd, n_embd]
            c_attn_weight = state_dict[f'transformer.h.{i}.attn.c_attn.weight'].cpu().float().numpy()
            c_attn_bias = state_dict[f'transformer.h.{i}.attn.c_attn.bias'].cpu().float().numpy()
            
            # Transpose weight from [n_embd, 3*n_embd] to [3*n_embd, n_embd]
            c_attn_weight = c_attn_weight.T
            assert c_attn_weight.shape == (3*n_embd, n_embd), f"c_attn_weight shape mismatch"
            assert c_attn_bias.shape == (3*n_embd,), f"c_attn_bias shape mismatch"
            f.write(c_attn_weight.tobytes())
            f.write(c_attn_bias.tobytes())
            
            # Attention output projection
            c_proj_weight = state_dict[f'transformer.h.{i}.attn.c_proj.weight'].cpu().float().numpy()
            c_proj_bias = state_dict[f'transformer.h.{i}.attn.c_proj.bias'].cpu().float().numpy()
            
            # Transpose from [n_embd, n_embd] to [n_embd, n_embd] (square, but let's be consistent)
            c_proj_weight = c_proj_weight.T
            assert c_proj_weight.shape == (n_embd, n_embd), f"c_proj_weight shape mismatch"
            assert c_proj_bias.shape == (n_embd,), f"c_proj_bias shape mismatch"
            f.write(c_proj_weight.tobytes())
            f.write(c_proj_bias.tobytes())
            
            # Layer Norm 2
            ln2_weight = state_dict[f'transformer.h.{i}.ln_2.weight'].cpu().float().numpy()
            ln2_bias = state_dict[f'transformer.h.{i}.ln_2.bias'].cpu().float().numpy()
            assert ln2_weight.shape == (n_embd,), f"ln2_weight shape mismatch"
            assert ln2_bias.shape == (n_embd,), f"ln2_bias shape mismatch"
            f.write(ln2_weight.tobytes())
            f.write(ln2_bias.tobytes())
            
            # MLP first layer (c_fc)
            c_fc_weight = state_dict[f'transformer.h.{i}.mlp.c_fc.weight'].cpu().float().numpy()
            c_fc_bias = state_dict[f'transformer.h.{i}.mlp.c_fc.bias'].cpu().float().numpy()
            
            # Transpose from [n_embd, 4*n_embd] to [4*n_embd, n_embd]
            c_fc_weight = c_fc_weight.T
            assert c_fc_weight.shape == (4*n_embd, n_embd), f"c_fc_weight shape mismatch"
            assert c_fc_bias.shape == (4*n_embd,), f"c_fc_bias shape mismatch"
            f.write(c_fc_weight.tobytes())
            f.write(c_fc_bias.tobytes())
            
            # MLP second layer (c_proj)
            c_proj_mlp_weight = state_dict[f'transformer.h.{i}.mlp.c_proj.weight'].cpu().float().numpy()
            c_proj_mlp_bias = state_dict[f'transformer.h.{i}.mlp.c_proj.bias'].cpu().float().numpy()
            
            # Transpose from [4*n_embd, n_embd] to [n_embd, 4*n_embd]
            c_proj_mlp_weight = c_proj_mlp_weight.T
            assert c_proj_mlp_weight.shape == (n_embd, 4*n_embd), f"c_proj_mlp_weight shape mismatch"
            assert c_proj_mlp_bias.shape == (n_embd,), f"c_proj_mlp_bias shape mismatch"
            f.write(c_proj_mlp_weight.tobytes())
            f.write(c_proj_mlp_bias.tobytes())
        
        # 4. Final layer norm
        print("  Writing final layer norm...")
        ln_f_weight = state_dict['transformer.ln_f.weight'].cpu().float().numpy()
        ln_f_bias = state_dict['transformer.ln_f.bias'].cpu().float().numpy()
        assert ln_f_weight.shape == (n_embd,), f"ln_f_weight shape mismatch"
        assert ln_f_bias.shape == (n_embd,), f"ln_f_bias shape mismatch"
        f.write(ln_f_weight.tobytes())
        f.write(ln_f_bias.tobytes())
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ Successfully converted weights!")
    print(f"  Output file: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"\nYou can now load these weights in your CUDA code with:")
    print(f"  load_gpt2_weights(\"{output_path}\", params, {vocab_size}, {n_embd}, {n_layer}, {n_positions});")


def verify_conversion(binary_path, model, model_name='gpt2'):
    """Optional: Verify the binary file matches the original model."""
    print("\nVerifying conversion...")
    
    config = get_model_config(model_name)
    n_layer = config['n_layer']
    n_embd = config['n_embd']
    vocab_size = config['vocab_size']
    n_positions = config['n_positions']
    
    with open(binary_path, 'rb') as f:
        # Check wte
        wte_binary = np.frombuffer(f.read(vocab_size * n_embd * 4), dtype=np.float32)
        wte_model = model.state_dict()['transformer.wte.weight'].cpu().float().numpy().flatten()
        
        if np.allclose(wte_binary, wte_model, rtol=1e-5):
            print("  ✓ Token embeddings match")
        else:
            print("  ✗ Token embeddings mismatch!")
            return False
    
    print("  ✓ Verification passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Download and convert GPT-2 from HuggingFace')
    parser.add_argument('--model', type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='GPT-2 model size to download')
    parser.add_argument('--output', type=str, default=None,
                        help='Output binary file path (default: <model_name>_weights.bin)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the conversion (slower)')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        args.output = f"{args.model.replace('-', '_')}_weights.bin"
    
    print("=" * 70)
    print("GPT-2 HuggingFace to Binary Converter")
    print("=" * 70)
    
    # Download model
    model = download_gpt2_from_hf(args.model)
    
    # Convert to binary
    convert_to_binary(model, args.output, args.model)
    
    # Optional verification
    if args.verify:
        verify_conversion(args.output, model, args.model)
    
    print("\n" + "=" * 70)
    print("Conversion complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()