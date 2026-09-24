import torch
import struct
import sys

def convert_gpt2_to_binary(checkpoint_path, output_path):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    with open(output_path, 'wb') as f:
        # Write embeddings
        f.write(model['transformer.wte.weight'].cpu().float().numpy().tobytes())
        f.write(model['transformer.wpe.weight'].cpu().float().numpy().tobytes())
        
        # Get number of layers
        n_layer = 12  # GPT-2 124M
        
        # Write each layer
        for i in range(n_layer):
            # Layer norm 1
            f.write(model[f'transformer.h.{i}.ln_1.weight'].cpu().float().numpy().tobytes())
            f.write(model[f'transformer.h.{i}.ln_1.bias'].cpu().float().numpy().tobytes())
            
            # Attention QKV
            f.write(model[f'transformer.h.{i}.attn.c_attn.weight'].cpu().float().numpy().tobytes())
            f.write(model[f'transformer.h.{i}.attn.c_attn.bias'].cpu().float().numpy().tobytes())
            
            # Attention projection
            f.write(model[f'transformer.h.{i}.attn.c_proj.weight'].cpu().float().numpy().tobytes())
            f.write(model[f'transformer.h.{i}.attn.c_proj.bias'].cpu().float().numpy().tobytes())
            
            # Layer norm 2
            f.write(model[f'transformer.h.{i}.ln_2.weight'].cpu().float().numpy().tobytes())
            f.write(model[f'transformer.h.{i}.ln_2.bias'].cpu().float().numpy().tobytes())
            
            # MLP
            f.write(model[f'transformer.h.{i}.mlp.c_fc.weight'].cpu().float().numpy().tobytes())
            f.write(model[f'transformer.h.{i}.mlp.c_fc.bias'].cpu().float().numpy().tobytes())
            f.write(model[f'transformer.h.{i}.mlp.c_proj.weight'].cpu().float().numpy().tobytes())
            f.write(model[f'transformer.h.{i}.mlp.c_proj.bias'].cpu().float().numpy().tobytes())
        
        # Final layer norm
        f.write(model['transformer.ln_f.weight'].cpu().float().numpy().tobytes())
        f.write(model['transformer.ln_f.bias'].cpu().float().numpy().tobytes())
    
    print(f"Converted weights saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_weights.py <input.pt> <output.bin>")
        sys.exit(1)
    
    convert_gpt2_to_binary(sys.argv[1], sys.argv[2])
