import struct

def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8', errors='ignore')

def read_value(f, val_type):
    if val_type == 0: return struct.unpack('<B', f.read(1))[0]
    elif val_type == 1: return struct.unpack('<b', f.read(1))[0]
    elif val_type == 2: return struct.unpack('<H', f.read(2))[0]
    elif val_type == 3: return struct.unpack('<h', f.read(2))[0]
    elif val_type == 4: return struct.unpack('<I', f.read(4))[0]
    elif val_type == 5: return struct.unpack('<i', f.read(4))[0]
    elif val_type == 6: return struct.unpack('<f', f.read(4))[0]
    elif val_type == 7: return struct.unpack('<b', f.read(1))[0] != 0
    elif val_type == 8: return read_string(f)
    elif val_type == 9: # Array
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, arr_type) for _ in range(arr_len)]
    else:
        return f"<type {val_type}>"

with open('Qwen3.5-0.8B-Q4_K_M.gguf', 'rb') as f:
    magic = f.read(4)
    if magic != b'GGUF':
        print("Not a GGUF file")
        import sys
        sys.exit(1)
    version = struct.unpack('<I', f.read(4))[0]
    tensors = struct.unpack('<Q', f.read(8))[0]
    kv_count = struct.unpack('<Q', f.read(8))[0]
    print(f"GGUF Version: {version}, Tensors: {tensors}, KV pairs: {kv_count}")
    
    # Skip KV pairs
    for _ in range(kv_count):
        key = read_string(f)
        val_type = struct.unpack('<I', f.read(4))[0]
        read_value(f, val_type)
        
    # Read tensor infos
    for i in range(tensors):
        name = read_string(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        t_type = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        if "blk.0." in name:
            print(f"Tensor {i}: {name} (shape {dims}, type {t_type})")
