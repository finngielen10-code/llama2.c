"""
Convert a Hugging Face Llama-architecture checkpoint (config.json + *.safetensors)
straight to the llama2.c version-2 Q8_0 format read by runq.c, using only numpy.

Writes the same layout as export.py's version2_export (header, fp32 norms, then
Q8_0 weights), but needs no torch/transformers and streams one tensor at a time,
so a 1.1B model converts in a couple of GB of RAM.

Supports grouped-query attention (num_key_value_heads < num_attention_heads),
bf16/fp16/fp32 weights, and tied or untied classifiers.

Usage:
    python scripts/hf_to_q80.py hf/TinyLlama-1.1B-Chat-v1.0 tinyllama_chat_q80.bin
"""
import glob
import json
import os
import struct
import sys

import numpy as np


class SafeTensors:
    """Minimal reader: maps tensor name -> (file, dtype, shape, byte range)."""

    DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}

    def __init__(self, model_dir):
        self.index = {}
        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not files:
            sys.exit(f"no .safetensors files in {model_dir}")
        for f in files:
            size = os.path.getsize(f)
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                if n > size - 8:
                    sys.exit(f"{f}: header length {n} is past the end of the file (truncated download?)")
                header = json.loads(fh.read(n))
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                # the byte range must match shape x dtype and lie inside the file, so a truncated
                # or corrupt file fails here with a clear message instead of deep inside numpy
                dt = meta.get("dtype")
                if dt in self.DTYPES:
                    start, end = meta["data_offsets"]
                    want = int(np.prod(meta["shape"], dtype=np.int64)) * np.dtype(self.DTYPES[dt]).itemsize
                    if not (0 <= start <= end) or end - start != want:
                        sys.exit(f"{f}: tensor {name} has a byte range that doesn't match its shape and dtype")
                    if 8 + n + end > size:
                        sys.exit(f"{f}: tensor {name} extends past the end of the file (truncated download?)")
                self.index[name] = (f, 8 + n, meta)

    def get(self, name):
        f, base, meta = self.index[name]
        dt = meta["dtype"]
        if dt not in self.DTYPES:
            sys.exit(f"unsupported dtype {dt} for {name}")
        start, end = meta["data_offsets"]
        a = np.fromfile(f, dtype=self.DTYPES[dt], count=(end - start) // np.dtype(self.DTYPES[dt]).itemsize,
                        offset=base + start)
        if dt == "BF16":
            a = (a.astype(np.uint32) << 16).view(np.float32)
        return a.astype(np.float32).reshape(meta["shape"])


def permute_reverse(w, n_heads):
    # HF interleaves rotary halves differently from Meta/llama2.c; undo it
    dim1, dim2 = w.shape
    return w.reshape(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(0, 2, 1, 3).reshape(dim1, dim2)


def quantize_q80(w, gs):
    w = w.reshape(-1, gs)
    scale = np.abs(w).max(axis=1) / 127.0
    safe = np.where(scale == 0, 1.0, scale)
    q = np.round(w / safe[:, None]).astype(np.int8)
    err = np.abs(q.astype(np.float32) * scale[:, None] - w).max()
    return q, scale.astype(np.float32), float(err)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model_dir, out_path = sys.argv[1], sys.argv[2]
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    # runq.c implements exactly the Llama 2 block: no biases, SiLU gate, plain RoPE with
    # theta 10000, RMSNorm eps 1e-5. Anything else would convert "fine" and then produce
    # garbage, so refuse it.
    problems = []
    if cfg.get("model_type") != "llama":
        problems.append(f"model_type is {cfg.get('model_type')!r}, not 'llama'")
    if cfg.get("hidden_act", "silu") != "silu":
        problems.append(f"hidden_act is {cfg.get('hidden_act')!r}, runq.c uses SiLU")
    if cfg.get("attention_bias") or cfg.get("mlp_bias"):
        problems.append("attention/MLP biases are not supported")
    if cfg.get("rope_scaling"):
        problems.append(f"rope_scaling {cfg.get('rope_scaling')!r} is not supported")
    if cfg.get("rope_theta", 10000.0) != 10000.0:
        problems.append(f"rope_theta is {cfg.get('rope_theta')}, runq.c hardcodes 10000")
    if abs(cfg.get("rms_norm_eps", 1e-5) - 1e-5) > 1e-9:
        problems.append(f"rms_norm_eps is {cfg.get('rms_norm_eps')}, runq.c hardcodes 1e-5")
    if problems:
        sys.exit("cannot convert for runq.c:\n  " + "\n  ".join(problems))

    dim = cfg["hidden_size"]
    hidden_dim = cfg["intermediate_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg.get("num_key_value_heads") or n_heads
    vocab_size = cfg["vocab_size"]
    max_seq_len = cfg["max_position_embeddings"]

    head_size = dim // n_heads
    kv_dim = n_kv_heads * head_size
    if dim % n_heads or n_heads % n_kv_heads or head_size % 2:
        sys.exit(f"bad head layout: dim={dim} n_heads={n_heads} n_kv_heads={n_kv_heads}")

    st = SafeTensors(model_dir)
    for name in st.index:
        if name.endswith(".bias"):
            sys.exit(f"cannot convert for runq.c: checkpoint has bias tensor {name}")
    shared = bool(cfg.get("tie_word_embeddings", False))
    if not shared and "lm_head.weight" not in st.index:
        sys.exit("config.json has tie_word_embeddings=false but there is no lm_head.weight "
                 "(missing shard?); refusing to use the embeddings as the classifier")

    gs = 64
    while dim % gs != 0 or hidden_dim % gs != 0:
        gs //= 2
    L = lambda i, n: f"model.layers.{i}.{n}"

    # every tensor we read, with the shape config.json implies; checked from the safetensors
    # header before anything is written, so a mismatch can't leave a half-written .bin
    norms = [(L(i, "input_layernorm.weight"), (dim,)) for i in range(n_layers)]
    norms += [(L(i, "post_attention_layernorm.weight"), (dim,)) for i in range(n_layers)]
    norms += [("model.norm.weight", (dim,))]
    mats = [("model.embed_tokens.weight", (vocab_size, dim), None)]
    mats += [(L(i, "self_attn.q_proj.weight"), (dim, dim), n_heads) for i in range(n_layers)]
    mats += [(L(i, "self_attn.k_proj.weight"), (kv_dim, dim), n_kv_heads) for i in range(n_layers)]
    mats += [(L(i, "self_attn.v_proj.weight"), (kv_dim, dim), None) for i in range(n_layers)]
    mats += [(L(i, "self_attn.o_proj.weight"), (dim, dim), None) for i in range(n_layers)]
    mats += [(L(i, "mlp.gate_proj.weight"), (hidden_dim, dim), None) for i in range(n_layers)]
    mats += [(L(i, "mlp.down_proj.weight"), (dim, hidden_dim), None) for i in range(n_layers)]
    mats += [(L(i, "mlp.up_proj.weight"), (hidden_dim, dim), None) for i in range(n_layers)]
    if not shared:
        mats.append(("lm_head.weight", (vocab_size, dim), None))
    for name, shape in norms + [(n, s) for n, s, _ in mats]:
        if name not in st.index:
            sys.exit(f"missing tensor {name}")
        got = tuple(st.index[name][2]["shape"])
        if got != shape:
            sys.exit(f"{name} has shape {got}, expected {shape} from config.json")

    with open(out_path, "wb") as out:
        out.write(struct.pack("I", 0x616b3432))  # "ak42"
        out.write(struct.pack("i", 2))           # version
        out.write(struct.pack("iiiiiii", dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len))
        out.write(struct.pack("B", int(shared)))
        out.write(struct.pack("i", gs))
        out.write(b"\0" * (256 - out.tell()))

        # fp32 norms
        for name, _ in norms:
            out.write(st.get(name).tobytes())

        # Q8_0 weights, in the order runq.c expects
        maxerr = 0.0
        for k, (name, _, perm_heads) in enumerate(mats):
            w = st.get(name)
            if perm_heads is not None:
                w = permute_reverse(w, perm_heads)
            q, s, err = quantize_q80(w, gs)
            out.write(q.tobytes())
            out.write(s.tobytes())
            maxerr = max(maxerr, err)
            print(f"\r{k + 1}/{len(mats)} {name} {tuple(w.shape)}", end="", flush=True)
    print(f"\nmax Q8_0 group error: {maxerr:.5f}")
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
