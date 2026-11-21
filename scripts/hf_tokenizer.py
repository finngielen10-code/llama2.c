"""
Export a Hugging Face byte-level BPE tokenizer (GPT-2 style tokenizer.json, as used by
SmolLM2) to the "BPE2" tokenizer file read by runq.c. numpy-free, stdlib only.

runq.c's original tokenizer.bin format is SentencePiece-style (scores, byte-fallback
<0xXX> tokens, a dummy leading space) and can't reproduce byte-level BPE, so this writes
a separate format that runq.c detects by its magic number:

    int32 magic 0x32455042 ("BPE2")   int32 version = 1
    int32 vocab_size   int32 bos_id   int32 eos_id   int32 n_merges   int32 max_token_length
    vocab_size x { uint8 flags (bit 0 = special token), int32 len, len raw bytes }
    n_merges x { int32 left_id, int32 right_id, int32 result_id }   (in merge-rank order)

Token strings are stored as the raw bytes they stand for (the GPT-2 byte-to-unicode
mapping is undone here), so runq.c decodes by printing the bytes.

The pre-tokenizer must be Digits(individual_digits=true) followed by ByteLevel
(use_regex=true, add_prefix_space=false); runq.c implements exactly that split. Anything
else is refused.

Usage:
    python scripts/hf_tokenizer.py hf/SmolLM2-1.7B-Instruct/tokenizer.json smollm2_tokenizer.bin
"""
import json
import struct
import sys


def bytes_to_unicode():
    # GPT-2's reversible byte <-> printable-unicode mapping
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out_path = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        tj = json.load(f)

    model = tj["model"]
    if model.get("type") != "BPE" or model.get("byte_fallback") or model.get("ignore_merges") \
            or model.get("dropout") or model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
        sys.exit("only plain byte-level BPE models are supported")
    if tj.get("normalizer") is not None:
        sys.exit(f"normalizers are not supported: {tj['normalizer']}")
    pre = tj.get("pre_tokenizer") or {}
    steps = pre.get("pretokenizers", []) if pre.get("type") == "Sequence" else [pre]
    kinds = [(s.get("type"), s.get("individual_digits"), s.get("add_prefix_space"), s.get("use_regex")) for s in steps]
    if kinds != [("Digits", True, None, None), ("ByteLevel", None, False, True)]:
        sys.exit(f"unsupported pre_tokenizer {json.dumps(pre)}; runq.c implements "
                 "Digits(individual) + ByteLevel(regex, no prefix space)")

    vocab = dict(model["vocab"])
    added = {a["content"]: (a["id"], a.get("special", False)) for a in tj.get("added_tokens", [])}
    for content, (i, _) in added.items():
        if vocab.get(content, i) != i:
            sys.exit(f"added token {content!r} id mismatch")
        vocab[content] = i
    n = max(vocab.values()) + 1
    if sorted(vocab.values()) != list(range(n)):
        sys.exit("token ids are not contiguous 0..vocab_size-1")

    u2b = {u: b for b, u in bytes_to_unicode().items()}
    pieces = [None] * n
    flags = [0] * n
    for tok, i in vocab.items():
        if tok in added and added[tok][1]:
            pieces[i] = tok.encode("utf-8")  # special: inserted by id, never produced by merges
            flags[i] = 1
        else:
            try:
                pieces[i] = bytes(u2b[c] for c in tok)
            except KeyError:
                sys.exit(f"token {tok!r} is not in the byte-level alphabet")
    missing = [b for b in range(256) if bytes([b]) not in pieces]
    if missing:
        # HF drops such bytes (no unk token); runq.c does the same
        print(f"note: {len(missing)} bytes have no token and are dropped when encoding: {missing}")

    merges = []
    for m in model["merges"]:
        a, b = m.split(" ") if isinstance(m, str) else m
        if a not in vocab or b not in vocab or a + b not in vocab:
            sys.exit(f"merge {a!r} + {b!r} refers to a missing token")
        merges.append((vocab[a], vocab[b], vocab[a + b]))

    # BOS/EOS for runq.c's generate mode: <|endoftext|> separates documents; <|im_end|> ends
    # an assistant turn (chat mode looks up <|im_start|>/<|im_end|> by name itself)
    bos = vocab.get("<|endoftext|>", 0)
    eos = vocab.get("<|im_end|>", bos)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<iiiiiii", 0x32455042, 1, n, bos, eos, len(merges), max(len(p) for p in pieces)))
        for p, fl in zip(pieces, flags):
            f.write(struct.pack("<Bi", fl, len(p)))
            f.write(p)
        for a, b, r in merges:
            f.write(struct.pack("<iii", a, b, r))
    print(f"wrote {out_path}: {n} tokens ({sum(flags)} special), {len(merges)} merges, bos={bos} eos={eos}")


if __name__ == "__main__":
    main()
