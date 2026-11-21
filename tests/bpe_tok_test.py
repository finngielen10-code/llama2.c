"""
Compare runq.c's byte-level BPE encoder against Hugging Face tokenizers on a mixed corpus:
hand-written edge cases, every line of the repo's Markdown/C/Python files, and random
strings mixing ASCII, whitespace, digits, punctuation and non-ASCII scripts/emoji.

Needs: pip install tokenizers (no torch). Build the harness first:
    gcc -O2 -I. tests/bpe_tok_test.c win.c -lm -o build/bpe_tok_test.exe
Usage:
    python tests/bpe_tok_test.py hf/SmolLM2-1.7B-Instruct/tokenizer.json smollm2_tokenizer.bin
"""
import glob
import os
import random
import subprocess
import sys

from tokenizers import Tokenizer

CASES = [
    "Hello, world!", "What is the capital of France?", "I'm here, you're there, they've gone, we'll see, he'd go.",
    "It's 3.14159 and 2024-11-21, call 555-0100.", "  leading spaces", "trailing spaces   ", "a  b   c    d",
    "line1\nline2\n\nline4\r\n", "\ttab\tseparated\t", "x=1;y=2;\nif (x>y) { return x; }", "def f(a, b):\n    return a + b\n",
    "naïve café résumé façade", "Привет, как дела?", "Γειά σου κόσμε", "你好，世界！这是一个测试。", "こんにちは世界",
    "مرحبا بالعالم", "emoji 😀👍🏽🇳🇿 done", "math: ∑x² ≤ ∞ and ½ + ¾", "quotes “smart” ‘single’ — dash … ellipsis",
    "URL https://example.com/path?q=1&r=2#frag", "email someone@example.com", "<|im_start|> not special as text",
    "ALLCAPS and MiXeD CaSe", "'s 't 're 've 'm 'll 'd", "don't won't can't shouldn't", "1234567890", "a1b2c3",
    " 1 2 3", "100,000.50 $", "    indented code", "\n\n\n", " ", "", "x\x04y\x7fz", "tab\t\tdouble",
    "Ⅳ chapter ① circled ２ fullwidth", "control\x01\x02 chars", "mixed 中文 and English", "Hindi नमस्ते दुनिया",
]


def random_strings(n, seed=0):
    rng = random.Random(seed)
    pools = ["abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789", " ", "  \n\t",
             ".,;:!?'\"()[]{}<>-_/\\@#$%^&*+=|~`", "éèàüöñçßøåæ", "абвгдежзий", "αβγδεζηθ", "你好世界中文测试",
             "😀👍🏽🚀❤️✨", "—–…“”‘’«»•", "²³½¾①Ⅳ"]
    out = []
    for _ in range(n):
        s = "".join(rng.choice(rng.choice(pools)) for _ in range(rng.randint(1, 60)))
        out.append(s)
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    hf = Tokenizer.from_file(sys.argv[1])
    # runq.c never turns special-token *text* typed by the user (e.g. "<|im_end|>") into the
    # special token: that would let chat input inject fake turn boundaries. Special tokens
    # are only inserted by id. Compare against HF's matching mode.
    hf.encode_special_tokens = True
    corpus = list(CASES)
    for path in glob.glob("*.md") + glob.glob("*.c") + glob.glob("*.py") + glob.glob("scripts/*.py"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        corpus += text.split("\n")
        corpus.append(text[:20000])  # also a long multi-line chunk
    corpus += random_strings(3000)
    corpus = [s for s in corpus if "\0" not in s]

    exe = os.path.abspath("build/bpe_tok_test.exe" if sys.platform == "win32" else "build/bpe_tok_test")
    data = b"\0".join(s.encode("utf-8") for s in corpus) + b"\0"
    res = subprocess.run([exe, sys.argv[2], str(hf.get_vocab_size())], input=data, capture_output=True, check=True)
    lines = res.stdout.decode().split("\n")
    bad = 0
    for s, line in zip(corpus, lines):
        ours = [int(x) for x in line.split()] if line.strip() else []
        ref = hf.encode(s, add_special_tokens=False).ids
        if ours != ref:
            bad += 1
            if bad <= 10:
                print(f"MISMATCH {s[:80]!r}\n  hf  : {ref[:30]}\n  ours: {ours[:30]}")
    total = len(corpus)
    print(f"{total - bad}/{total} strings tokenize identically ({bad} mismatches)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
