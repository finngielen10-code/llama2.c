// Tokenizer check for runq.c's byte-level BPE encoder: reads NUL-separated strings from
// stdin and prints the token ids of each on its own line, for tests/bpe_tok_test.py to
// compare against Hugging Face tokenizers.
//   gcc -O2 -I. tests/bpe_tok_test.c win.c -lm -o build/bpe_tok_test.exe
#define TESTING
#include "runq.c"

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: bpe_tok_test <tokenizer.bin> <vocab_size> < strings\n"); return 1; }
    Tokenizer t;
    build_tokenizer(&t, argv[1], atoi(argv[2]));
    if (!t.bpe) { fprintf(stderr, "not a BPE2 tokenizer\n"); return 1; }
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    size_t cap = 1 << 20, n = 0;
    char* buf = malloc(cap);
    int c;
    while ((c = getchar()) != EOF) {
        if (n + 1 >= cap) { cap *= 2; buf = realloc(buf, cap); }
        buf[n++] = (char)c;
    }
    size_t i = 0;
    int* tokens = malloc((n + 3) * sizeof(int));
    while (i < n) {
        char* s = buf + i;
        size_t len = strlen(s);
        int nt = 0;
        encode(&t, s, 0, 0, tokens, &nt);
        for (int k = 0; k < nt; k++) printf(k ? " %d" : "%d", tokens[k]);
        printf("\n");
        i += len + 1;
    }
    free(tokens); free(buf); free_tokenizer(&t);
    return 0;
}
