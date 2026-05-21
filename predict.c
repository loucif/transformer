#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define CHAR_VOCAB_SIZE 28
#define CHAR_D_MODEL 16
#define CHAR_NUM_HEADS 2
#define CHAR_NUM_LAYERS 1
#define CHAR_D_FF 32
#define CHAR_MAX_SEQ_LEN 16

const char TRAINING_CHARSET[CHAR_VOCAB_SIZE] =
    "abcdefghijklmnopqrstuvwxyz \n";

int char_to_id(char c) {
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c == ' ') return 26;
    if (c == '\n') return 27;
    return -1;
}

char id_to_char(int id) {
    if (id >= 0 && id < CHAR_VOCAB_SIZE) return TRAINING_CHARSET[id];
    return '?';
}

void generate_text(const Transformer *model, const int *prompt, int prompt_len, int gen_len) {
    int *tokens = (int *)malloc((prompt_len + gen_len) * sizeof(int));
    float *logits = (float *)malloc((prompt_len + gen_len) * CHAR_VOCAB_SIZE * sizeof(float));
    memcpy(tokens, prompt, prompt_len * sizeof(int));
    int current_len = prompt_len;

    for (int i = 0; i < prompt_len; i++) printf("%c", id_to_char(tokens[i]));

    for (int step = 0; step < gen_len; step++) {
        transformer_forward(model, tokens, logits, current_len);
        float *last = &logits[(current_len - 1) * CHAR_VOCAB_SIZE];
        int next = transformer_sample(last, CHAR_VOCAB_SIZE, 0.8f);
        tokens[current_len++] = next;
        printf("%c", id_to_char(next));
    }
    printf("\n");

    free(tokens);
    free(logits);
}

int main(void) {
    srand(time(NULL));

    TransformerConfig config = {
        .vocab_size = CHAR_VOCAB_SIZE,
        .d_model = CHAR_D_MODEL,
        .num_heads = CHAR_NUM_HEADS,
        .num_layers = CHAR_NUM_LAYERS,
        .d_ff = CHAR_D_FF,
        .max_seq_len = CHAR_MAX_SEQ_LEN,
        .dropout_rate = 0.0f
    };

    Transformer model;
    transformer_init(&model, config);

    if (transformer_load(&model, "weights.bin") != 0) {
        printf("Error: Could not load weights.bin — run train.c first.\n");
        transformer_free(&model);
        return 1;
    }
    printf("Loaded weights.bin\n\n");

    printf("=== Character-Level Text Generator ===\n");
    printf("Type a prompt (lowercase letters, spaces, newlines) and press Enter.\n");
    printf("Type 'quit' to exit.\n\n");

    char input[256];
    while (1) {
        printf("prompt> ");
        if (!fgets(input, sizeof(input), stdin)) break;

        size_t len = strlen(input);
        if (len > 0 && input[len - 1] == '\n') input[len - 1] = '\0';

        if (strcmp(input, "quit") == 0) break;

        int prompt_tokens[CHAR_MAX_SEQ_LEN];
        int prompt_len = 0;

        for (int i = 0; input[i] != '\0' && prompt_len < CHAR_MAX_SEQ_LEN; i++) {
            int id = char_to_id(input[i]);
            if (id >= 0) prompt_tokens[prompt_len++] = id;
        }

        if (prompt_len == 0) continue;

        generate_text(&model, prompt_tokens, prompt_len, 50);
    }

    transformer_free(&model);
    return 0;
}
