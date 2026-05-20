#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

#define VOCAB_SIZE 1000
#define D_MODEL 64
#define NUM_HEADS 4
#define NUM_LAYERS 2
#define D_FF 128
#define MAX_SEQUENCE_LENGTH 32

#define VOCAB_DISPLAY_SIZE 32

const char *vocabulary[VOCAB_DISPLAY_SIZE] = {
    [0] = "<unk>",
    [1] = "the",
    [2] = "quick",
    [3] = "brown",
    [4] = "fox",
    [5] = "jumps",
    [6] = "over",
    [7] = "lazy",
    [8] = "dog",
    [9] = "in",
    [10] = "a",
    [11] = "sunny",
    [12] = "garden",
    [13] = "while",
    [14] = "birds",
    [15] = "sing",
    [16] = "softly",
    [17] = "and",
    [18] = "wind",
    [19] = "blows",
    [20] = "gently",
    [21] = "through",
    [22] = "trees",
    [23] = "creating",
    [24] = "peaceful",
    [25] = "melody",
    [26] = "that",
    [27] = "fills",
    [28] = "the",
    [29] = "air",
    [30] = "with",
    [31] = "joy",
};

const char *get_word(int token_id) {
    if (token_id >= 0 && token_id < VOCAB_DISPLAY_SIZE && vocabulary[token_id] != NULL) {
        return vocabulary[token_id];
    }
    return "<unk>";
}

void print_matrix(const float *data, int rows, int cols, const char *name, const char **row_labels) {
    printf("\n***** %s *****\n", name);
    for (int row = 0; row < rows; row++) {
        if (row_labels && row_labels[row]) {
            printf("%-12s ", row_labels[row]);
        }
        for (int col = 0; col < cols; col++) {
            printf("%8.4f ", data[row * cols + col]);
        }
        printf("\n");
    }
}

void print_encoding(const int *tokens, int length) {
    printf("\n***** Encoding *****\n");
    for (int i = 0; i < length; i++) {
        const char *word = get_word(tokens[i]);
        printf("%-12s %d\n", word, tokens[i]);
    }
}

Transformer model;

int main(void) {
    TransformerConfig config = {
        .vocab_size = VOCAB_SIZE,
        .d_model = D_MODEL,
        .num_heads = NUM_HEADS,
        .num_layers = NUM_LAYERS,
        .d_ff = D_FF,
        .max_seq_len = MAX_SEQUENCE_LENGTH,
        .dropout_rate = 0.1f
    };
    transformer_init(&model, config);

    int input_tokens[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int sequence_length = 8;

    print_encoding(input_tokens, sequence_length);

    const char *word_labels[MAX_SEQUENCE_LENGTH];
    for (int i = 0; i < sequence_length; i++) {
        word_labels[i] = get_word(input_tokens[i]);
    }

    print_matrix(
        model.positional_embeddings.data,
        sequence_length,
        D_MODEL,
        "Positional Embedding",
        word_labels
    );

    float *token_embeddings = (float *)malloc(sequence_length * D_MODEL * sizeof(float));
    for (int pos = 0; pos < sequence_length; pos++) {
        int token = input_tokens[pos];
        for (int dim = 0; dim < D_MODEL; dim++) {
            token_embeddings[pos * D_MODEL + dim] = model.token_embeddings.data[token * D_MODEL + dim];
        }
    }
    print_matrix(token_embeddings, sequence_length, D_MODEL, "Token Embedding", word_labels);

    float *combined_embedding = (float *)malloc(sequence_length * D_MODEL * sizeof(float));
    for (int pos = 0; pos < sequence_length; pos++) {
        for (int dim = 0; dim < D_MODEL; dim++) {
            combined_embedding[pos * D_MODEL + dim] = token_embeddings[pos * D_MODEL + dim]
                                                    + model.positional_embeddings.data[pos * D_MODEL + dim];
        }
    }
    print_matrix(combined_embedding, sequence_length, D_MODEL, "Combined Embedding", word_labels);

    float *attention_output = (float *)malloc(sequence_length * D_MODEL * sizeof(float));
    multihead_attention_forward(
        &model.blocks[0].self_attention,
        combined_embedding,
        attention_output,
        sequence_length,
        D_MODEL,
        1
    );
    print_matrix(attention_output, sequence_length, D_MODEL, "Single Head Attention", word_labels);

    float *ff_input = (float *)malloc(sequence_length * D_MODEL * sizeof(float));
    for (int pos = 0; pos < sequence_length; pos++) {
        layernorm_forward(&model.blocks[0].layer_norm_attn, &attention_output[pos * D_MODEL], &ff_input[pos * D_MODEL], D_MODEL);
    }
    float *ff_output = (float *)malloc(sequence_length * D_MODEL * sizeof(float));
    feedforward_forward(&model.blocks[0].feed_forward, ff_input, ff_output, sequence_length);
    print_matrix(ff_output, sequence_length, D_MODEL, "Feed Forward Output", word_labels);

    float *logits = (float *)malloc(sequence_length * VOCAB_SIZE * sizeof(float));
    transformer_forward(&model, input_tokens, logits, sequence_length);

    printf("\n***** Decoder Output *****\n");
    for (int i = 0; i < 10; i++) {
        printf("%8.4f ", logits[(sequence_length - 1) * VOCAB_SIZE + i]);
    }
    printf("...\n");

    float *last_logits = &logits[(sequence_length - 1) * VOCAB_SIZE];
    int predicted_token = transformer_sample(last_logits, VOCAB_SIZE, 1.0f);
    printf("\n>> Predicted Word: %s (token id: %d)\n", get_word(predicted_token), predicted_token);

    free(token_embeddings);
    free(combined_embedding);
    free(attention_output);
    free(ff_input);
    free(ff_output);
    free(logits);
    transformer_free(&model);

    printf("\n ***** Done! *****\n");
    return 0;
}
