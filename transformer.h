#ifndef TRANSFORMER_H
#define TRANSFORMER_H

/*
 * ============================================================================
 *  Transformer.h - Decoder-Only Transformer Library (Single Header)
 * ============================================================================
 *
 *  A pure C implementation of the decoder-only Transformer architecture
 *  based on the paper "Attention Is All You Need" (Vaswani et al., 2017).
 *  https://arxiv.org/abs/1706.03762
 *
 *  Architecture: Decoder-only (GPT-style)
 *  - Causal (masked) self-attention prevents attending to future tokens
 *  - Pre-normalization (LayerNorm before each sub-layer)
 *  - Position-wise feed-forward network with ReLU activation
 *  - Sinusoidal positional encoding
 *  - Residual connections around every sub-layer
 *
 *  Usage:
 *      In exactly ONE source file, define TRANSFORMER_IMPLEMENTATION before
 *      including this header to get the function definitions:
 *
 *          #define TRANSFORMER_IMPLEMENTATION
 *          #include "transformer.h"
 *
 *      In all other files, include normally:
 *
 *          #include "transformer.h"
 *
 *  Forward Pass Flow:
 *      Input Tokens -> Token Embedding + Positional Encoding
 *                   -> [Transformer Block] x N layers
 *                   -> Final Layer Normalization
 *                   -> LM Head (Linear Projection)
 *                   -> Logits (one per vocabulary token)
 *
 *  Each Transformer Block:
 *      Input -> LayerNorm -> Multi-Head Self-Attention -> Residual Add
 *            -> LayerNorm -> Feed-Forward Network      -> Residual Add
 *            -> LayerNorm -> Output
 *
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define PI 3.14159265358979323846f

/*
 * ============================================================================
 *  Data Structures
 * ============================================================================
 */

/*
 * Matrix - A 2D dense matrix stored in row-major order.
 *
 *  Layout: data[row * cols + col]
 *  Memory: Contiguous float array of size (rows * cols)
 */
typedef struct {
    int rows;       /* Number of rows    */
    int cols;       /* Number of columns */
    float *data;    /* Row-major flattened data */
} Matrix;

/*
 * TransformerConfig - Hyperparameters defining the model architecture.
 *
 *  vocab_size    - Total number of unique tokens in the vocabulary
 *  d_model       - Embedding dimension (width of every layer)
 *  num_heads     - Number of parallel attention heads
 *  num_layers    - Number of stacked transformer decoder blocks
 *  d_ff          - Inner dimension of the feed-forward network (typically 4 * d_model)
 *  max_seq_len   - Maximum supported sequence length for positional encoding
 *  dropout_rate  - Dropout probability (reserved for future training support)
 */
typedef struct {
    int vocab_size;
    int d_model;
    int num_heads;
    int num_layers;
    int d_ff;
    int max_seq_len;
    float dropout_rate;
} TransformerConfig;

/*
 * AttentionHeadWeights - Learnable projection matrices for a single attention head.
 *
 *  Each head projects the input into three subspaces:
 *  - Query:    What am I looking for?
 *  - Key:      What do I contain?
 *  - Value:    What information do I provide?
 *
 *  All matrices are (d_model x head_dim) where head_dim = d_model / num_heads
 */
typedef struct {
    Matrix weight_query;   /* Query projection matrix   W_Q */
    Matrix weight_key;     /* Key projection matrix     W_K */
    Matrix weight_value;   /* Value projection matrix   W_V */
} AttentionHeadWeights;

/*
 * MultiHeadAttention - The core attention mechanism.
 *
 *  Splits the embedding into multiple parallel heads, each computing
 *  independent attention. Outputs are concatenated and projected back
 *  to d_model dimensions.
 *
 *  Formula: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
 *
 *  The causal mask (when enabled) sets attention scores for future
 *  positions to -infinity, ensuring autoregressive generation.
 */
typedef struct {
    int num_heads;                  /* Number of parallel attention heads        */
    int d_model;                    /* Model embedding dimension                  */
    int head_dim;                   /* Dimension per head (d_model / num_heads)   */
    AttentionHeadWeights *heads;    /* Array of per-head weight matrices          */
    Matrix weight_output;           /* Output projection matrix  W_O (d_model x d_model) */
    float *query_cache;             /* KV cache for inference (reserved)          */
    float *key_cache;               /* KV cache for inference (reserved)          */
    float *value_cache;             /* KV cache for inference (reserved)          */
} MultiHeadAttention;

/*
 * FeedForward - Position-wise feed-forward network (FFN).
 *
 *  Applied independently to each position in the sequence.
 *  Consists of two linear transformations with a ReLU activation:
 *
 *      FFN(x) = ReLU(x * W_up + b_up) * W_down + b_down
 *
 *  The intermediate dimension d_ff is typically 4x larger than d_model,
 *  allowing the network to expand and compress representations.
 */
typedef struct {
    Matrix weight_up;       /* First linear layer weights  (d_model x d_ff) */
    Matrix bias_up;         /* First linear layer bias     (1 x d_ff)       */
    Matrix weight_down;     /* Second linear layer weights (d_ff x d_model) */
    Matrix bias_down;       /* Second linear layer bias    (1 x d_model)    */
    int d_model;            /* Input/output dimension                      */
    int d_ff;               /* Hidden (expanded) dimension                 */
} FeedForward;

/*
 * LayerNorm - Layer normalization parameters.
 *
 *  Normalizes activations across the feature dimension:
 *
 *      output = gamma * (input - mean) / sqrt(variance + epsilon) + beta
 *
 *  gamma (weight) and beta (bias) are learnable per-feature parameters.
 *  epsilon = 1e-5 for numerical stability.
 */
typedef struct {
    float *gamma;   /* Learnable scale parameter (initialized to 1.0) */
    float *beta;    /* Learnable shift parameter (initialized to 0.0) */
    int size;       /* Number of features to normalize over           */
} LayerNorm;

/*
 * TransformerBlock - A single decoder layer.
 *
 *  Standard pre-normalization architecture (LayerNorm before each sub-layer):
 *
 *  Flow:
 *      x -> LayerNorm_attn -> Self-Attention -> x + attn_out
 *         -> LayerNorm_ff  -> Feed-Forward   -> x + ff_out
 *
 *  No extra LayerNorm after the residual — that is handled at the model level.
 */
typedef struct {
    MultiHeadAttention self_attention;  /* Masked self-attention sub-layer */
    FeedForward feed_forward;           /* Position-wise FFN sub-layer     */
    LayerNorm layer_norm_attn;          /* Pre-attention normalization     */
    LayerNorm layer_norm_ff;            /* Pre-FFN normalization           */
    int d_model;                        /* Model embedding dimension        */
} TransformerBlock;

/*
 * Transformer - The complete decoder-only language model.
 *
 *  Components:
 *  1. Token Embeddings:     Lookup table mapping token IDs to dense vectors
 *  2. Positional Embeddings: Sinusoidal encodings of token positions
 *  3. Transformer Blocks:   Stacked decoder layers (num_layers deep)
 *  4. Final LayerNorm:      Normalization before the output projection
 *  5. LM Head:              Linear projection from d_model to vocab_size
 */
typedef struct {
    TransformerConfig config;         /* Model hyperparameters              */
    Matrix token_embeddings;          /* (vocab_size x d_model)             */
    Matrix positional_embeddings;     /* (max_seq_len x d_model)            */
    TransformerBlock *blocks;         /* Array of decoder blocks            */
    Matrix final_ln_gamma;            /* Final layer norm scale (1 x d_model) */
    Matrix final_ln_beta;             /* Final layer norm shift (1 x d_model) */
    Matrix lm_head;                   /* Language model head (d_model x vocab_size) */
} Transformer;

/*
 * ============================================================================
 *  Public API - Matrix Operations
 * ============================================================================
 */

/* Create a zero-initialized matrix of the given dimensions */
Matrix matrix_create(int rows, int cols);

/* Free the memory associated with a matrix and reset its dimensions */
void matrix_free(Matrix *matrix);

/* Set all elements of a matrix to zero */
void matrix_zero(Matrix *matrix);

/* Set all elements of a matrix to a scalar value */
void matrix_fill(Matrix *matrix, float value);

/* Print a matrix to stdout with an optional label */
void matrix_print(const Matrix *matrix, const char *name);

/* Deep copy source matrix into destination (reallocates if needed) */
void matrix_copy(const Matrix *source, Matrix *destination);

/* Element-wise addition: output = matrix_a + matrix_b */
void matrix_add(const Matrix *matrix_a, const Matrix *matrix_b, Matrix *output);

/* Standard matrix multiplication: output = matrix_a @ matrix_b */
void matrix_multiply(const Matrix *matrix_a, const Matrix *matrix_b, Matrix *output);

/* Transpose a matrix: output[i][j] = matrix_a[j][i] */
void matrix_transpose(const Matrix *matrix_a, Matrix *output);

/* Apply softmax row-wise: each row becomes a probability distribution */
void matrix_softmax(Matrix *matrix);

/* Apply ReLU activation in-place: max(0, x) for each element */
void matrix_relu(Matrix *matrix);

/* Element-wise scaling: output = matrix * scale */
void matrix_scale(const Matrix *matrix, float scale, Matrix *output);

/* Initialize matrix with uniform random values in [-scale, +scale] */
void matrix_init_random(Matrix *matrix, float scale);

/* Initialize matrix with a fixed deterministic pattern for testing.
 * Uses sin(row) * cos(col) * 0.5 to produce reproducible values. */
void matrix_init_fixed(Matrix *matrix);

/* Initialize matrix using Xavier/Glorot initialization.
 * Samples from uniform distribution scaled by sqrt(6 / (fan_in + fan_out)) */
void matrix_init_xavier(Matrix *matrix);

/*
 * ============================================================================
 *  Public API - Positional Encoding
 * ============================================================================
 */

/* Compute sinusoidal positional encoding for a given d_model.
 *
 *  For each position pos and dimension i:
 *    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
 *    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 *
 *  This allows the model to attend to relative positions since
 *  PE(pos+k) can be represented as a linear function of PE(pos).
 */
void compute_positional_encoding(Matrix *positional_encoding, int d_model);

/*
 * ============================================================================
 *  Public API - Layer Normalization
 * ============================================================================
 */

/* Initialize layer norm parameters (gamma=1.0, beta=0.0) */
void layernorm_init(LayerNorm *layer_norm, int size);

/* Free layer norm parameters */
void layernorm_free(LayerNorm *layer_norm);

/* Forward pass: normalize input vector and apply affine transform.
 * Normalizes over the last dimension (feature-wise). */
void layernorm_forward(const LayerNorm *layer_norm, const float *input, float *output, int size);

/*
 * ============================================================================
 *  Public API - Multi-Head Attention
 * ============================================================================
 */

/* Initialize multi-head attention with weight matrices for each head.
 * Each head's W_Q, W_K, W_V are (d_model x head_dim).
 * Output projection W_O is (d_model x d_model). */
void multihead_attention_init(MultiHeadAttention *attention, int d_model, int num_heads);

/* Free all attention weight matrices */
void multihead_attention_free(MultiHeadAttention *attention);

/* Forward pass through multi-head self-attention.
 *
 *  Steps:
 *  1. Project input into Q, K, V for each head
 *  2. Compute attention scores: scores = Q @ K^T / sqrt(head_dim)
 *  3. Apply causal mask if use_causal_mask is true (sets future positions to -1e9)
 *  4. Apply softmax row-wise to get attention weights
 *  5. Compute weighted sum: output = scores @ V
 *  6. Concatenate all head outputs
 *  7. Apply output projection: final = concat @ W_O
 *
 *  input:  (sequence_length x d_model)
 *  output: (sequence_length x d_model)
 */
void multihead_attention_forward(
    const MultiHeadAttention *attention,
    const float *input,
    float *output,
    int sequence_length,
    int d_model,
    int use_causal_mask
);

/*
 * ============================================================================
 *  Public API - Feed-Forward Network
 * ============================================================================
 */

/* Initialize FFN with two linear layers and biases.
 * W_up: (d_model x d_ff), W_down: (d_ff x d_model) */
void feedforward_init(FeedForward *feed_forward, int d_model, int d_ff);

/* Free FFN weight matrices and biases */
void feedforward_free(FeedForward *feed_forward);

/* Forward pass through the position-wise feed-forward network.
 *
 *  For each position independently:
 *    hidden = ReLU(input @ W_up + b_up)
 *    output = hidden @ W_down + b_down
 *
 *  input:  (sequence_length x d_model)
 *  output: (sequence_length x d_model)
 */
void feedforward_forward(const FeedForward *feed_forward, const float *input, float *output, int sequence_length);

/*
 * ============================================================================
 *  Public API - Transformer Block
 * ============================================================================
 */

/* Initialize a transformer decoder block with attention, FFN, and 3 layer norms */
void transformer_block_init(TransformerBlock *block, int d_model, int num_heads, int d_ff);

/* Free all sub-components of the transformer block */
void transformer_block_free(TransformerBlock *block);

/* Forward pass through one transformer decoder block.
 *
 *  Pre-norm residual architecture:
 *    normed = LayerNorm_attn(input)
 *    attn_out = SelfAttention(normed)
 *    x = input + attn_out                          (residual connection)
 *    normed = LayerNorm_ff(x)
 *    ff_out = FFN(normed)
 *    output = x + ff_out                           (residual connection)
 *    output = LayerNorm_final(output)
 */
void transformer_block_forward(
    const TransformerBlock *block,
    const float *input,
    float *output,
    int sequence_length,
    int use_causal_mask
);

/*
 * ============================================================================
 *  Public API - Full Transformer Model
 * ============================================================================
 */

/* Initialize the complete transformer model.
 * Creates token embeddings, positional encodings, all decoder blocks,
 * final layer norm parameters, and the LM head projection matrix. */
void transformer_init(Transformer *model, TransformerConfig config);

/* Free all model parameters and sub-components */
void transformer_free(Transformer *model);

/* Complete forward pass from input tokens to output logits.
 *
 *  Steps:
 *  1. Token Embedding: Lookup learned vectors for each input token
 *  2. Positional Encoding: Add sinusoidal position vectors
 *  3. Transformer Blocks: Pass through all decoder layers sequentially
 *  4. Final LayerNorm: Normalize the final hidden states
 *  5. LM Head: Project from d_model to vocab_size for each position
 *
 *  input_tokens: Array of token IDs (length = sequence_length)
 *  logits:       Output buffer of size (sequence_length * vocab_size)
 *                logits[position * vocab_size + token_id]
 */
void transformer_forward(
    const Transformer *model,
    const int *input_tokens,
    float *logits,
    int sequence_length
);

/* Save trained model weights to a binary file.
 * Writes a header with the config followed by all weight data.
 * Returns 0 on success, -1 on error. */
int transformer_save(const Transformer *model, const char *filepath);

/* Load trained model weights from a binary file.
 * Model must already be initialized with a matching TransformerConfig.
 * Returns 0 on success, -1 on error. */
int transformer_load(Transformer *model, const char *filepath);

/* Sample a token ID from the logits distribution using temperature scaling.
 *
 *  temperature > 1.0: More random/diverse output
 *  temperature = 1.0: Original distribution
 *  temperature < 1.0: More confident/deterministic output
 *
 *  Uses categorical sampling: P(token_i) = exp(logit_i / T) / sum(exp(logits / T))
 */
int transformer_sample(const float *logits, int vocab_size, float temperature);

/*
 * ============================================================================
 *  Implementation Section
 *  All function definitions follow. Only compiled when TRANSFORMER_IMPLEMENTATION
 *  is defined in exactly one translation unit.
 * ============================================================================
 */
#ifdef TRANSFORMER_IMPLEMENTATION

/* ---- Matrix Operations ---- */

Matrix matrix_create(int rows, int cols) {
    Matrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    /* Allocate zero-initialized contiguous memory */
    matrix.data = (float *)calloc(rows * cols, sizeof(float));
    return matrix;
}

void matrix_free(Matrix *matrix) {
    if (matrix->data) {
        free(matrix->data);
        matrix->data = NULL;
    }
    matrix->rows = 0;
    matrix->cols = 0;
}

void matrix_zero(Matrix *matrix) {
    memset(matrix->data, 0, matrix->rows * matrix->cols * sizeof(float));
}

void matrix_fill(Matrix *matrix, float value) {
    for (int i = 0; i < matrix->rows * matrix->cols; i++) {
        matrix->data[i] = value;
    }
}

void matrix_copy(const Matrix *source, Matrix *destination) {
    int size = source->rows * source->cols;
    /* Reallocate destination if it's too small */
    if (destination->rows * destination->cols < size) {
        free(destination->data);
        destination->data = (float *)malloc(size * sizeof(float));
    }
    destination->rows = source->rows;
    destination->cols = source->cols;
    memcpy(destination->data, source->data, size * sizeof(float));
}

void matrix_print(const Matrix *matrix, const char *name) {
    printf("%s (%dx%d):\n", name, matrix->rows, matrix->cols);
    for (int row = 0; row < matrix->rows; row++) {
        for (int col = 0; col < matrix->cols; col++) {
            printf("%8.4f ", matrix->data[row * matrix->cols + col]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrix_add(const Matrix *matrix_a, const Matrix *matrix_b, Matrix *output) {
    int size = matrix_a->rows * matrix_a->cols;
    for (int i = 0; i < size; i++) {
        output->data[i] = matrix_a->data[i] + matrix_b->data[i];
    }
}

/* O(n^3) naive matrix multiplication.
 * For production use, consider blocking or SIMD optimization. */
void matrix_multiply(const Matrix *matrix_a, const Matrix *matrix_b, Matrix *output) {
    for (int row = 0; row < matrix_a->rows; row++) {
        for (int col = 0; col < matrix_b->cols; col++) {
            float accumulation = 0.0f;
            for (int dim = 0; dim < matrix_a->cols; dim++) {
                accumulation += matrix_a->data[row * matrix_a->cols + dim]
                              * matrix_b->data[dim * matrix_b->cols + col];
            }
            output->data[row * output->cols + col] = accumulation;
        }
    }
}

void matrix_transpose(const Matrix *matrix_a, Matrix *output) {
    for (int row = 0; row < matrix_a->rows; row++) {
        for (int col = 0; col < matrix_a->cols; col++) {
            output->data[col * output->cols + row] = matrix_a->data[row * matrix_a->cols + col];
        }
    }
}

void matrix_scale(const Matrix *matrix, float scale, Matrix *output) {
    int size = matrix->rows * matrix->cols;
    for (int i = 0; i < size; i++) {
        output->data[i] = matrix->data[i] * scale;
    }
}

void matrix_relu(Matrix *matrix) {
    int size = matrix->rows * matrix->cols;
    for (int i = 0; i < size; i++) {
        if (matrix->data[i] < 0.0f) matrix->data[i] = 0.0f;
    }
}

/* Numerically stable softmax: subtracts max before exp to prevent overflow */
void matrix_softmax(Matrix *matrix) {
    for (int row = 0; row < matrix->rows; row++) {
        /* Find max for numerical stability */
        float max_value = matrix->data[row * matrix->cols];
        for (int col = 1; col < matrix->cols; col++) {
            if (matrix->data[row * matrix->cols + col] > max_value) {
                max_value = matrix->data[row * matrix->cols + col];
            }
        }
        /* Compute exp(x - max) and sum */
        float sum = 0.0f;
        for (int col = 0; col < matrix->cols; col++) {
            matrix->data[row * matrix->cols + col] = expf(matrix->data[row * matrix->cols + col] - max_value);
            sum += matrix->data[row * matrix->cols + col];
        }
        /* Normalize to get probabilities */
        for (int col = 0; col < matrix->cols; col++) {
            matrix->data[row * matrix->cols + col] /= sum;
        }
    }
}

void matrix_init_random(Matrix *matrix, float scale) {
    for (int i = 0; i < matrix->rows * matrix->cols; i++) {
        matrix->data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f * scale - scale;
    }
}

/* Deterministic initialization for reproducible testing.
 * Pattern: sin(row * 0.1) * cos(col * 0.1) * 0.5 */
void matrix_init_fixed(Matrix *matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    for (int i = 0; i < rows * cols; i++) {
        int r = i / cols;
        int c = i % cols;
        matrix->data[i] = sinf((float)(r + 1) * 0.1f) * cosf((float)(c + 1) * 0.1f) * 0.5f;
    }
}

/* Xavier/Glorot uniform initialization.
 * Preserves variance across layers by scaling with sqrt(6 / (fan_in + fan_out)) */
void matrix_init_xavier(Matrix *matrix) {
    float scale = sqrtf(6.0f / (matrix->rows + matrix->cols));
    matrix_init_random(matrix, scale);
}

/* ---- Positional Encoding ---- */

/* Sinusoidal positional encoding as described in the original paper.
 *
 *  Uses alternating sin/cos functions at different frequency scales so that
 *  the model can learn to attend to relative positions. The wavelength for
 *  dimension i forms a geometric progression from 2*pi to 10000*2*pi.
 */
void compute_positional_encoding(Matrix *positional_encoding, int d_model) {
    for (int position = 0; position < positional_encoding->rows; position++) {
        for (int dim = 0; dim < d_model; dim += 2) {
            /* Compute the divisor term: 10000^(2i/d_model) */
            float divisor_term = expf(-((float)dim / d_model) * logf(10000.0f));
            /* Even dimensions: sine function */
            positional_encoding->data[position * d_model + dim] = sinf(position * divisor_term);
            /* Odd dimensions: cosine function */
            if (dim + 1 < d_model) {
                positional_encoding->data[position * d_model + dim + 1] = cosf(position * divisor_term);
            }
        }
    }
}

/* ---- Layer Normalization ---- */

void layernorm_init(LayerNorm *layer_norm, int size) {
    layer_norm->size = size;
    layer_norm->gamma = (float *)malloc(size * sizeof(float));
    layer_norm->beta = (float *)malloc(size * sizeof(float));
    /* Initialize gamma to 1.0 (identity scale) and beta to 0.0 (no shift) */
    for (int i = 0; i < size; i++) {
        layer_norm->gamma[i] = 1.0f;
        layer_norm->beta[i] = 0.0f;
    }
}

void layernorm_free(LayerNorm *layer_norm) {
    free(layer_norm->gamma);
    free(layer_norm->beta);
    layer_norm->gamma = NULL;
    layer_norm->beta = NULL;
}

/* Layer normalization forward pass.
 *
 *  Computes mean and variance over the feature dimension, then applies:
 *    x_hat = (x - mean) / sqrt(variance + epsilon)
 *    output = gamma * x_hat + beta
 *
 *  epsilon = 1e-5 prevents division by zero.
 */
void layernorm_forward(const LayerNorm *layer_norm, const float *input, float *output, int size) {
    /* Compute mean */
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= size;

    /* Compute variance */
    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        float difference = input[i] - mean;
        variance += difference * difference;
    }
    variance /= size;
    float standard_deviation = sqrtf(variance + 1e-5f);

    /* Normalize and apply affine transform */
    for (int i = 0; i < size; i++) {
        output[i] = layer_norm->gamma[i] * ((input[i] - mean) / standard_deviation) + layer_norm->beta[i];
    }
}

/* ---- Multi-Head Attention ---- */

void multihead_attention_init(MultiHeadAttention *attention, int d_model, int num_heads) {
    attention->num_heads = num_heads;
    attention->d_model = d_model;
    attention->head_dim = d_model / num_heads;

    /* Allocate and initialize weight matrices for each attention head */
    attention->heads = (AttentionHeadWeights *)malloc(num_heads * sizeof(AttentionHeadWeights));
    for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        attention->heads[head_idx].weight_query = matrix_create(d_model, attention->head_dim);
        attention->heads[head_idx].weight_key = matrix_create(d_model, attention->head_dim);
        attention->heads[head_idx].weight_value = matrix_create(d_model, attention->head_dim);
        matrix_init_fixed(&attention->heads[head_idx].weight_query);
        matrix_init_fixed(&attention->heads[head_idx].weight_key);
        matrix_init_fixed(&attention->heads[head_idx].weight_value);
    }

    /* Output projection that maps concatenated heads back to d_model */
    attention->weight_output = matrix_create(d_model, d_model);
    matrix_init_fixed(&attention->weight_output);

    /* KV cache pointers (reserved for autoregressive inference optimization) */
    attention->query_cache = NULL;
    attention->key_cache = NULL;
    attention->value_cache = NULL;
}

void multihead_attention_free(MultiHeadAttention *attention) {
    for (int head_idx = 0; head_idx < attention->num_heads; head_idx++) {
        matrix_free(&attention->heads[head_idx].weight_query);
        matrix_free(&attention->heads[head_idx].weight_key);
        matrix_free(&attention->heads[head_idx].weight_value);
    }
    free(attention->heads);
    matrix_free(&attention->weight_output);
    free(attention->query_cache);
    free(attention->key_cache);
    free(attention->value_cache);
}

void multihead_attention_forward(
    const MultiHeadAttention *attention,
    const float *input,
    float *output,
    int sequence_length,
    int d_model,
    int use_causal_mask
) {
    int head_dim = attention->head_dim;
    int num_heads = attention->num_heads;
    /* Scaling factor prevents dot products from growing too large,
     * which would push softmax into regions with tiny gradients */
    float scaling_factor = 1.0f / sqrtf((float)head_dim);

    /* Allocate Q, K, V buffers: one vector per head per position */
    int head_vector_size = sequence_length * head_dim;
    float *query = (float *)calloc(num_heads * head_vector_size, sizeof(float));
    float *key = (float *)calloc(num_heads * head_vector_size, sizeof(float));
    float *value = (float *)calloc(num_heads * head_vector_size, sizeof(float));

    /* Step 1: Project input into Query, Key, Value for each head
     * Q_h = input @ W_Q_h, K_h = input @ W_K_h, V_h = input @ W_V_h */
    for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        for (int row = 0; row < sequence_length; row++) {
            for (int col = 0; col < d_model; col++) {
                float input_value = input[row * d_model + col];
                for (int dim = 0; dim < head_dim; dim++) {
                    query[head_idx * head_vector_size + row * head_dim + dim] += input_value * attention->heads[head_idx].weight_query.data[col * head_dim + dim];
                    key[head_idx * head_vector_size + row * head_dim + dim] += input_value * attention->heads[head_idx].weight_key.data[col * head_dim + dim];
                    value[head_idx * head_vector_size + row * head_dim + dim] += input_value * attention->heads[head_idx].weight_value.data[col * head_dim + dim];
                }
            }
        }
    }

    /* Buffer for each head's attention output */
    float *head_outputs = (float *)calloc(num_heads * sequence_length * head_dim, sizeof(float));

    /* Step 2-5: Compute attention for each head independently */
    for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        /* Step 2: Compute attention scores = Q @ K^T / sqrt(d_k) */
        int score_matrix_size = sequence_length * sequence_length;
        float *attention_scores = (float *)calloc(score_matrix_size, sizeof(float));

        for (int row = 0; row < sequence_length; row++) {
            for (int col = 0; col < sequence_length; col++) {
                float dot_product = 0.0f;
                for (int dim = 0; dim < head_dim; dim++) {
                    dot_product += query[head_idx * head_vector_size + row * head_dim + dim]
                                 * key[head_idx * head_vector_size + col * head_dim + dim];
                }
                attention_scores[row * sequence_length + col] = dot_product * scaling_factor;

                /* Step 3: Apply causal mask - prevent attending to future positions.
                 * Setting to -1e9 ensures softmax produces ~0 probability. */
                if (use_causal_mask && col > row) {
                    attention_scores[row * sequence_length + col] = -1e9f;
                }
            }
        }

        /* Step 4: Apply softmax row-wise to get attention weights */
        for (int row = 0; row < sequence_length; row++) {
            float max_score = attention_scores[row * sequence_length];
            for (int col = 1; col < sequence_length; col++) {
                if (attention_scores[row * sequence_length + col] > max_score)
                    max_score = attention_scores[row * sequence_length + col];
            }
            float sum = 0.0f;
            for (int col = 0; col < sequence_length; col++) {
                attention_scores[row * sequence_length + col] = expf(attention_scores[row * sequence_length + col] - max_score);
                sum += attention_scores[row * sequence_length + col];
            }
            for (int col = 0; col < sequence_length; col++) {
                attention_scores[row * sequence_length + col] /= sum;
            }
        }

        /* Step 5: Compute weighted sum of values: output = scores @ V */
        for (int row = 0; row < sequence_length; row++) {
            for (int dim = 0; dim < head_dim; dim++) {
                float weighted_sum = 0.0f;
                for (int col = 0; col < sequence_length; col++) {
                    weighted_sum += attention_scores[row * sequence_length + col]
                                  * value[head_idx * head_vector_size + col * head_dim + dim];
                }
                head_outputs[head_idx * sequence_length * head_dim + row * head_dim + dim] = weighted_sum;
            }
        }

        free(attention_scores);
    }

    /* Step 6: Concatenate all head outputs into a single matrix
     * Each head contributes head_dim columns, total = num_heads * head_dim = d_model */
    float *concatenated = (float *)calloc(sequence_length * d_model, sizeof(float));
    for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        for (int row = 0; row < sequence_length; row++) {
            for (int dim = 0; dim < head_dim; dim++) {
                concatenated[row * d_model + head_idx * head_dim + dim] = head_outputs[head_idx * sequence_length * head_dim + row * head_dim + dim];
            }
        }
    }

    /* Step 7: Apply output projection: final_output = concatenated @ W_O */
    for (int row = 0; row < sequence_length; row++) {
        for (int col = 0; col < d_model; col++) {
            float accumulation = 0.0f;
            for (int dim = 0; dim < d_model; dim++) {
                accumulation += concatenated[row * d_model + dim] * attention->weight_output.data[dim * d_model + col];
            }
            output[row * d_model + col] = accumulation;
        }
    }

    /* Free temporary buffers */
    free(query);
    free(key);
    free(value);
    free(head_outputs);
    free(concatenated);
}

/* ---- Feed-Forward Network ---- */

void feedforward_init(FeedForward *feed_forward, int d_model, int d_ff) {
    feed_forward->d_model = d_model;
    feed_forward->d_ff = d_ff;
    feed_forward->weight_up = matrix_create(d_model, d_ff);
    feed_forward->bias_up = matrix_create(1, d_ff);
    feed_forward->weight_down = matrix_create(d_ff, d_model);
    feed_forward->bias_down = matrix_create(1, d_model);
    matrix_init_fixed(&feed_forward->weight_up);
    matrix_init_fixed(&feed_forward->weight_down);
}

void feedforward_free(FeedForward *feed_forward) {
    matrix_free(&feed_forward->weight_up);
    matrix_free(&feed_forward->bias_up);
    matrix_free(&feed_forward->weight_down);
    matrix_free(&feed_forward->bias_down);
}

void feedforward_forward(const FeedForward *feed_forward, const float *input, float *output, int sequence_length) {
    int d_model = feed_forward->d_model;
    int d_ff = feed_forward->d_ff;

    /* Temporary buffer for the expanded hidden state */
    float *hidden_state = (float *)calloc(sequence_length * d_ff, sizeof(float));

    /* Step 1: Linear projection up: hidden = input @ W_up + b_up */
    for (int row = 0; row < sequence_length; row++) {
        for (int col = 0; col < d_ff; col++) {
            float accumulation = feed_forward->bias_up.data[col];
            for (int dim = 0; dim < d_model; dim++) {
                accumulation += input[row * d_model + dim] * feed_forward->weight_up.data[dim * d_ff + col];
            }
            hidden_state[row * d_ff + col] = accumulation;
        }
    }

    /* Step 2: ReLU activation: hidden = max(0, hidden) */
    for (int i = 0; i < sequence_length * d_ff; i++) {
        if (hidden_state[i] < 0.0f) hidden_state[i] = 0.0f;
    }

    /* Step 3: Linear projection down: output = hidden @ W_down + b_down */
    for (int row = 0; row < sequence_length; row++) {
        for (int col = 0; col < d_model; col++) {
            float accumulation = feed_forward->bias_down.data[col];
            for (int dim = 0; dim < d_ff; dim++) {
                accumulation += hidden_state[row * d_ff + dim] * feed_forward->weight_down.data[dim * d_model + col];
            }
            output[row * d_model + col] = accumulation;
        }
    }

    free(hidden_state);
}

/* ---- Transformer Block ---- */

void transformer_block_init(TransformerBlock *block, int d_model, int num_heads, int d_ff) {
    block->d_model = d_model;
    multihead_attention_init(&block->self_attention, d_model, num_heads);
    feedforward_init(&block->feed_forward, d_model, d_ff);
    layernorm_init(&block->layer_norm_attn, d_model);
    layernorm_init(&block->layer_norm_ff, d_model);
}

void transformer_block_free(TransformerBlock *block) {
    multihead_attention_free(&block->self_attention);
    feedforward_free(&block->feed_forward);
    layernorm_free(&block->layer_norm_attn);
    layernorm_free(&block->layer_norm_ff);
}

void transformer_block_forward(
    const TransformerBlock *block,
    const float *input,
    float *output,
    int sequence_length,
    int use_causal_mask
) {
    int d_model = block->d_model;
    int vector_size = sequence_length * d_model;

    /* Allocate temporary buffers for intermediate computations */
    float *normalized = (float *)malloc(vector_size * sizeof(float));
    float *attention_output = (float *)malloc(vector_size * sizeof(float));
    float *feed_forward_output = (float *)malloc(vector_size * sizeof(float));

    /* --- Sub-layer 1: Self-Attention with residual connection --- */

    /* Pre-norm: normalize input before attention */
    for (int pos = 0; pos < sequence_length; pos++) {
        layernorm_forward(&block->layer_norm_attn, &input[pos * d_model], &normalized[pos * d_model], d_model);
    }

    /* Forward through multi-head self-attention */
    multihead_attention_forward(&block->self_attention, normalized, attention_output, sequence_length, d_model, use_causal_mask);

    /* Residual connection: output = input + attention_output */
    for (int i = 0; i < vector_size; i++) {
        output[i] = input[i] + attention_output[i];
    }

    /* --- Sub-layer 2: Feed-Forward Network with residual connection --- */

    /* Pre-norm: normalize the attention output before FFN */
    for (int pos = 0; pos < sequence_length; pos++) {
        layernorm_forward(&block->layer_norm_ff, &output[pos * d_model], &normalized[pos * d_model], d_model);
    }

    /* Forward through position-wise FFN */
    feedforward_forward(&block->feed_forward, normalized, feed_forward_output, sequence_length);

    /* Residual connection: output = output + feed_forward_output */
    for (int i = 0; i < vector_size; i++) {
        output[i] = output[i] + feed_forward_output[i];
    }

    /* Free temporary buffers */
    free(normalized);
    free(attention_output);
    free(feed_forward_output);
}

/* ---- Full Transformer Model ---- */

void transformer_init(Transformer *model, TransformerConfig config) {
    model->config = config;

    /* Token embedding table: maps each token ID to a dense vector */
    model->token_embeddings = matrix_create(config.vocab_size, config.d_model);
    matrix_init_fixed(&model->token_embeddings);

    /* Sinusoidal positional encodings for each position up to max_seq_len */
    model->positional_embeddings = matrix_create(config.max_seq_len, config.d_model);
    compute_positional_encoding(&model->positional_embeddings, config.d_model);

    /* Stack of transformer decoder blocks */
    model->blocks = (TransformerBlock *)malloc(config.num_layers * sizeof(TransformerBlock));
    for (int layer_idx = 0; layer_idx < config.num_layers; layer_idx++) {
        transformer_block_init(&model->blocks[layer_idx], config.d_model, config.num_heads, config.d_ff);
    }

    /* Final layer normalization parameters (gamma=1.0, beta=0.0) */
    model->final_ln_gamma = matrix_create(1, config.d_model);
    model->final_ln_beta = matrix_create(1, config.d_model);
    matrix_fill(&model->final_ln_gamma, 1.0f);

    /* Language model head: projects from d_model back to vocabulary size */
    model->lm_head = matrix_create(config.d_model, config.vocab_size);
    matrix_init_fixed(&model->lm_head);
}

void transformer_free(Transformer *model) {
    matrix_free(&model->token_embeddings);
    matrix_free(&model->positional_embeddings);
    for (int layer_idx = 0; layer_idx < model->config.num_layers; layer_idx++) {
        transformer_block_free(&model->blocks[layer_idx]);
    }
    free(model->blocks);
    matrix_free(&model->final_ln_gamma);
    matrix_free(&model->final_ln_beta);
    matrix_free(&model->lm_head);
}

void transformer_forward(
    const Transformer *model,
    const int *input_tokens,
    float *logits,
    int sequence_length
) {
    int d_model = model->config.d_model;
    int vector_size = sequence_length * d_model;

    /* Step 1: Token Embedding + Positional Encoding
     * hidden_states[i] = token_embedding[tokens[i]] + positional_encoding[i] */
    float *hidden_states = (float *)malloc(vector_size * sizeof(float));
    for (int position = 0; position < sequence_length; position++) {
        int token = input_tokens[position];
        for (int dim = 0; dim < d_model; dim++) {
            hidden_states[position * d_model + dim] = model->token_embeddings.data[token * d_model + dim]
                               + model->positional_embeddings.data[position * d_model + dim];
        }
    }

    /* Step 2: Forward through all transformer blocks sequentially */
    float *block_output = (float *)malloc(vector_size * sizeof(float));
    for (int layer_idx = 0; layer_idx < model->config.num_layers; layer_idx++) {
        transformer_block_forward(&model->blocks[layer_idx], hidden_states, block_output, sequence_length, 1);
        /* Copy block output back to hidden_states for next layer */
        memcpy(hidden_states, block_output, vector_size * sizeof(float));
    }
    free(block_output);

    /* Step 3: Final layer normalization */
    float *normalized = (float *)malloc(vector_size * sizeof(float));
    LayerNorm final_layer_norm;
    final_layer_norm.size = d_model;
    final_layer_norm.gamma = model->final_ln_gamma.data;
    final_layer_norm.beta = model->final_ln_beta.data;
    for (int pos = 0; pos < sequence_length; pos++) {
        layernorm_forward(&final_layer_norm, &hidden_states[pos * d_model], &normalized[pos * d_model], d_model);
    }

    /* Step 4: LM Head projection - convert hidden states to vocabulary logits
     * logits[position, token_id] = normalized[position] @ lm_head[:, token_id] */
    int vocab_size = model->config.vocab_size;
    for (int position = 0; position < sequence_length; position++) {
        for (int token_idx = 0; token_idx < vocab_size; token_idx++) {
            float accumulation = 0.0f;
            for (int dim = 0; dim < d_model; dim++) {
                accumulation += normalized[position * d_model + dim] * model->lm_head.data[dim * vocab_size + token_idx];
            }
            logits[position * vocab_size + token_idx] = accumulation;
        }
    }

    free(hidden_states);
    free(normalized);
}

/* Temperature-scaled categorical sampling from logits.
 *
 *  The temperature parameter controls the randomness of sampling:
 *  - High temperature (T > 1): flattens distribution, more diverse output
 *  - Low temperature (T < 1): sharpens distribution, more deterministic
 *  - T = 1: samples from the raw softmax distribution
 *
 *  Uses the "logits - max" trick for numerical stability before exp().
 */
int transformer_sample(const float *logits, int vocab_size, float temperature) {
    /* Convert logits to probabilities via softmax with temperature */
    float *probabilities = (float *)malloc(vocab_size * sizeof(float));
    float max_value = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_value) max_value = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probabilities[i] = expf((logits[i] - max_value) / temperature);
        sum += probabilities[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        probabilities[i] /= sum;
    }

    /* Sample from the categorical distribution using inverse transform sampling */
    float random_value = (float)rand() / (float)RAND_MAX;
    float cumulative_probability = 0.0f;
    int sampled_token = vocab_size - 1;
    for (int i = 0; i < vocab_size; i++) {
        cumulative_probability += probabilities[i];
        if (random_value <= cumulative_probability) {
            sampled_token = i;
            break;
        }
    }

    free(probabilities);
    return sampled_token;
}

/* ---- Save / Load ---- */

#define TRANSFORMER_MAGIC "TMFM"

static int fwrite_matrix(FILE *f, const Matrix *m) {
    size_t n = (size_t)m->rows * m->cols;
    return fwrite(m->data, sizeof(float), n, f) == n ? 0 : -1;
}

static int fread_matrix(FILE *f, Matrix *m) {
    size_t n = (size_t)m->rows * m->cols;
    return fread(m->data, sizeof(float), n, f) == n ? 0 : -1;
}

static int fwrite_floats(FILE *f, const float *data, int n) {
    return fwrite(data, sizeof(float), (size_t)n, f) == (size_t)n ? 0 : -1;
}

static int fread_floats(FILE *f, float *data, int n) {
    return fread(data, sizeof(float), (size_t)n, f) == (size_t)n ? 0 : -1;
}

int transformer_save(const Transformer *model, const char *filepath) {
    FILE *f = fopen(filepath, "wb");
    if (!f) return -1;

    TransformerConfig cfg = model->config;

    /* Write header */
    fwrite(TRANSFORMER_MAGIC, 1, 4, f);
    fwrite(&cfg.vocab_size, sizeof(int), 1, f);
    fwrite(&cfg.d_model, sizeof(int), 1, f);
    fwrite(&cfg.num_heads, sizeof(int), 1, f);
    fwrite(&cfg.num_layers, sizeof(int), 1, f);
    fwrite(&cfg.d_ff, sizeof(int), 1, f);
    fwrite(&cfg.max_seq_len, sizeof(int), 1, f);

    /* Token and positional embeddings */
    fwrite_matrix(f, &model->token_embeddings);
    fwrite_matrix(f, &model->positional_embeddings);

    /* Transformer blocks */
    for (int l = 0; l < cfg.num_layers; l++) {
        for (int h = 0; h < cfg.num_heads; h++) {
            fwrite_matrix(f, &model->blocks[l].self_attention.heads[h].weight_query);
            fwrite_matrix(f, &model->blocks[l].self_attention.heads[h].weight_key);
            fwrite_matrix(f, &model->blocks[l].self_attention.heads[h].weight_value);
        }
        fwrite_matrix(f, &model->blocks[l].self_attention.weight_output);
        fwrite_matrix(f, &model->blocks[l].feed_forward.weight_up);
        fwrite_floats(f, model->blocks[l].feed_forward.bias_up.data, cfg.d_ff);
        fwrite_matrix(f, &model->blocks[l].feed_forward.weight_down);
        fwrite_floats(f, model->blocks[l].feed_forward.bias_down.data, cfg.d_model);
        fwrite_floats(f, model->blocks[l].layer_norm_attn.gamma, cfg.d_model);
        fwrite_floats(f, model->blocks[l].layer_norm_attn.beta, cfg.d_model);
        fwrite_floats(f, model->blocks[l].layer_norm_ff.gamma, cfg.d_model);
        fwrite_floats(f, model->blocks[l].layer_norm_ff.beta, cfg.d_model);
    }

    /* Final layer norm and LM head */
    fwrite_floats(f, model->final_ln_gamma.data, cfg.d_model);
    fwrite_floats(f, model->final_ln_beta.data, cfg.d_model);
    fwrite_matrix(f, &model->lm_head);

    fclose(f);
    return 0;
}

int transformer_load(Transformer *model, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return -1;

    TransformerConfig cfg = model->config;

    /* Read and verify header */
    char magic[4];
    int vcb, dm, nh, nl, df, msl;
    fread(magic, 1, 4, f);
    fread(&vcb, sizeof(int), 1, f);
    fread(&dm, sizeof(int), 1, f);
    fread(&nh, sizeof(int), 1, f);
    fread(&nl, sizeof(int), 1, f);
    fread(&df, sizeof(int), 1, f);
    fread(&msl, sizeof(int), 1, f);

    if (memcmp(magic, TRANSFORMER_MAGIC, 4) != 0 ||
        vcb != cfg.vocab_size || dm != cfg.d_model || nh != cfg.num_heads ||
        nl != cfg.num_layers || df != cfg.d_ff || msl != cfg.max_seq_len) {
        fclose(f);
        return -1;
    }

    /* Token and positional embeddings */
    fread_matrix(f, &model->token_embeddings);
    fread_matrix(f, &model->positional_embeddings);

    /* Transformer blocks */
    for (int l = 0; l < cfg.num_layers; l++) {
        for (int h = 0; h < cfg.num_heads; h++) {
            fread_matrix(f, &model->blocks[l].self_attention.heads[h].weight_query);
            fread_matrix(f, &model->blocks[l].self_attention.heads[h].weight_key);
            fread_matrix(f, &model->blocks[l].self_attention.heads[h].weight_value);
        }
        fread_matrix(f, &model->blocks[l].self_attention.weight_output);
        fread_matrix(f, &model->blocks[l].feed_forward.weight_up);
        fread_floats(f, model->blocks[l].feed_forward.bias_up.data, cfg.d_ff);
        fread_matrix(f, &model->blocks[l].feed_forward.weight_down);
        fread_floats(f, model->blocks[l].feed_forward.bias_down.data, cfg.d_model);
        fread_floats(f, model->blocks[l].layer_norm_attn.gamma, cfg.d_model);
        fread_floats(f, model->blocks[l].layer_norm_attn.beta, cfg.d_model);
        fread_floats(f, model->blocks[l].layer_norm_ff.gamma, cfg.d_model);
        fread_floats(f, model->blocks[l].layer_norm_ff.beta, cfg.d_model);
    }

    /* Final layer norm and LM head */
    fread_floats(f, model->final_ln_gamma.data, cfg.d_model);
    fread_floats(f, model->final_ln_beta.data, cfg.d_model);
    fread_matrix(f, &model->lm_head);

    fclose(f);
    return 0;
}

#endif /* TRANSFORMER_IMPLEMENTATION */
#endif /* TRANSFORMER_H */
