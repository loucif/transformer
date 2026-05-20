#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

/*
 * ============================================================================
 *  Tiny Character-Level Transformer Trainer
 * ============================================================================
 *
 *  A minimal training loop that learns to predict the next character
 *  from a small text sample using SGD.
 *
 *  Architecture (tiny for fast CPU training):
 *  - Vocabulary: 27 characters (a-z + space + newline)
 *  - d_model: 16
 *  - num_heads: 2 (head_dim = 8)
 *  - num_layers: 1
 *  - d_ff: 32
 *  - max_seq_len: 16
 *
 *  Usage:
 *    gcc -o train train.c -lm -O2
 *    ./train
 */

#define CHAR_VOCAB_SIZE 28
#define CHAR_D_MODEL 16
#define CHAR_NUM_HEADS 2
#define CHAR_NUM_LAYERS 1
#define CHAR_D_FF 32
#define CHAR_MAX_SEQ_LEN 16
#define LEARNING_RATE 0.01f
#define NUM_EPOCHS 500
#define EVAL_EVERY 50

/* Character vocabulary: a-z (0-25), space (26), newline (27) */
const char char_vocab[CHAR_VOCAB_SIZE] = {
    'a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z',
    ' ','\n'
};

int char_to_id(char c) {
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c == ' ') return 26;
    if (c == '\n') return 27;
    return 27; /* unknown -> newline */
}

char id_to_char(int id) {
    if (id >= 0 && id < CHAR_VOCAB_SIZE) return char_vocab[id];
    return '?';
}

/* Training text */
const char *training_text =
    "hello world this is a test of the character level transformer model "
    "the quick brown fox jumps over the lazy dog "
    "attention is all you need in modern language models "
    "deep learning with neural networks is powerful ";

/*
 * ============================================================================
 *  Gradient Storage - mirrors model parameters
 * ============================================================================
 */

typedef struct {
    Matrix weight_query;
    Matrix weight_key;
    Matrix weight_value;
} AttentionHeadGrads;

typedef struct {
    int num_heads;
    int d_model;
    AttentionHeadGrads *heads;
    Matrix weight_output;
} MultiHeadAttentionGrads;

typedef struct {
    Matrix weight_up;
    Matrix bias_up;
    Matrix weight_down;
    Matrix bias_down;
} FeedForwardGrads;

typedef struct {
    float *gamma;
    float *beta;
    int size;
} LayerNormGrads;

typedef struct {
    MultiHeadAttentionGrads self_attention;
    FeedForwardGrads feed_forward;
    LayerNormGrads layer_norm_attn;
    LayerNormGrads layer_norm_ff;
    LayerNormGrads layer_norm_final;
} TransformerBlockGrads;

typedef struct {
    Matrix token_embeddings;
    Matrix positional_embeddings;
    TransformerBlockGrads *blocks;
    Matrix final_ln_gamma;
    Matrix final_ln_beta;
    Matrix lm_head;
} TransformerGrads;

/*
 * ============================================================================
 *  Gradient Helpers
 * ============================================================================
 */

void grad_init_matrix(Matrix *grad, int rows, int cols) {
    *grad = matrix_create(rows, cols);
}

void grad_zero_matrix(Matrix *grad) {
    matrix_zero(grad);
}

void grad_add_scaled(Matrix *grad, const float *values, float scale, int size) {
    for (int i = 0; i < size; i++) {
        grad->data[i] += values[i] * scale;
    }
}

void grad_init_layernorm(LayerNormGrads *grad, int size) {
    grad->size = size;
    grad->gamma = (float *)calloc(size, sizeof(float));
    grad->beta = (float *)calloc(size, sizeof(float));
}

void grad_zero_layernorm(LayerNormGrads *grad) {
    for (int i = 0; i < grad->size; i++) {
        grad->gamma[i] = 0.0f;
        grad->beta[i] = 0.0f;
    }
}

void grad_init_attention(MultiHeadAttentionGrads *grad, int d_model, int num_heads) {
    grad->num_heads = num_heads;
    grad->d_model = d_model;
    int head_dim = d_model / num_heads;
    grad->heads = (AttentionHeadGrads *)malloc(num_heads * sizeof(AttentionHeadGrads));
    for (int h = 0; h < num_heads; h++) {
        grad_init_matrix(&grad->heads[h].weight_query, d_model, head_dim);
        grad_init_matrix(&grad->heads[h].weight_key, d_model, head_dim);
        grad_init_matrix(&grad->heads[h].weight_value, d_model, head_dim);
    }
    grad_init_matrix(&grad->weight_output, d_model, d_model);
}

void grad_zero_attention(MultiHeadAttentionGrads *grad) {
    for (int h = 0; h < grad->num_heads; h++) {
        grad_zero_matrix(&grad->heads[h].weight_query);
        grad_zero_matrix(&grad->heads[h].weight_key);
        grad_zero_matrix(&grad->heads[h].weight_value);
    }
    grad_zero_matrix(&grad->weight_output);
}

void grad_init_ffn(FeedForwardGrads *grad, int d_model, int d_ff) {
    grad_init_matrix(&grad->weight_up, d_model, d_ff);
    grad_init_matrix(&grad->bias_up, 1, d_ff);
    grad_init_matrix(&grad->weight_down, d_ff, d_model);
    grad_init_matrix(&grad->bias_down, 1, d_model);
}

void grad_zero_ffn(FeedForwardGrads *grad) {
    grad_zero_matrix(&grad->weight_up);
    grad_zero_matrix(&grad->bias_up);
    grad_zero_matrix(&grad->weight_down);
    grad_zero_matrix(&grad->bias_down);
}

void grad_init_block(TransformerBlockGrads *grad, int d_model, int num_heads, int d_ff) {
    grad_init_attention(&grad->self_attention, d_model, num_heads);
    grad_init_ffn(&grad->feed_forward, d_model, d_ff);
    grad_init_layernorm(&grad->layer_norm_attn, d_model);
    grad_init_layernorm(&grad->layer_norm_ff, d_model);
    grad_init_layernorm(&grad->layer_norm_final, d_model);
}

void grad_zero_block(TransformerBlockGrads *grad) {
    grad_zero_attention(&grad->self_attention);
    grad_zero_ffn(&grad->feed_forward);
    grad_zero_layernorm(&grad->layer_norm_attn);
    grad_zero_layernorm(&grad->layer_norm_ff);
    grad_zero_layernorm(&grad->layer_norm_final);
}

void grad_init_model(TransformerGrads *grad, TransformerConfig config) {
    grad_init_matrix(&grad->token_embeddings, config.vocab_size, config.d_model);
    grad_init_matrix(&grad->positional_embeddings, config.max_seq_len, config.d_model);
    grad->blocks = (TransformerBlockGrads *)malloc(config.num_layers * sizeof(TransformerBlockGrads));
    for (int l = 0; l < config.num_layers; l++) {
        grad_init_block(&grad->blocks[l], config.d_model, config.num_heads, config.d_ff);
    }
    grad_init_matrix(&grad->final_ln_gamma, 1, config.d_model);
    grad_init_matrix(&grad->final_ln_beta, 1, config.d_model);
    grad_init_matrix(&grad->lm_head, config.d_model, config.vocab_size);
}

void grad_zero_model(TransformerGrads *grad, TransformerConfig config) {
    grad_zero_matrix(&grad->token_embeddings);
    grad_zero_matrix(&grad->positional_embeddings);
    for (int l = 0; l < config.num_layers; l++) {
        grad_zero_block(&grad->blocks[l]);
    }
    grad_zero_matrix(&grad->final_ln_gamma);
    grad_zero_matrix(&grad->final_ln_beta);
    grad_zero_matrix(&grad->lm_head);
}

void grad_free_model(TransformerGrads *grad, TransformerConfig config) {
    matrix_free(&grad->token_embeddings);
    matrix_free(&grad->positional_embeddings);
    for (int l = 0; l < config.num_layers; l++) {
        for (int h = 0; h < grad->blocks[l].self_attention.num_heads; h++) {
            matrix_free(&grad->blocks[l].self_attention.heads[h].weight_query);
            matrix_free(&grad->blocks[l].self_attention.heads[h].weight_key);
            matrix_free(&grad->blocks[l].self_attention.heads[h].weight_value);
        }
        free(grad->blocks[l].self_attention.heads);
        matrix_free(&grad->blocks[l].self_attention.weight_output);
        matrix_free(&grad->blocks[l].feed_forward.weight_up);
        matrix_free(&grad->blocks[l].feed_forward.bias_up);
        matrix_free(&grad->blocks[l].feed_forward.weight_down);
        matrix_free(&grad->blocks[l].feed_forward.bias_down);
        free(grad->blocks[l].layer_norm_attn.gamma);
        free(grad->blocks[l].layer_norm_attn.beta);
        free(grad->blocks[l].layer_norm_ff.gamma);
        free(grad->blocks[l].layer_norm_ff.beta);
        free(grad->blocks[l].layer_norm_final.gamma);
        free(grad->blocks[l].layer_norm_final.beta);
    }
    free(grad->blocks);
    matrix_free(&grad->final_ln_gamma);
    matrix_free(&grad->final_ln_beta);
    matrix_free(&grad->lm_head);
}

/*
 * ============================================================================
 *  SGD Optimizer
 * ============================================================================
 */

void sgd_update_matrix(Matrix *param, const Matrix *grad, float lr) {
    int size = param->rows * param->cols;
    for (int i = 0; i < size; i++) {
        param->data[i] -= lr * grad->data[i];
    }
}

void sgd_update_layernorm(LayerNorm *param, const LayerNormGrads *grad, float lr) {
    for (int i = 0; i < param->size; i++) {
        param->gamma[i] -= lr * grad->gamma[i];
        param->beta[i] -= lr * grad->beta[i];
    }
}

void sgd_update_attention(MultiHeadAttention *param, const MultiHeadAttentionGrads *grad, float lr) {
    for (int h = 0; h < param->num_heads; h++) {
        sgd_update_matrix(&param->heads[h].weight_query, &grad->heads[h].weight_query, lr);
        sgd_update_matrix(&param->heads[h].weight_key, &grad->heads[h].weight_key, lr);
        sgd_update_matrix(&param->heads[h].weight_value, &grad->heads[h].weight_value, lr);
    }
    sgd_update_matrix(&param->weight_output, &grad->weight_output, lr);
}

void sgd_update_ffn(FeedForward *param, const FeedForwardGrads *grad, float lr) {
    sgd_update_matrix(&param->weight_up, &grad->weight_up, lr);
    sgd_update_matrix(&param->bias_up, &grad->bias_up, lr);
    sgd_update_matrix(&param->weight_down, &grad->weight_down, lr);
    sgd_update_matrix(&param->bias_down, &grad->bias_down, lr);
}

void sgd_update_block(TransformerBlock *param, const TransformerBlockGrads *grad, float lr) {
    sgd_update_attention(&param->self_attention, &grad->self_attention, lr);
    sgd_update_ffn(&param->feed_forward, &grad->feed_forward, lr);
    sgd_update_layernorm(&param->layer_norm_attn, &grad->layer_norm_attn, lr);
    sgd_update_layernorm(&param->layer_norm_ff, &grad->layer_norm_ff, lr);
    sgd_update_layernorm(&param->layer_norm_final, &grad->layer_norm_final, lr);
}

void sgd_step(Transformer *model, const TransformerGrads *grads, float lr, TransformerConfig config) {
    sgd_update_matrix(&model->token_embeddings, &grads->token_embeddings, lr);
    sgd_update_matrix(&model->lm_head, &grads->lm_head, lr);
    sgd_update_matrix(&model->final_ln_gamma, &grads->final_ln_gamma, lr);
    sgd_update_matrix(&model->final_ln_beta, &grads->final_ln_beta, lr);
    for (int l = 0; l < config.num_layers; l++) {
        sgd_update_block(&model->blocks[l], &grads->blocks[l], lr);
    }
}

/*
 * ============================================================================
 *  Cross-Entropy Loss
 * ============================================================================
 */

float cross_entropy_loss(const float *logits, const int *targets, int vocab_size, int seq_len) {
    float total_loss = 0.0f;
    for (int pos = 0; pos < seq_len; pos++) {
        int target = targets[pos];
        /* Compute softmax for this position */
        float max_val = logits[pos * vocab_size];
        for (int v = 1; v < vocab_size; v++) {
            if (logits[pos * vocab_size + v] > max_val) max_val = logits[pos * vocab_size + v];
        }
        float sum = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            sum += expf(logits[pos * vocab_size + v] - max_val);
        }
        float log_prob = (logits[pos * vocab_size + target] - max_val) - logf(sum);
        total_loss -= log_prob;
    }
    return total_loss / seq_len;
}

/*
 * ============================================================================
 *  Backward Pass - Layer Normalization
 * ============================================================================
 */

void layernorm_backward(
    const LayerNorm *layer_norm,
    const float *input,
    const float *d_output,
    float *d_input,
    LayerNormGrads *grads,
    int size
) {
    /* Recompute mean and variance from forward pass */
    float mean = 0.0f;
    for (int i = 0; i < size; i++) mean += input[i];
    mean /= size;

    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= size;
    float std = sqrtf(variance + 1e-5f);
    float inv_std = 1.0f / std;

    /* Compute normalized input */
    float *x_hat = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        x_hat[i] = (input[i] - mean) * inv_std;
    }

    /* Gradients for gamma and beta */
    for (int i = 0; i < size; i++) {
        grads->gamma[i] += d_output[i] * x_hat[i];
        grads->beta[i] += d_output[i];
    }

    /* Gradient through normalization */
    float *d_x_hat = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        d_x_hat[i] = d_output[i] * layer_norm->gamma[i];
    }

    float sum_d_x_hat = 0.0f;
    float sum_d_x_hat_x_hat = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_d_x_hat += d_x_hat[i];
        sum_d_x_hat_x_hat += d_x_hat[i] * x_hat[i];
    }

    for (int i = 0; i < size; i++) {
        d_input[i] = inv_std * (d_x_hat[i] - sum_d_x_hat / size - x_hat[i] * sum_d_x_hat_x_hat / size);
    }

    free(x_hat);
    free(d_x_hat);
}

/*
 * ============================================================================
 *  Backward Pass - Multi-Head Attention
 * ============================================================================
 */

void multihead_attention_backward(
    const MultiHeadAttention *attention,
    const float *input,
    const float *d_output,
    float *d_input,
    MultiHeadAttentionGrads *grads,
    int sequence_length,
    int d_model,
    int use_causal_mask
) {
    int head_dim = attention->head_dim;
    int num_heads = attention->num_heads;
    float scaling_factor = 1.0f / sqrtf((float)head_dim);
    int head_vector_size = sequence_length * head_dim;

    /* Recompute Q, K, V */
    float *query = (float *)calloc(num_heads * head_vector_size, sizeof(float));
    float *key = (float *)calloc(num_heads * head_vector_size, sizeof(float));
    float *value = (float *)calloc(num_heads * head_vector_size, sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        for (int row = 0; row < sequence_length; row++) {
            for (int col = 0; col < d_model; col++) {
                float inp = input[row * d_model + col];
                for (int dim = 0; dim < head_dim; dim++) {
                    query[h * head_vector_size + row * head_dim + dim] += inp * attention->heads[h].weight_query.data[col * head_dim + dim];
                    key[h * head_vector_size + row * head_dim + dim] += inp * attention->heads[h].weight_key.data[col * head_dim + dim];
                    value[h * head_vector_size + row * head_dim + dim] += inp * attention->heads[h].weight_value.data[col * head_dim + dim];
                }
            }
        }
    }

    /* Backward through output projection: d_concat = d_output @ W_O^T */
    float *d_concat = (float *)calloc(sequence_length * d_model, sizeof(float));
    for (int row = 0; row < sequence_length; row++) {
        for (int dim = 0; dim < d_model; dim++) {
            for (int col = 0; col < d_model; col++) {
                d_concat[row * d_model + dim] += d_output[row * d_model + col] * attention->weight_output.data[dim * d_model + col];
            }
        }
    }

    /* Gradient for W_O: d_W_O = concat^T @ d_output */
    float *concat = (float *)calloc(sequence_length * d_model, sizeof(float));
    for (int h = 0; h < num_heads; h++) {
        for (int row = 0; row < sequence_length; row++) {
            for (int dim = 0; dim < head_dim; dim++) {
                concat[row * d_model + h * head_dim + dim] = 0.0f; /* will fill below */
            }
        }
    }

    /* Recompute head outputs to get concat */
    for (int h = 0; h < num_heads; h++) {
        int score_matrix_size = sequence_length * sequence_length;
        float *attention_scores = (float *)calloc(score_matrix_size, sizeof(float));

        for (int row = 0; row < sequence_length; row++) {
            for (int col = 0; col < sequence_length; col++) {
                float dot = 0.0f;
                for (int dim = 0; dim < head_dim; dim++) {
                    dot += query[h * head_vector_size + row * head_dim + dim] * key[h * head_vector_size + col * head_dim + dim];
                }
                attention_scores[row * sequence_length + col] = dot * scaling_factor;
                if (use_causal_mask && col > row) {
                    attention_scores[row * sequence_length + col] = -1e9f;
                }
            }
        }

        /* Softmax */
        for (int row = 0; row < sequence_length; row++) {
            float max_s = attention_scores[row * sequence_length];
            for (int col = 1; col < sequence_length; col++) {
                if (attention_scores[row * sequence_length + col] > max_s) max_s = attention_scores[row * sequence_length + col];
            }
            float sum = 0.0f;
            for (int col = 0; col < sequence_length; col++) {
                attention_scores[row * sequence_length + col] = expf(attention_scores[row * sequence_length + col] - max_s);
                sum += attention_scores[row * sequence_length + col];
            }
            for (int col = 0; col < sequence_length; col++) {
                attention_scores[row * sequence_length + col] /= sum;
            }
        }

        /* Head output */
        float *head_out = (float *)calloc(sequence_length * head_dim, sizeof(float));
        for (int row = 0; row < sequence_length; row++) {
            for (int dim = 0; dim < head_dim; dim++) {
                float ws = 0.0f;
                for (int col = 0; col < sequence_length; col++) {
                    ws += attention_scores[row * sequence_length + col] * value[h * head_vector_size + col * head_dim + dim];
                }
                head_out[row * head_dim + dim] = ws;
            }
        }

        /* Fill concat */
        for (int row = 0; row < sequence_length; row++) {
            for (int dim = 0; dim < head_dim; dim++) {
                concat[row * d_model + h * head_dim + dim] = head_out[row * head_dim + dim];
            }
        }

        /* Backward through head output: d_head_out = slice of d_concat */
        float *d_head_out = (float *)calloc(sequence_length * head_dim, sizeof(float));
        for (int row = 0; row < sequence_length; row++) {
            for (int dim = 0; dim < head_dim; dim++) {
                d_head_out[row * head_dim + dim] = d_concat[row * d_model + h * head_dim + dim];
            }
        }

        /* Backward through scores @ V: d_scores = d_head_out @ V^T */
        float *d_scores = (float *)calloc(score_matrix_size, sizeof(float));
        for (int row = 0; row < sequence_length; row++) {
            for (int col = 0; col < sequence_length; col++) {
                float ds = 0.0f;
                for (int dim = 0; dim < head_dim; dim++) {
                    ds += d_head_out[row * head_dim + dim] * value[h * head_vector_size + col * head_dim + dim];
                }
                d_scores[row * sequence_length + col] = ds;
            }
        }

        /* Gradient for V: d_V = scores^T @ d_head_out */
        for (int col = 0; col < sequence_length; col++) {
            for (int dim = 0; dim < head_dim; dim++) {
                float dv = 0.0f;
                for (int row = 0; row < sequence_length; row++) {
                    dv += attention_scores[row * sequence_length + col] * d_head_out[row * head_dim + dim];
                }
                for (int inp_dim = 0; inp_dim < d_model; inp_dim++) {
                    grads->heads[h].weight_value.data[inp_dim * head_dim + dim] += input[col * d_model + inp_dim] * dv;
                }
            }
        }

        /* Backward through softmax: d_scores *= softmax * (1 - softmax) simplified */
        float *d_softmax = (float *)calloc(score_matrix_size, sizeof(float));
        for (int row = 0; row < sequence_length; row++) {
            float sum_ds_s = 0.0f;
            for (int col = 0; col < sequence_length; col++) {
                sum_ds_s += d_scores[row * sequence_length + col] * attention_scores[row * sequence_length + col];
            }
            for (int col = 0; col < sequence_length; col++) {
                d_softmax[row * sequence_length + col] = attention_scores[row * sequence_length + col] * (d_scores[row * sequence_length + col] - sum_ds_s);
            }
        }

        /* Apply causal mask gradient (zero out masked positions) */
        for (int row = 0; row < sequence_length; row++) {
            for (int col = 0; col < sequence_length; col++) {
                if (use_causal_mask && col > row) {
                    d_softmax[row * sequence_length + col] = 0.0f;
                }
            }
        }

        /* Backward through scaling */
        for (int i = 0; i < score_matrix_size; i++) {
            d_softmax[i] *= scaling_factor;
        }

        /* Backward through Q @ K^T: d_Q = d_softmax @ K, d_K = d_softmax^T @ Q */
        float *d_Q = (float *)calloc(head_vector_size, sizeof(float));
        float *d_K = (float *)calloc(head_vector_size, sizeof(float));

        for (int row = 0; row < sequence_length; row++) {
            for (int dim = 0; dim < head_dim; dim++) {
                float dq = 0.0f, dk = 0.0f;
                for (int col = 0; col < sequence_length; col++) {
                    dq += d_softmax[row * sequence_length + col] * key[h * head_vector_size + col * head_dim + dim];
                    dk += d_softmax[col * sequence_length + row] * query[h * head_vector_size + col * head_dim + dim];
                }
                d_Q[row * head_dim + dim] = dq;
                d_K[row * head_dim + dim] = dk;
            }
        }

        /* Gradients for W_Q, W_K */
        for (int row = 0; row < sequence_length; row++) {
            for (int inp_dim = 0; inp_dim < d_model; inp_dim++) {
                float inp = input[row * d_model + inp_dim];
                for (int dim = 0; dim < head_dim; dim++) {
                    grads->heads[h].weight_query.data[inp_dim * head_dim + dim] += inp * d_Q[row * head_dim + dim];
                    grads->heads[h].weight_key.data[inp_dim * head_dim + dim] += inp * d_K[row * head_dim + dim];
                }
            }
        }

        free(attention_scores);
        free(head_out);
        free(d_head_out);
        free(d_scores);
        free(d_softmax);
        free(d_Q);
        free(d_K);
    }

    /* Gradient for W_O: d_W_O = concat^T @ d_output (computed outside head loop) */
    for (int dim = 0; dim < d_model; dim++) {
        for (int col = 0; col < d_model; col++) {
            float grad = 0.0f;
            for (int row = 0; row < sequence_length; row++) {
                grad += concat[row * d_model + dim] * d_output[row * d_model + col];
            }
            grads->weight_output.data[dim * d_model + col] += grad;
        }
    }

    free(query);
    free(key);
    free(value);
    free(d_concat);
    free(concat);
}

/*
 * ============================================================================
 *  Backward Pass - Feed-Forward Network
 * ============================================================================
 */

void feedforward_backward(
    const FeedForward *ff,
    const float *input,
    const float *d_output,
    float *d_input,
    FeedForwardGrads *grads,
    int sequence_length
) {
    int d_model = ff->d_model;
    int d_ff = ff->d_ff;

    /* Recompute forward pass hidden state */
    float *hidden = (float *)calloc(sequence_length * d_ff, sizeof(float));
    for (int row = 0; row < sequence_length; row++) {
        for (int col = 0; col < d_ff; col++) {
            float acc = ff->bias_up.data[col];
            for (int dim = 0; dim < d_model; dim++) {
                acc += input[row * d_model + dim] * ff->weight_up.data[dim * d_ff + col];
            }
            hidden[row * d_ff + col] = acc;
        }
    }

    /* ReLU mask */
    int *relu_mask = (int *)calloc(sequence_length * d_ff, sizeof(int));
    for (int i = 0; i < sequence_length * d_ff; i++) {
        relu_mask[i] = hidden[i] > 0.0f ? 1 : 0;
        hidden[i] = relu_mask[i] ? hidden[i] : 0.0f;
    }

    /* Gradient through second linear layer */
    float *d_hidden = (float *)calloc(sequence_length * d_ff, sizeof(float));
    for (int row = 0; row < sequence_length; row++) {
        for (int dim = 0; dim < d_ff; dim++) {
            for (int col = 0; col < d_model; col++) {
                d_hidden[row * d_ff + dim] += d_output[row * d_model + col] * ff->weight_down.data[dim * d_model + col];
                grads->weight_down.data[dim * d_model + col] += hidden[row * d_ff + dim] * d_output[row * d_model + col];
            }
        }
        for (int col = 0; col < d_model; col++) {
            grads->bias_down.data[col] += d_output[row * d_model + col];
        }
    }

    /* Through ReLU */
    for (int i = 0; i < sequence_length * d_ff; i++) {
        d_hidden[i] *= relu_mask[i];
    }

    /* Gradient through first linear layer */
    for (int row = 0; row < sequence_length; row++) {
        for (int dim = 0; dim < d_model; dim++) {
            for (int col = 0; col < d_ff; col++) {
                d_input[row * d_model + dim] += d_hidden[row * d_ff + col] * ff->weight_up.data[dim * d_ff + col];
                grads->weight_up.data[dim * d_ff + col] += input[row * d_model + dim] * d_hidden[row * d_ff + col];
            }
        }
        for (int col = 0; col < d_ff; col++) {
            grads->bias_up.data[col] += d_hidden[row * d_ff + col];
        }
    }

    free(hidden);
    free(relu_mask);
    free(d_hidden);
}

/*
 * ============================================================================
 *  Backward Pass - Transformer Block
 * ============================================================================
 */

void transformer_block_backward(
    const TransformerBlock *block,
    const float *input,
    const float *d_output,
    float *d_input,
    TransformerBlockGrads *grads,
    int sequence_length,
    int use_causal_mask
) {
    int d_model = block->d_model;
    int vector_size = sequence_length * d_model;

    /* Recompute forward pass intermediates */
    float *normed1 = (float *)malloc(vector_size * sizeof(float));
    float *attn_out = (float *)malloc(vector_size * sizeof(float));
    float *after_attn = (float *)malloc(vector_size * sizeof(float));
    float *normed2 = (float *)malloc(vector_size * sizeof(float));
    float *ff_out = (float *)malloc(vector_size * sizeof(float));
    float *normed3 = (float *)malloc(vector_size * sizeof(float));

    for (int pos = 0; pos < sequence_length; pos++) {
        layernorm_forward(&block->layer_norm_attn, &input[pos * d_model], &normed1[pos * d_model], d_model);
    }
    multihead_attention_forward(&block->self_attention, normed1, attn_out, sequence_length, d_model, use_causal_mask);
    for (int i = 0; i < vector_size; i++) after_attn[i] = input[i] + attn_out[i];
    for (int pos = 0; pos < sequence_length; pos++) {
        layernorm_forward(&block->layer_norm_ff, &after_attn[pos * d_model], &normed2[pos * d_model], d_model);
    }
    feedforward_forward(&block->feed_forward, normed2, ff_out, sequence_length);
    for (int i = 0; i < vector_size; i++) after_attn[i] = after_attn[i] + ff_out[i];
    for (int pos = 0; pos < sequence_length; pos++) {
        layernorm_forward(&block->layer_norm_final, &after_attn[pos * d_model], &normed3[pos * d_model], d_model);
    }

    /* Backward through final layernorm */
    float *d_after_ff = (float *)calloc(vector_size, sizeof(float));
    for (int pos = 0; pos < sequence_length; pos++) {
        layernorm_backward(&block->layer_norm_final, &after_attn[pos * d_model], &d_output[pos * d_model], &d_after_ff[pos * d_model], &grads->layer_norm_final, d_model);
    }

    /* d_input accumulates from residual connections */
    float *d_attn_out = (float *)calloc(vector_size, sizeof(float));
    float *d_ff_out = (float *)calloc(vector_size, sizeof(float));

    /* Backward through FFN residual: d_after_attn = d_after_ff, d_ff_out = d_after_ff */
    float *d_after_attn = (float *)calloc(vector_size, sizeof(float));
    for (int i = 0; i < vector_size; i++) {
        d_after_attn[i] = d_after_ff[i];
        d_ff_out[i] = d_after_ff[i];
    }

    /* Backward through FFN */
    float *d_normed2 = (float *)calloc(vector_size, sizeof(float));
    feedforward_backward(&block->feed_forward, normed2, d_ff_out, d_normed2, &grads->feed_forward, sequence_length);

    /* Backward through layernorm_ff */
    float *d_after_attn_ln = (float *)calloc(vector_size, sizeof(float));
    for (int pos = 0; pos < sequence_length; pos++) {
        layernorm_backward(&block->layer_norm_ff, &after_attn[pos * d_model], &d_normed2[pos * d_model], &d_after_attn_ln[pos * d_model], &grads->layer_norm_ff, d_model);
    }
    for (int i = 0; i < vector_size; i++) d_after_attn[i] += d_after_attn_ln[i];

    /* Backward through attention residual: d_input += d_after_attn, d_attn_out = d_after_attn */
    for (int i = 0; i < vector_size; i++) {
        d_input[i] = d_after_attn[i];
        d_attn_out[i] = d_after_attn[i];
    }

    /* Backward through attention */
    float *d_normed1 = (float *)calloc(vector_size, sizeof(float));
    multihead_attention_backward(&block->self_attention, normed1, d_attn_out, d_normed1, &grads->self_attention, sequence_length, d_model, use_causal_mask);

    /* Backward through layernorm_attn */
    for (int pos = 0; pos < sequence_length; pos++) {
        layernorm_backward(&block->layer_norm_attn, &input[pos * d_model], &d_normed1[pos * d_model], &d_input[pos * d_model], &grads->layer_norm_attn, d_model);
    }

    free(normed1); free(attn_out); free(after_attn);
    free(normed2); free(ff_out); free(normed3);
    free(d_after_ff); free(d_attn_out); free(d_ff_out);
    free(d_after_attn); free(d_normed2); free(d_after_attn_ln); free(d_normed1);
}

/*
 * ============================================================================
 *  Full Model Backward Pass
 * ============================================================================
 */

void transformer_backward(
    const Transformer *model,
    const int *input_tokens,
    const float *logits,
    const int *target_tokens,
    TransformerGrads *grads,
    int sequence_length
) {
    int d_model = model->config.d_model;
    int vocab_size = model->config.vocab_size;
    int vector_size = sequence_length * d_model;

    /* Recompute forward pass */
    float *hidden_states = (float *)malloc(vector_size * sizeof(float));
    float **layer_inputs = (float **)malloc((model->config.num_layers + 1) * sizeof(float *));
    layer_inputs[0] = (float *)malloc(vector_size * sizeof(float));

    for (int pos = 0; pos < sequence_length; pos++) {
        int token = input_tokens[pos];
        for (int dim = 0; dim < d_model; dim++) {
            hidden_states[pos * d_model + dim] = model->token_embeddings.data[token * d_model + dim]
                                               + model->positional_embeddings.data[pos * d_model + dim];
            layer_inputs[0][pos * d_model + dim] = hidden_states[pos * d_model + dim];
        }
    }

    for (int l = 0; l < model->config.num_layers; l++) {
        layer_inputs[l + 1] = (float *)malloc(vector_size * sizeof(float));
        transformer_block_forward(&model->blocks[l], hidden_states, layer_inputs[l + 1], sequence_length, 1);
        memcpy(hidden_states, layer_inputs[l + 1], vector_size * sizeof(float));
    }

    /* Final layernorm */
    float *normalized = (float *)malloc(vector_size * sizeof(float));
    for (int pos = 0; pos < sequence_length; pos++) {
        float mean = 0.0f, var = 0.0f;
        for (int dim = 0; dim < d_model; dim++) mean += hidden_states[pos * d_model + dim];
        mean /= d_model;
        for (int dim = 0; dim < d_model; dim++) {
            float diff = hidden_states[pos * d_model + dim] - mean;
            var += diff * diff;
        }
        var /= d_model;
        float std = sqrtf(var + 1e-5f);
        for (int dim = 0; dim < d_model; dim++) {
            float x_hat = (hidden_states[pos * d_model + dim] - mean) / std;
            normalized[pos * d_model + dim] = model->final_ln_gamma.data[dim] * x_hat + model->final_ln_beta.data[dim];
        }
    }

    /* Compute d_logits from cross-entropy */
    float *d_logits = (float *)calloc(sequence_length * vocab_size, sizeof(float));
    for (int pos = 0; pos < sequence_length; pos++) {
        /* Softmax */
        float max_val = logits[pos * vocab_size];
        for (int v = 1; v < vocab_size; v++) {
            if (logits[pos * vocab_size + v] > max_val) max_val = logits[pos * vocab_size + v];
        }
        float sum = 0.0f;
        float *probs = (float *)malloc(vocab_size * sizeof(float));
        for (int v = 0; v < vocab_size; v++) {
            probs[v] = expf(logits[pos * vocab_size + v] - max_val);
            sum += probs[v];
        }
        for (int v = 0; v < vocab_size; v++) probs[v] /= sum;

        /* d_logits = probs - one_hot(target) */
        for (int v = 0; v < vocab_size; v++) {
            d_logits[pos * vocab_size + v] = probs[v];
        }
        d_logits[pos * vocab_size + target_tokens[pos]] -= 1.0f;
        free(probs);
    }

    /* Backward through LM head */
    float *d_normalized = (float *)calloc(vector_size, sizeof(float));
    for (int pos = 0; pos < sequence_length; pos++) {
        for (int dim = 0; dim < d_model; dim++) {
            for (int v = 0; v < vocab_size; v++) {
                d_normalized[pos * d_model + dim] += d_logits[pos * vocab_size + v] * model->lm_head.data[dim * vocab_size + v];
                grads->lm_head.data[dim * vocab_size + v] += normalized[pos * d_model + dim] * d_logits[pos * vocab_size + v];
            }
        }
    }

    /* Backward through final layernorm */
    float *d_hidden = (float *)calloc(vector_size, sizeof(float));
    for (int pos = 0; pos < sequence_length; pos++) {
        layernorm_backward(
            &(LayerNorm){.gamma = model->final_ln_gamma.data, .beta = model->final_ln_beta.data, .size = d_model},
            &hidden_states[pos * d_model],
            &d_normalized[pos * d_model],
            &d_hidden[pos * d_model],
            &(LayerNormGrads){.gamma = grads->final_ln_gamma.data, .beta = grads->final_ln_beta.data, .size = d_model},
            d_model
        );
    }

    /* Backward through transformer blocks (reverse order) */
    for (int l = model->config.num_layers - 1; l >= 0; l--) {
        float *d_layer_input = (float *)calloc(vector_size, sizeof(float));
        transformer_block_backward(&model->blocks[l], layer_inputs[l], d_hidden, d_layer_input, &grads->blocks[l], sequence_length, 1);
        free(d_hidden);
        d_hidden = d_layer_input;
    }

    /* Gradients for token embeddings and positional embeddings */
    for (int pos = 0; pos < sequence_length; pos++) {
        int token = input_tokens[pos];
        for (int dim = 0; dim < d_model; dim++) {
            grads->token_embeddings.data[token * d_model + dim] += d_hidden[pos * d_model + dim];
            grads->positional_embeddings.data[pos * d_model + dim] += d_hidden[pos * d_model + dim];
        }
    }

    free(hidden_states);
    for (int l = 0; l <= model->config.num_layers; l++) free(layer_inputs[l]);
    free(layer_inputs);
    free(normalized);
    free(d_logits);
    free(d_normalized);
    free(d_hidden);
}

/*
 * ============================================================================
 *  Generate Text
 * ============================================================================
 */

void generate_text(const Transformer *model, const int *prompt, int prompt_len, int gen_len) {
    int *tokens = (int *)malloc((prompt_len + gen_len) * sizeof(int));
    float *logits = (float *)malloc((prompt_len + gen_len) * CHAR_VOCAB_SIZE * sizeof(float));
    memcpy(tokens, prompt, prompt_len * sizeof(int));
    int current_len = prompt_len;

    printf("Prompt: ");
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

/*
 * ============================================================================
 *  Main - Training Loop
 * ============================================================================
 */

int main(void) {
    srand(42);

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

    /* Initialize gradients */
    TransformerGrads grads;
    grad_init_model(&grads, config);

    /* Prepare training data */
    int text_len = strlen(training_text);
    int *data = (int *)malloc(text_len * sizeof(int));
    for (int i = 0; i < text_len; i++) {
        data[i] = char_to_id(training_text[i]);
    }

    int seq_len = 8;
    printf("=== Character-Level Transformer Training ===\n");
    printf("Vocab: %d chars, d_model: %d, heads: %d, layers: %d\n",
           CHAR_VOCAB_SIZE, CHAR_D_MODEL, CHAR_NUM_HEADS, CHAR_NUM_LAYERS);
    printf("Training text: %d characters\n\n", text_len);

    /* Training loop */
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int num_batches = 0;

        grad_zero_model(&grads, config);

        /* Iterate over sequence chunks */
        for (int start = 0; start < text_len - seq_len; start += seq_len) {
            int *input = &data[start];
            int *target = &data[start + 1];

            /* Forward */
            float *logits = (float *)malloc(seq_len * CHAR_VOCAB_SIZE * sizeof(float));
            transformer_forward(&model, input, logits, seq_len);

            /* Loss */
            float loss = cross_entropy_loss(logits, target, CHAR_VOCAB_SIZE, seq_len);
            total_loss += loss;
            num_batches++;

            /* Backward */
            transformer_backward(&model, input, logits, target, &grads, seq_len);
            free(logits);
        }

        /* Average gradients */
        int total_params = 0;
        /* (simplified: just scale by 1/num_batches) */
        float inv_batches = 1.0f / num_batches;

        /* SGD step */
        sgd_step(&model, &grads, LEARNING_RATE * inv_batches, config);

        float avg_loss = total_loss / num_batches;

        if (epoch % EVAL_EVERY == 0 || epoch == NUM_EPOCHS - 1) {
            printf("Epoch %4d | Loss: %.4f | Perplexity: %.2f\n",
                   epoch, avg_loss, expf(avg_loss));

            /* Generate sample */
            int prompt[] = {char_to_id('t'), char_to_id('h'), char_to_id('e'), char_to_id(' ')};
            generate_text(&model, prompt, 4, 20);
        }
    }

    /* Cleanup */
    free(data);
    grad_free_model(&grads, config);
    transformer_free(&model);

    printf("\nTraining complete!\n");
    return 0;
}
