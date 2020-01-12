/*
 * Minimal character-level Vanilla RNN model
 * 
 * Based on Python code written by Andrej Karpathy (@karpathy)
 * https://gist.github.com/karpathy/d4dee566867f8291f086
 * BSD License
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <stdbool.h>
#include <time.h>
#include <sys/select.h>
#include <sys/signal.h>
#include <sys/stat.h>
#include <sys/time.h>

#include "rnn.h"
#include "randdouble.h"
#include "unicode.h"
#include "map.h"

#define MAX_SYMBOLS 65536
#define MAX_CHARS 256

static const uint64_t dump_sanity = 0x65746174534e4e52; // RNNState
static const uint64_t sample_sanity = 0x6c706d61534e4e52; // RNNSampl
static const uint64_t iter_sanity = 0x49726574494e4e52; // RNNIterI

__attribute((noreturn)) void _fail(const char *reason, const char *file, uint32_t line) {
    fprintf(stderr, "%s (%s:%u)\n", reason, file, line);
    exit(1);
}

static void *check_alloc(size_t n) {
    void *ptr = malloc(n);
    if (ptr == NULL) exit(1);
    return ptr;
}

static void ignore_signal(int dummy) {}

#define alloc_typed(type, n) ((type *) check_alloc((n) * sizeof(type)))
#define alloc_double(n) alloc_typed(double, n)
#define alloc_double_ptr(n) alloc_typed(double*, n)
#define alloc_size_t(n) alloc_typed(size_t, n)
#define alloc_symbol(n) alloc_typed(symbol_t, n)
#define alloc_wwchar(n) alloc_typed(wwchar_t, n)

#define check_read(ptr, size, n) \
    do { \
        if (fread(ptr, size, n, stdin) != n) fail("failed to read data"); \
    } while(0)

#define read_typed(type, ptr, n) check_read((void *) (ptr), sizeof(type), n)
#define read_double(ptr, n) read_typed(double, ptr, n)
#define read_size_t(ptr, n) read_typed(size_t, ptr, n)
#define read_symbol(ptr, n) read_typed(symbol_t, ptr, n)
#define read_wwchar(ptr, n) read_typed(wwchar_t, ptr, n)

#define check_write(ptr, size, n) \
    do { \
        if (fwrite(ptr, size, n, stdout) != n) fail("failed to write data"); \
    } while(0)

#define write_typed(type, ptr, n) check_write((void *) (ptr), sizeof(type), n)
#define write_double(ptr, n) write_typed(double, ptr, n)
#define write_size_t(ptr, n) write_typed(size_t, ptr, n)
#define write_symbol(ptr, n) write_typed(symbol_t, ptr, n)
#define write_wwchar(ptr, n) write_typed(wwchar_t, ptr, n)

static double **alloc_2d_double(size_t outer, size_t inner) {
    double **ptr = alloc_double_ptr(outer);
    for (size_t i = 0; i < outer; i++) ptr[i] = alloc_double(inner);
    return ptr;
}

static void free_2d(void **ptr, size_t outer) {
    for (size_t i = 0; i < outer; i++) free(ptr[i]);
    free(ptr);
}

static inline void zero_double(double *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = 0.0;
}

static inline void nrand_double(double *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = randnormal() * 0.01;
}

static inline void copy_double(double *dst, const double *src, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = src[i];
}

/// multiplies mat(rows, cols) by vec(cols, 1) into out(rows, 1)
static inline void mul_mat_col(size_t rows, size_t cols, const double *mat, const double *vec, double *out) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            out[i] += mat[i + rows * j] * vec[j];
        }
    }
}

/// multiplies col(rows, 1) by col(1, cols) into out(rows, cols)
static inline void mul_col_row(size_t rows, size_t cols, const double *col, const double *row, double *out) {
    for (size_t j = 0; j < cols; j++) {
        for (size_t i = 0; i < rows; i++) {
            out[i + rows * j] += row[j] * col[i];
        }
    }
}

/// transposes mat(rows, cols) into out(cols, rows)
static inline void transpose_mat(size_t rows, size_t cols, const double *mat, double *out) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            out[j + cols * i] = mat[i + rows * j];
        }
    }
}

static inline void add_double(size_t len, const double *from, double *to) {
    for (size_t i = 0; i < len; i++) to[i] += from[i];
}

static inline void learn_double(size_t len, const double *dparam, double *mem, double *param, double rate) {
    for (size_t i = 0; i < len; i++) {
        mem[i] += dparam[i] * dparam[i];
        param[i] += -rate * dparam[i] / sqrt(mem[i] + 1e-8);
    }
}

static inline void vec_to_exp_probs(size_t n, const double *vec, double *out) {
    double tot = 0;
    for (size_t i = 0; i < n; i++) tot += out[i] = exp(vec[i]);
    for (size_t i = 0; i < n; i++) out[i] /= tot;
}

static inline void tanh_vec(size_t n, double *vec) {
    for (size_t i = 0; i < n; i++) vec[i] = tanh(vec[i]);
}

static inline void rev_tanh_vec(size_t n, double *vec, const double *hs) {
    for (size_t i = 0; i < n; i++) vec[i] *= 1 - hs[i] * hs[i];
}

static inline void clip_double(double *ptr, size_t n, double limit) {
    for (size_t i = 0; i < n; i++) ptr[i] = ptr[i] > limit ? limit : ptr[i] < -limit ? -limit : ptr[i];
}

static const size_t seq_length = 25;
static const double learning_rate = 1e-1;

static const size_t sample_length = 200;

static size_t hidden_size;

/* weights input->hidden1 */
static double *Wxh1;
/* weights hidden1->hidden1 */
static double *Wh1h1;
/* weights hidden1->hidden2 */
static double *Wh1h2;
/* weights hidden2->hidden2 */
static double *Wh2h2;
/* weights hidden2->output */
static double *Wh2y;
/* bias hidden1 */
static double *bh1;
/* bias hidden2 */
static double *bh2;
/* bias output */
static double *by;

/* cached transpose of Wh2y */
static double *Wh2y_T;
/* cached transpose of Wh2h2 */
static double *Wh2h2_T;
/* cached transpose of Wh1h2 */
static double *Wh1h2_T;
/* cached transpose of Wh1h1 */
static double *Wh1h1_T;

/* Wxh1 learning memory */
static double *mWxh1;
/* Wh1h1 learning memory */
static double *mWh1h1;
/* Wh1h2 learning memory */
static double *mWh1h2;
/* Wh2h2 learning memory */
static double *mWh2h2;
/* Wh2y learning memory */
static double *mWh2y;
/* bh1 learning memory */
static double *mbh1;
/* bh2 learning memory */
static double *mbh2;
/* by learning memory */
static double *mby;

/* Wxh1 delta: how far off from "desired" it was */
static double *dWxh1;
/* Wh1h1 delta: how far off from "desired" it was */
static double *dWh1h1;
/* Wh1h2 delta: how far off from "desired" it was */
static double *dWh1h2;
/* Wh2h2 delta: how far off from "desired" it was */
static double *dWh2h2;
/* Wh2y delta: how far off from "desired" it was */
static double *dWh2y;
/* bh1 delta: how far off from "desired" it was */
static double *dbh1;
/* bh2 delta: how far off from "desired" it was */
static double *dbh2;
/* by delta: how far off from "desired" it was */
static double *dby;

/* values of hidden1 layer in previous iteration */
static double *h1prev;
/* values of hidden2 layer in previous iteration */
static double *h2prev;
/* values of input layer */
static double **xs;
/* values of hidden layer for previous char (overlaps hs) */
static double **ph1s;
/* values of hidden layer for this char */
static double **h1s;
/* values of output layer */
static double **ph2s;
/* values of hidden layer for this char */
static double **h2s;
/* values of output layer */
static double **ys;
/* probabilities of output chars */
static double **ps;

/* delta of hidden1 layer backpropagated from next char */
static double *dh1next;
/* delta of hidden2 layer backpropagated from next char */
static double *dh2next;
/* output layer delta: how far off from "desired" the value was */
static double *dy;
/* hidden1 layer delta: how far off from "desired" the value was */
static double *dh1;
/* hidden2 layer delta: how far off from "desired" the value was */
static double *dh2;

static double smooth_loss;

static map_tree_node vocabulary;

static size_t vocab_size;

static symbol_t *input_data;
static size_t input_len;

static size_t n, p;

static void iteration_v1();
static void iteration_v2();
static void sample_v1(size_t, wwchar_t *, symbol_t);
static void sample_v2(size_t, wwchar_t *, symbol_t);

static void (*iteration_func)();
static void (*sample_func)(size_t, wwchar_t *, symbol_t);

static void alloc_model() {
    Wxh1 = alloc_double(hidden_size * vocab_size);
    Wh1h1 = alloc_double(hidden_size * hidden_size);
    Wh1h2 = alloc_double(hidden_size * hidden_size);
    Wh2h2 = alloc_double(hidden_size * hidden_size);
    Wh2y = alloc_double(vocab_size * hidden_size);
    bh1 = alloc_double(hidden_size);
    bh2 = alloc_double(hidden_size);
    by = alloc_double(vocab_size);

    Wh2y_T = alloc_double(vocab_size * hidden_size);
    Wh2h2_T = alloc_double(hidden_size * hidden_size);
    Wh1h2_T = alloc_double(hidden_size * hidden_size);
    Wh1h1_T = alloc_double(hidden_size * hidden_size);

    mWxh1 = alloc_double(hidden_size * vocab_size);
    mWh1h1 = alloc_double(hidden_size * hidden_size);
    mWh1h2 = alloc_double(hidden_size * hidden_size);
    mWh2h2 = alloc_double(hidden_size * hidden_size);
    mWh2y = alloc_double(vocab_size * hidden_size);
    mbh1 = alloc_double(hidden_size);
    mbh2 = alloc_double(hidden_size);
    mby = alloc_double(vocab_size);

    h1prev = alloc_double(hidden_size);
    h2prev = alloc_double(hidden_size);

    xs = alloc_2d_double(seq_length, vocab_size);
    ph1s = alloc_2d_double(seq_length + 1, hidden_size);
    h1s = &ph1s[1];
    ph2s = alloc_2d_double(seq_length + 1, hidden_size);
    h2s = &ph2s[1];
    ys = alloc_2d_double(seq_length, vocab_size);
    ps = alloc_2d_double(seq_length, vocab_size);

    dWxh1 = alloc_double(hidden_size * vocab_size);
    dWh1h1 = alloc_double(hidden_size * hidden_size);
    dWh1h2 = alloc_double(hidden_size * hidden_size);
    dWh2h2 = alloc_double(hidden_size * hidden_size);
    dWh2y = alloc_double(vocab_size * hidden_size);
    dbh1 = alloc_double(hidden_size);
    dbh2 = alloc_double(hidden_size);
    dby = alloc_double(vocab_size);
    dh1next = alloc_double(hidden_size);
    dh2next = alloc_double(hidden_size);
    dy = alloc_double(vocab_size);
    dh1 = alloc_double(hidden_size);
    dh2 = alloc_double(hidden_size);
}

static void free_model() {
    free(Wxh1);
    free(Wh1h1);
    free(Wh1h2);
    free(Wh2h2);
    free(Wh2y);
    free(bh1);
    free(bh2);
    free(by);

    free(Wh2y_T);
    free(Wh2h2_T);
    free(Wh1h2_T);
    free(Wh1h1_T);

    free(mWxh1);
    free(mWh1h1);
    free(mWh1h2);
    free(mWh2h2);
    free(mWh2y);
    free(mbh1);
    free(mbh2);
    free(mby);

    free(h1prev);
    free(h2prev);

    free_2d((void **) xs, seq_length);
    free_2d((void **) ph1s, seq_length + 1);
    free_2d((void **) ph2s, seq_length + 1);
    free_2d((void **) ys, seq_length);
    free_2d((void **) ps, seq_length);

    free(dWxh1);
    free(dWh1h1);
    free(dWh1h2);
    free(dWh2h2);
    free(dWh2y);
    free(dbh1);
    free(dbh2);
    free(dby);
    free(dh1next);
    free(dh2next);
    free(dy);
    free(dh1);
    free(dh2);

    map_free(&vocabulary, sizeof(symbol_t));
}

static void init_from_text() {
    read_size_t(&hidden_size, 1);

    int read_char;
    size_t raw_input_len = 0;
    size_t buf_len = 16;
    char *input_buf = malloc(buf_len);
    if (input_buf == NULL) fail("failed to alloc input buffer");
    while ((read_char = fgetc(stdin)) != EOF && read_char) {
        if (raw_input_len >= buf_len - 1) {
            buf_len *= 2;
            input_buf = realloc(input_buf, buf_len);
            if (input_buf == NULL) fail("failed to grow input buffer");
        }
        input_buf[raw_input_len++] = read_char;
    }
    input_buf[raw_input_len] = 0;

    fprintf(stderr, "read %lu bytes\n", raw_input_len);

    input_len = utf8strlen(input_buf);
    wwchar_t *input_codepoints = alloc_wwchar(input_len);
    if (!utf8toww(input_codepoints, input_buf)) fail("invalid utf-8 input");

    input_data = alloc_symbol(input_len);
    
    map_new(&vocabulary);

    map_tree_node input_char_map;
    map_new(&input_char_map);
    symbol_t next_symbol = 0;

    for (size_t i = 0; i < input_len; i++) {
        wwchar_t ch = input_codepoints[i];
        if (map_get_or_set_symbol(&input_char_map, ch, &input_data[i], next_symbol)) {
            map_put_codepoint(&vocabulary, next_symbol, ch);
            if (++next_symbol == (symbol_t) -1) fail("out of symbols");
        }
    }

    free(input_buf);
    free(input_codepoints);
    map_free(&input_char_map, sizeof(wwchar_t));

    vocab_size = next_symbol;

    fprintf(stderr, "input size %lu\n", input_len);
    fprintf(stderr, "vocabulary size %lu\n", vocab_size);
    fprintf(stderr, "hidden layer size %lu\n", hidden_size);

    alloc_model();

    nrand_double(Wxh1, hidden_size * vocab_size);
    nrand_double(Wh1h1, hidden_size * hidden_size);
    nrand_double(Wh1h2, hidden_size * hidden_size);
    nrand_double(Wh2h2, hidden_size * hidden_size);
    nrand_double(Wh2y, vocab_size * hidden_size);
    zero_double(bh1, hidden_size);
    zero_double(bh2, hidden_size);
    zero_double(by, vocab_size);

    zero_double(mWxh1, hidden_size * vocab_size);
    zero_double(mWh1h1, hidden_size * hidden_size);
    zero_double(mWh1h2, hidden_size * hidden_size);
    zero_double(mWh2h2, hidden_size * hidden_size);
    zero_double(mWh2y, vocab_size * hidden_size);
    zero_double(mbh1, hidden_size);
    zero_double(mbh2, hidden_size);
    zero_double(mby, vocab_size);

    zero_double(h1prev, hidden_size);
    zero_double(h2prev, hidden_size);
    
    n = 0;
    p = 0;

    smooth_loss = -log(1.0 / vocab_size) * seq_length;

    iteration_func = iteration_v2;
    sample_func = sample_v2;
}

static void init_input_from_stdin() {
    map_new(&vocabulary);

    wwchar_t codepoint;
    for (symbol_t next_symbol = 0; next_symbol < vocab_size; next_symbol++) {
        read_wwchar(&codepoint, 1);
        map_put_codepoint(&vocabulary, next_symbol, codepoint);
    }

    input_data = alloc_symbol(input_len);
    read_symbol(input_data, input_len);
}

static void init_from_stdin_v1() {
    init_input_from_stdin();

    alloc_model();

    read_double(Wxh1, hidden_size * vocab_size);
    read_double(Wh1h1, hidden_size * hidden_size);
    read_double(Wh2y, vocab_size * hidden_size);
    read_double(bh1, hidden_size);
    read_double(by, vocab_size);

    read_double(mWxh1, hidden_size * vocab_size);
    read_double(mWh1h1, hidden_size * hidden_size);
    read_double(mWh2y, vocab_size * hidden_size);
    read_double(mbh1, hidden_size);
    read_double(mby, vocab_size);

    read_double(h1prev, hidden_size);

    read_double(&n, 1);
    read_double(&p, 1);

    read_double(&smooth_loss, 1);

    iteration_func = iteration_v1;
    sample_func = sample_v1;
}

static void init_from_stdin_v2() {
    alloc_model();

    read_double(Wxh1, hidden_size * vocab_size);
    read_double(Wh1h1, hidden_size * hidden_size);
    read_double(Wh1h2, hidden_size * hidden_size);
    read_double(Wh2h2, hidden_size * hidden_size);
    read_double(Wh2y, vocab_size * hidden_size);
    read_double(bh1, hidden_size);
    read_double(bh2, hidden_size);
    read_double(by, vocab_size);

    read_double(mWxh1, hidden_size * vocab_size);
    read_double(mWh1h1, hidden_size * hidden_size);
    read_double(mWh1h2, hidden_size * hidden_size);
    read_double(mWh2h2, hidden_size * hidden_size);
    read_double(mWh2y, vocab_size * hidden_size);
    read_double(mbh1, hidden_size);
    read_double(mbh2, hidden_size);
    read_double(mby, vocab_size);

    read_double(h1prev, hidden_size);
    read_double(h2prev, hidden_size);

    read_double(&n, 1);
    read_double(&p, 1);

    read_double(&smooth_loss, 1);

    iteration_func = iteration_v2;
    sample_func = sample_v2;
}

static void init_from_stdin() {
    size_t version = 0;
    read_size_t(&vocab_size, 1);

    if (vocab_size != 0) {
        // version 0
        read_size_t(&input_len, 1);
        hidden_size = 300;
    } else {
        // versioned file
        read_size_t(&version, 1);
        if (version == 2 || version == 1) {
            read_size_t(&hidden_size, 1);
            read_size_t(&vocab_size, 1);
            read_size_t(&input_len, 1);
        } else fail("unknown data version");
    }

    if (vocab_size > 0x110000) fail("invalid vocabulary size; bad state file?");

    fprintf(stderr, "file version %lu\n", version);
    fprintf(stderr, "input size %lu\n", input_len);
    fprintf(stderr, "vocabulary size %lu\n", vocab_size);
    fprintf(stderr, "hidden layer size %lu\n", hidden_size);

    if (version < 2) {
        init_from_stdin_v1();
    } else {
        init_from_stdin_v2();
    }

    fprintf(stderr, "iteration %lu\n", n);
    fprintf(stderr, "loss %f\n", smooth_loss);

    // sanity check
    uint64_t sanity;
    read_typed(uint64_t, &sanity, 1);
    if (sanity != dump_sanity) fail("dump length mismatch");
}

static void dump_to_stdout() {
    static const size_t new_version_magic = 0;
    static const size_t version = 1;
    write_size_t(&new_version_magic, 1);
    write_size_t(&version, 1);

    write_size_t(&hidden_size, 1);
    write_size_t(&vocab_size, 1);
    write_size_t(&input_len, 1);
    
    wwchar_t codepoint;
    for (symbol_t next_symbol = 0; next_symbol < vocab_size; next_symbol++) {
        map_get_codepoint(&vocabulary, next_symbol, &codepoint);
        write_wwchar(&codepoint, 1);
    }

    write_symbol(input_data, input_len);

    write_double(Wxh1, hidden_size * vocab_size);
    write_double(Wh1h1, hidden_size * hidden_size);
    write_double(Wh1h2, hidden_size * hidden_size);
    write_double(Wh2h2, hidden_size * hidden_size);
    write_double(Wh2y, vocab_size * hidden_size);
    write_double(bh1, hidden_size);
    write_double(bh2, hidden_size);
    write_double(by, vocab_size);

    write_double(mWxh1, hidden_size * vocab_size);
    write_double(mWh1h1, hidden_size * hidden_size);
    write_double(mWh1h2, hidden_size * hidden_size);
    write_double(mWh2h2, hidden_size * hidden_size);
    write_double(mWh2y, vocab_size * hidden_size);
    write_double(mbh1, hidden_size);
    write_double(mbh2, hidden_size);
    write_double(mby, vocab_size);

    write_double(h1prev, hidden_size);
    write_double(h2prev, hidden_size);

    write_double(&n, 1);
    write_double(&p, 1);

    write_double(&smooth_loss, 1);

    write_typed(uint64_t, &dump_sanity, 1);
}

static void iteration_v1() {
    // compute transposes here since matrices are not modified until at end of loop
    transpose_mat(vocab_size, hidden_size, Wh2y, Wh2y_T);
    transpose_mat(hidden_size, hidden_size, Wh1h1, Wh1h1_T);

    // if at end of input, reset to start and reset hidden layer to zeroes
    if (p + seq_length + 1 >= input_len) {
        zero_double(h1prev, hidden_size);
        p = 0;
    }

    // input = input_data[p : p+seq_length]
    const symbol_t *inputs = &input_data[p];
    // training output = input_data[p+1 : p+seq_length+1]
    const symbol_t *targets = &input_data[p + 1];

    // initialize loss at 0
    double loss = 0.0;
    // initialize hidden layer at previous training iteration's value
    copy_double(ph1s[0], h1prev, hidden_size);

    // t is the character position in the training sequence
    for (size_t t = 0; t < seq_length; t++) {
        // set input neurons for the round, xs[t], to 1 for the current input char
        zero_double(xs[t], vocab_size);
        xs[t][inputs[t]] = 1;

        // compute new values of hidden layer
        // hs[t] = Wxh @ xs[t] + Whh @ hs[t-1] + bh
        copy_double(h1s[t], bh1, hidden_size);
        mul_mat_col(hidden_size, vocab_size, Wxh1, xs[t], h1s[t]);
        mul_mat_col(hidden_size, hidden_size, Wh1h1, ph1s[t], h1s[t]);
        // apply activation function to hidden layer
        // hs[t] = tanh(hs[t])
        tanh_vec(hidden_size, h1s[t]);

        // compute new values of output layer
        // ys[t] = Why @ hs[t] + by
        copy_double(ys[t], by, vocab_size);
        mul_mat_col(vocab_size, hidden_size, Wh2y, h1s[t], ys[t]);

        // compute the probabilities of each char from the output layer values
        // ps[t] = exp(ps[t]) / sum(exp(ps[t]))
        vec_to_exp_probs(vocab_size, ys[t], ps[t]);

        // compute the loss
        // loss += -log(ps[t][targets[t]])
        loss += -log(ps[t][targets[t]]);
    }

    // initialize deltas to zero
    zero_double(dWxh1, hidden_size * vocab_size);
    zero_double(dWh1h1, hidden_size * hidden_size);
    zero_double(dWh2y, vocab_size * hidden_size);
    zero_double(dbh1, hidden_size);
    zero_double(dby, vocab_size);
    zero_double(dh1next, hidden_size);

    // compute deltas backward from end of training sequence
    for (size_t t = seq_length - 1; t != (size_t) -1; t--) {
        // output delta = difference from "everything 0 except the correct char"
        copy_double(dy, ps[t], vocab_size);
        dy[targets[t]] -= 1;

        // backpropagate output delta to hidden->output weights and output bias
        // dWhy += dy @ hs[t]
        mul_col_row(vocab_size, hidden_size, dy, h1s[t], dWh2y);
        // dby += dy
        add_double(vocab_size, dy, dby);

        // hidden delta = hidden delta from next char + delta from output
        // dh = Why_T @ dy + dhnext
        copy_double(dh1, dh1next, hidden_size);
        mul_mat_col(hidden_size, vocab_size, Wh2y_T, dy, dh1);

        // reverse activation function
        // dh = (1 - hs[t]^2) * dh
        rev_tanh_vec(hidden_size, dh1, h1s[t]);

        // backpropagate hidden delta to hidden->hidden, input->hidden weights and hidden bias
        // dbh += dh
        add_double(hidden_size, dh1, dbh1);
        // dWxh += dh @ xs[t]
        mul_col_row(hidden_size, vocab_size, dh1, xs[t], dWxh1);
        // dWhh += dh @ hs[t-1]
        mul_col_row(hidden_size, hidden_size, dh1, ph1s[t], dWh1h1);

        // compute hidden delta for prev char from current hidden delta
        // dhnext = Whh_T @ dh
        zero_double(dh1next, hidden_size);
        mul_mat_col(hidden_size, hidden_size, Wh1h1_T, dh1, dh1next);
    }

    // clip delta values to prevent values from exploding
    clip_double(dWxh1, hidden_size * vocab_size, 5.0);
    clip_double(dWh1h1, hidden_size * hidden_size, 5.0);
    clip_double(dWh2y, vocab_size * hidden_size, 5.0);
    clip_double(dbh1, hidden_size, 5.0);
    clip_double(dby, vocab_size, 5.0);

    // save latest values of hidden layer for next training iteration
    copy_double(h1prev, h1s[seq_length - 1], hidden_size);

    // track loss smoothly, as it can vary wildly between iterations
    smooth_loss = smooth_loss * 0.999 + loss * 0.001;

    // update weights and biases using deltas:
    // memory += delta^2
    // value += -rate * delta / sqrt(mem + 1e-8)
    learn_double(hidden_size * vocab_size, dWxh1, mWxh1, Wxh1, learning_rate);
    learn_double(hidden_size * hidden_size, dWh1h1, mWh1h1, Wh1h1, learning_rate);
    learn_double(vocab_size * hidden_size, dWh2y, mWh2y, Wh2y, learning_rate);
    learn_double(hidden_size, dbh1, mbh1, bh1, learning_rate);
    learn_double(vocab_size, dby, mby, by, learning_rate);

    p += seq_length;
    n++;
}

static void iteration_v2() {
    // compute transposes here since matrices are not modified until at end of loop
    transpose_mat(vocab_size, hidden_size, Wh2y, Wh2y_T);
    transpose_mat(hidden_size, hidden_size, Wh2h2, Wh2h2_T);
    transpose_mat(hidden_size, hidden_size, Wh1h2, Wh1h2_T);
    transpose_mat(hidden_size, hidden_size, Wh1h1, Wh1h1_T);

    // if at end of input, reset to start and reset hidden layer to zeroes
    if (p + seq_length + 1 >= input_len) {
        zero_double(h1prev, hidden_size);
        zero_double(h2prev, hidden_size);
        p = 0;
    }

    // input = input_data[p : p+seq_length]
    const symbol_t *inputs = &input_data[p];
    // training output = input_data[p+1 : p+seq_length+1]
    const symbol_t *targets = &input_data[p + 1];

    // initialize loss at 0
    double loss = 0.0;
    // initialize hidden layer at previous training iteration's value
    copy_double(ph1s[0], h1prev, hidden_size);
    copy_double(ph2s[0], h2prev, hidden_size);

    // t is the character position in the training sequence
    for (size_t t = 0; t < seq_length; t++) {
        // set input neurons for the round, xs[t], to 1 for the current input char
        zero_double(xs[t], vocab_size);
        xs[t][inputs[t]] = 1;

        // compute new values of hidden1 layer
        // hs[t] = Wxh1 @ xs[t] + Whh @ h1s[t-1] + bh1
        copy_double(h1s[t], bh1, hidden_size);
        mul_mat_col(hidden_size, vocab_size, Wxh1, xs[t], h1s[t]);
        mul_mat_col(hidden_size, hidden_size, Wh1h1, ph1s[t], h1s[t]);
        // apply activation function to hidden1 layer
        // hs[t] = tanh(hs[t])
        tanh_vec(hidden_size, h1s[t]);

        // repeat previous steps for hidden2 layer
        copy_double(h2s[t], bh2, hidden_size);
        mul_mat_col(hidden_size, hidden_size, Wh1h2, h1s[t], h2s[t]);
        mul_mat_col(hidden_size, hidden_size, Wh2h2, ph2s[t], h2s[t]);
        tanh_vec(hidden_size, h2s[t]);

        // compute new values of output layer
        // ys[t] = Why @ hs[t] + by
        copy_double(ys[t], by, vocab_size);
        mul_mat_col(vocab_size, hidden_size, Wh2y, h2s[t], ys[t]);

        // compute the probabilities of each char from the output layer values
        // ps[t] = exp(ps[t]) / sum(exp(ps[t]))
        vec_to_exp_probs(vocab_size, ys[t], ps[t]);

        // compute the loss
        // loss += -log(ps[t][targets[t]])
        loss += -log(ps[t][targets[t]]);
    }

    // initialize deltas to zero
    zero_double(dWxh1, hidden_size * vocab_size);
    zero_double(dWh1h1, hidden_size * hidden_size);
    zero_double(dWh1h2, hidden_size * hidden_size);
    zero_double(dWh2h2, hidden_size * hidden_size);
    zero_double(dWh2y, vocab_size * hidden_size);
    zero_double(dbh1, hidden_size);
    zero_double(dbh2, hidden_size);
    zero_double(dby, vocab_size);
    zero_double(dh1next, hidden_size);
    zero_double(dh2next, hidden_size);

    // compute deltas backward from end of training sequence
    for (size_t t = seq_length - 1; t != (size_t) -1; t--) {
        // output delta = difference from "everything 0 except the correct char"
        copy_double(dy, ps[t], vocab_size);
        dy[targets[t]] -= 1;

        // backpropagate output delta to hidden2->output weights and output bias
        // dWh2y += dy @ h2s[t]
        mul_col_row(vocab_size, hidden_size, dy, h2s[t], dWh2y);
        // dby += dy
        add_double(vocab_size, dy, dby);

        // hidden2 delta = hidden2 delta from next char + delta from output
        // dh2 = Wh2y_T @ dy + dh2next
        copy_double(dh2, dh2next, hidden_size);
        mul_mat_col(hidden_size, vocab_size, Wh2y_T, dy, dh2);

        // reverse activation function
        // dh2 = (1 - h2s[t]^2) * dh2
        rev_tanh_vec(hidden_size, dh2, h2s[t]);

        // backpropagate hidden2 delta to hidden2->hidden2, hidden1->hidden2 weights and hidden bias
        // dbh2 += dh2
        add_double(hidden_size, dh2, dbh2);
        // dWh1h2 += dh2 @ h1s[t]
        mul_col_row(hidden_size, hidden_size, dh2, h1s[t], dWh1h2);
        // dWh2h2 += dh2 @ h2s[t-1]
        mul_col_row(hidden_size, hidden_size, dh2, ph2s[t], dWh2h2);

        // repeat previous steps for hidden1 layer
        // dh1 = Wh1h2_T @ dh2 + dh1next
        copy_double(dh1, dh1next, hidden_size);
        mul_mat_col(hidden_size, vocab_size, Wh1h2_T, dh2, dh1);
        // dh1 = (1 - h1s[t]^2) * dh1
        rev_tanh_vec(hidden_size, dh1, h1s[t]);
        // dbh1 += dh1
        add_double(hidden_size, dh1, dbh1);
        // dWxh1 += dh1 @ xs[t]
        mul_col_row(hidden_size, vocab_size, dh1, xs[t], dWxh1);
        // dWh1h1 += dh1 @ h1s[t-1]
        mul_col_row(hidden_size, hidden_size, dh1, ph1s[t], dWh1h1);

        // compute hidden delta for prev char from current hidden delta
        // dh1next = Wh1h1_T @ dh1
        zero_double(dh1next, hidden_size);
        mul_mat_col(hidden_size, hidden_size, Wh1h1_T, dh1, dh1next);
        // dh2next = Wh2h2_T @ dh2
        zero_double(dh2next, hidden_size);
        mul_mat_col(hidden_size, hidden_size, Wh2h2_T, dh2, dh2next);
    }

    // clip delta values to prevent values from exploding
    clip_double(dWxh1, hidden_size * vocab_size, 5.0);
    clip_double(dWh1h1, hidden_size * hidden_size, 5.0);
    clip_double(dWh1h2, hidden_size * hidden_size, 5.0);
    clip_double(dWh2h2, hidden_size * hidden_size, 5.0);
    clip_double(dWh2y, vocab_size * hidden_size, 5.0);
    clip_double(dbh1, hidden_size, 5.0);
    clip_double(dbh2, hidden_size, 5.0);
    clip_double(dby, vocab_size, 5.0);

    // save latest values of hidden layer for next training iteration
    copy_double(h1prev, h1s[seq_length - 1], hidden_size);
    copy_double(h2prev, h2s[seq_length - 1], hidden_size);

    // track loss smoothly, as it can vary wildly between iterations
    smooth_loss = smooth_loss * 0.999 + loss * 0.001;

    // update weights and biases using deltas:
    // memory += delta^2
    // value += -rate * delta / sqrt(mem + 1e-8)
    learn_double(hidden_size * vocab_size, dWxh1, mWxh1, Wxh1, learning_rate);
    learn_double(hidden_size * hidden_size, dWh1h1, mWh1h1, Wh1h1, learning_rate);
    learn_double(hidden_size * hidden_size, dWh1h2, mWh1h2, Wh1h2, learning_rate);
    learn_double(hidden_size * hidden_size, dWh2h2, mWh2h2, Wh2h2, learning_rate);
    learn_double(vocab_size * hidden_size, dWh2y, mWh2y, Wh2y, learning_rate);
    learn_double(hidden_size, dbh1, mbh1, bh1, learning_rate);
    learn_double(hidden_size, dbh2, mbh2, bh2, learning_rate);
    learn_double(vocab_size, dby, mby, by, learning_rate);

    p += seq_length;
    n++;
}

static inline double choose_by_probs(const double *p) {
    double value = randdouble();
    symbol_t symbol = 0;
    while (symbol < vocab_size - 1 && value >= p[symbol]) {
        value -= p[symbol];
        symbol++;
    }
    return symbol;
}

static void sample_v1(size_t length, wwchar_t *to, symbol_t seed) {
    double *x = xs[0];
    double *h = h1s[0];
    double *ph = ph1s[0];
    double *y = ys[0];
    double *p = ps[0];
    copy_double(ph, h1prev, hidden_size);
    zero_double(x, vocab_size);
    for (size_t i = 0; i < length; i++) {
        x[seed] = 1;
        // h = Wxh @ x + Whh @ ph + bh
        copy_double(h, bh1, hidden_size);
        mul_mat_col(hidden_size, vocab_size, Wxh1, x, h);
        mul_mat_col(hidden_size, hidden_size, Wh1h1, ph, h);
        // h = tanh(h)
        tanh_vec(hidden_size, h);
        // y = Why @ h + by
        copy_double(y, by, vocab_size);
        mul_mat_col(vocab_size, hidden_size, Wh2y, h, y);
        // p = exp(p) / sum(exp(p))
        vec_to_exp_probs(vocab_size, y, p);

        x[seed] = 0;

        seed = choose_by_probs(p);

        wwchar_t codepoint;
        if (!map_get_codepoint(&vocabulary, seed, &codepoint)) fail("missing codepoint???");
        to[i] = codepoint;

        // ph = h
        copy_double(ph, h, hidden_size);
    }
}

static void sample_v2(size_t length, wwchar_t *to, symbol_t seed) {
    double *x = xs[0];
    double *h1 = h1s[0];
    double *ph1 = ph1s[0];
    double *h2 = h2s[0];
    double *ph2 = ph2s[0];
    double *y = ys[0];
    double *p = ps[0];
    copy_double(ph1, h1prev, hidden_size);
    copy_double(ph2, h2prev, hidden_size);
    zero_double(x, vocab_size);
    for (size_t i = 0; i < length; i++) {
        x[seed] = 1;
        // h1 = Wxh1 @ x + Wh1h1 @ ph1 + bh1
        copy_double(h1, bh1, hidden_size);
        mul_mat_col(hidden_size, vocab_size, Wxh1, x, h1);
        mul_mat_col(hidden_size, hidden_size, Wh1h1, ph1, h1);
        // h1 = tanh(h1)
        tanh_vec(hidden_size, h1);
        // h2 = Wh1h2 @ h1 + Wh2h2 @ ph2 + bh2
        copy_double(h2, bh2, hidden_size);
        mul_mat_col(hidden_size, hidden_size, Wh1h2, h1, h2);
        mul_mat_col(hidden_size, hidden_size, Wh2h2, ph2, h2);
        // h2 = tanh(h2)
        tanh_vec(hidden_size, h2);
        // y = Wh2y @ h2 + by
        copy_double(y, by, vocab_size);
        mul_mat_col(vocab_size, hidden_size, Wh2y, h2, y);
        // p = exp(p) / sum(exp(p))
        vec_to_exp_probs(vocab_size, y, p);

        x[seed] = 0;

        seed = choose_by_probs(p);

        wwchar_t codepoint;
        if (!map_get_codepoint(&vocabulary, seed, &codepoint)) fail("missing codepoint???");
        to[i] = codepoint;

        // ph1 = h1
        copy_double(ph1, h1, hidden_size);
        // ph2 = h2
        copy_double(ph2, h2, hidden_size);
    }
}

int main(int argc, const char **argv) {
    srand(time(NULL));
    signal(SIGINT, ignore_signal);

    if (argc > 1 && strcmp(argv[1], "--resume") == 0) {
        init_from_stdin();
    } else {
        init_from_text();
    }

    fd_set fdset;
    struct timeval timeout;
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;

    wwchar_t *sample_ww;
    char *sample_utf8;
    size_t sample_len_utf8;

    while (1) {
        iteration_func();

        FD_ZERO(&fdset);
        FD_SET(fileno(stdin), &fdset);

        if (select(FD_SETSIZE, &fdset, NULL, NULL, &timeout) == 1) {
            switch (fgetc(stdin)) {
            case 's':
                sample_ww = alloc_wwchar(sample_length + 1);

                sample_func(sample_length, sample_ww, input_data[p]);
                sample_ww[sample_length] = 0;
                
                sample_len_utf8 = wwutf8len(sample_ww);
                sample_utf8 = check_alloc(sample_len_utf8 + 1);
                
                if (!wwtoutf8(sample_utf8, sample_ww)) fail("failed to convert to utf-8");
                
                fwrite(sample_utf8, 1, sample_len_utf8 + 1, stdout);
                write_typed(uint64_t, &sample_sanity, 1);
                fflush(stdout);
                
                free(sample_ww);
                free(sample_utf8);
                break;
            case 'i':
                fwrite(&n, sizeof(size_t), 1, stdout);
                fwrite(&smooth_loss, sizeof(double), 1, stdout);
                write_typed(uint64_t, &iter_sanity, 1);
                fflush(stdout);
                break;
            case EOF:
            case 'q':
                dump_to_stdout();
                fflush(stdout);
                free_model();
                return 0;
            }
        }
    }
}
