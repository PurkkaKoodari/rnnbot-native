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

#define DUMP_SANITY 0x65746174534e4e52 // RNNState

__attribute((noreturn)) void fail(const char *reason) {
    fprintf(stderr, "%s\n", reason);
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

static void check_read(void *ptr, size_t size, size_t n) {
    if (fread(ptr, size, n, stdin) != n) fail("failed to read data");
}

#define read_typed(type, ptr, n) check_read(ptr, sizeof(type), n)
#define read_double(ptr, n) read_typed(double, ptr, n)
#define read_size_t(ptr, n) read_typed(size_t, ptr, n)
#define read_symbol(ptr, n) read_typed(symbol_t, ptr, n)
#define read_wwchar(ptr, n) read_typed(wwchar_t, ptr, n)

static void check_write(void *ptr, size_t size, size_t n) {
    if (fwrite(ptr, size, n, stdout) != n) fail("failed to write data");
}

#define write_typed(type, ptr, n) check_write(ptr, sizeof(type), n)
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

static const size_t hidden_size = 100;
static const size_t seq_length = 25;
static const double learning_rate = 1e-1;

static const size_t sample_length = 200;

static double *Wxh;
static double *Whh;
static double *Why;
static double *bh;
static double *by;

static double *Why_T;
static double *Whh_T;

static double *mWxh;
static double *mWhh;
static double *mWhy;
static double *mbh;
static double *mby;

static double *dWxh;
static double *dWhh;
static double *dWhy;
static double *dbh;
static double *dby;

static double *hprev;
static double **xs;
static double **phs;
static double **hs;
static double **ys;
static double **ps;

static double *dhnext;
static double *dy;
static double *dh;

static double smooth_loss;

static map_tree_node vocabulary;

static size_t vocab_size;

static symbol_t *input_data;
static size_t input_len;

static size_t n, p;

static void alloc_model() {
    Wxh = alloc_double(hidden_size * vocab_size);
    Whh = alloc_double(hidden_size * hidden_size);
    Why = alloc_double(vocab_size * hidden_size);
    bh = alloc_double(hidden_size);
    by = alloc_double(vocab_size);

    Why_T = alloc_double(vocab_size * hidden_size);
    Whh_T = alloc_double(hidden_size * hidden_size);

    mWxh = alloc_double(hidden_size * vocab_size);
    mWhh = alloc_double(hidden_size * hidden_size);
    mWhy = alloc_double(vocab_size * hidden_size);
    mbh = alloc_double(hidden_size);
    mby = alloc_double(vocab_size);

    hprev = alloc_double(hidden_size);

    xs = alloc_2d_double(seq_length, vocab_size);
    phs = alloc_2d_double(seq_length + 1, hidden_size);
    hs = &phs[1];
    ys = alloc_2d_double(seq_length, vocab_size);
    ps = alloc_2d_double(seq_length, vocab_size);

    dWxh = alloc_double(hidden_size * vocab_size);
    dWhh = alloc_double(hidden_size * hidden_size);
    dWhy = alloc_double(vocab_size * hidden_size);
    dbh = alloc_double(hidden_size);
    dby = alloc_double(vocab_size);
    dhnext = alloc_double(hidden_size);
    dy = alloc_double(vocab_size);
    dh = alloc_double(hidden_size);
}

static void free_model() {
    free(Wxh);
    free(Whh);
    free(Why);
    free(bh);
    free(by);

    free(Whh_T);
    free(Why_T);

    free(mWxh);
    free(mWhh);
    free(mWhy);
    free(mbh);
    free(mby);

    free(hprev);

    free_2d((void **) xs, seq_length);
    free_2d((void **) phs, seq_length + 1);
    free_2d((void **) ys, seq_length);
    free_2d((void **) ps, seq_length);

    free(dWxh);
    free(dWhh);
    free(dWhy);
    free(dbh);
    free(dby);
    free(dhnext);
    free(dy);
    free(dh);

    map_free(&vocabulary, sizeof(symbol_t));
}

static void init_from_text() {
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

    alloc_model();

    nrand_double(Wxh, hidden_size * vocab_size);
    nrand_double(Whh, hidden_size * hidden_size);
    nrand_double(Why, vocab_size * hidden_size);
    zero_double(bh, hidden_size);
    zero_double(by, vocab_size);

    zero_double(mWxh, hidden_size * vocab_size);
    zero_double(mWhh, hidden_size * hidden_size);
    zero_double(mWhy, vocab_size * hidden_size);
    zero_double(mbh, hidden_size);
    zero_double(mby, vocab_size);

    zero_double(hprev, hidden_size);
    
    n = 0;
    p = 0;

    smooth_loss = -log(1.0 / vocab_size) * seq_length;
}

static void init_from_stdin() {
    read_size_t(&vocab_size, 1);
    read_size_t(&input_len, 1);
    
    map_new(&vocabulary);

    wwchar_t codepoint;
    for (symbol_t next_symbol = 0; next_symbol < vocab_size; next_symbol++) {
        read_wwchar(&codepoint, 1);
        map_put_codepoint(&vocabulary, next_symbol, codepoint);
    }

    input_data = alloc_symbol(input_len);
    read_symbol(input_data, input_len);

    alloc_model();

    read_double(Wxh, hidden_size * vocab_size);
    read_double(Whh, hidden_size * hidden_size);
    read_double(Why, vocab_size * hidden_size);
    read_double(bh, hidden_size);
    read_double(by, vocab_size);

    read_double(mWxh, hidden_size * vocab_size);
    read_double(mWhh, hidden_size * hidden_size);
    read_double(mWhy, vocab_size * hidden_size);
    read_double(mbh, hidden_size);
    read_double(mby, vocab_size);

    read_double(hprev, hidden_size);

    read_double(&n, 1);
    read_double(&p, 1);

    read_double(&smooth_loss, 1);

    // sanity check
    uint64_t sanity;
    read_typed(uint64_t, &sanity, 1);
    if (sanity != DUMP_SANITY) fail("dump length mismatch");
}

static void dump_to_stdout() {
    write_size_t(&vocab_size, 1);
    write_size_t(&input_len, 1);
    
    wwchar_t codepoint;
    for (symbol_t next_symbol = 0; next_symbol < vocab_size; next_symbol++) {
        map_get_codepoint(&vocabulary, next_symbol, &codepoint);
        write_wwchar(&codepoint, 1);
    }

    write_symbol(input_data, input_len);

    write_double(Wxh, hidden_size * vocab_size);
    write_double(Whh, hidden_size * hidden_size);
    write_double(Why, vocab_size * hidden_size);
    write_double(bh, hidden_size);
    write_double(by, vocab_size);

    write_double(mWxh, hidden_size * vocab_size);
    write_double(mWhh, hidden_size * hidden_size);
    write_double(mWhy, vocab_size * hidden_size);
    write_double(mbh, hidden_size);
    write_double(mby, vocab_size);

    write_double(hprev, hidden_size);

    write_double(&n, 1);
    write_double(&p, 1);

    write_double(&smooth_loss, 1);

    uint64_t sanity = DUMP_SANITY;
    write_typed(uint64_t, &sanity, 1);
}

static void iteration() {
    // compute transposes here since matrices are not modified until at end of loop
    transpose_mat(vocab_size, hidden_size, Why, Why_T);
    transpose_mat(hidden_size, hidden_size, Whh, Whh_T);

    if (p + seq_length + 1 >= input_len) {
        zero_double(hprev, hidden_size);
        p = 0;
    }

    const symbol_t *inputs = &input_data[p];
    const symbol_t *targets = &input_data[p + 1];

    double loss = 0.0;
    copy_double(phs[0], hprev, hidden_size);
    for (size_t t = 0; t < seq_length; t++) {
        zero_double(xs[t], vocab_size);
        xs[t][inputs[t]] = 1;
        // hs[t] = Wxh @ xs[t] + Whh @ hs[t-1] + bh
        copy_double(hs[t], bh, hidden_size);
        mul_mat_col(hidden_size, vocab_size, Wxh, xs[t], hs[t]);
        mul_mat_col(hidden_size, hidden_size, Whh, phs[t], hs[t]);
        // hs[t] = tanh(hs[t])
        tanh_vec(hidden_size, hs[t]);
        // ys[t] = Why @ hs[t] + by
        copy_double(ys[t], by, vocab_size);
        mul_mat_col(vocab_size, hidden_size, Why, hs[t], ys[t]);
        // ps[t] = exp(ps[t]) / sum(exp(ps[t]))
        vec_to_exp_probs(vocab_size, ys[t], ps[t]);
        // loss += -log(ps[t][targets[t]])
        loss += -log(ps[t][targets[t]]);
    }

    zero_double(dWxh, hidden_size * vocab_size);
    zero_double(dWhh, hidden_size * hidden_size);
    zero_double(dWhy, vocab_size * hidden_size);
    zero_double(dbh, hidden_size);
    zero_double(dby, vocab_size);
    zero_double(dhnext, hidden_size);

    for (size_t t = seq_length - 1; t != (size_t) -1; t--) {
        copy_double(dy, ps[t], vocab_size);
        dy[targets[t]] -= 1;
        // dWhy += dy @ hs[t]
        mul_col_row(vocab_size, hidden_size, dy, hs[t], dWhy);
        // dby += dy
        add_double(vocab_size, dy, dby);
        // dh = Why_T @ dy + dhnext
        copy_double(dh, dhnext, hidden_size);
        mul_mat_col(hidden_size, vocab_size, Why_T, dy, dh);
        // dh = (1 - hs[t]^2) * dh
        rev_tanh_vec(hidden_size, dh, hs[t]);
        // dbh += dh
        add_double(hidden_size, dh, dbh);
        // dWxh += dh @ xs[t]
        mul_col_row(hidden_size, vocab_size, dh, xs[t], dWxh);
        // dWhh += dh @ hs[t-1]
        mul_col_row(hidden_size, hidden_size, dh, phs[t], dWhh);
        // dhnext = Whh_T @ dh
        zero_double(dhnext, hidden_size);
        mul_mat_col(hidden_size, hidden_size, Whh_T, dh, dhnext);
    }

    clip_double(dWxh, hidden_size * vocab_size, 5.0);
    clip_double(dWhh, hidden_size * hidden_size, 5.0);
    clip_double(dWhy, vocab_size * hidden_size, 5.0);
    clip_double(dbh, hidden_size, 5.0);
    clip_double(dby, vocab_size, 5.0);

    copy_double(hprev, hs[seq_length - 1], hidden_size);

    smooth_loss = smooth_loss * 0.999 + loss * 0.001;

    learn_double(hidden_size * vocab_size, dWxh, mWxh, Wxh, learning_rate);
    learn_double(hidden_size * hidden_size, dWhh, mWhh, Whh, learning_rate);
    learn_double(vocab_size * hidden_size, dWhy, mWhy, Why, learning_rate);
    learn_double(hidden_size, dbh, mbh, bh, learning_rate);
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

static void sample(size_t length, wwchar_t *to, symbol_t seed) {
    double *x = xs[0];
    double *h = hs[0];
    double *ph = phs[0];
    double *y = ys[0];
    double *p = ps[0];
    copy_double(ph, hprev, hidden_size);
    zero_double(x, vocab_size);
    for (size_t i = 0; i < length; i++) {
        x[seed] = 1;
        // h = Wxh @ x + Whh @ ph + bh
        copy_double(h, bh, hidden_size);
        mul_mat_col(hidden_size, vocab_size, Wxh, x, h);
        mul_mat_col(hidden_size, hidden_size, Whh, ph, h);
        // h = tanh(h)
        tanh_vec(hidden_size, h);
        // y = Why @ h + by
        copy_double(y, by, vocab_size);
        mul_mat_col(vocab_size, hidden_size, Why, h, y);
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

int main(int argc, const char **argv) {
    srand(time(NULL));
    signal(SIGINT, ignore_signal);

    if (argc > 1 && strcmp(argv[1], "--resume") == 0) {
        init_from_stdin();
    } else {
        init_from_text();
    }

    fprintf(stderr, "input size %lu\n", input_len);
    fprintf(stderr, "vocabulary size %lu\n", vocab_size);
    fprintf(stderr, "iteration %lu\n", n);

    fd_set fdset;
    struct timeval timeout;
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;

    wwchar_t *sample_ww;
    char *sample_utf8;
    size_t sample_len_utf8;

    while (1) {
        iteration();

        FD_ZERO(&fdset);
        FD_SET(fileno(stdin), &fdset);

        if (select(FD_SETSIZE, &fdset, NULL, NULL, &timeout) == 1) {
            switch (fgetc(stdin)) {
            case 's':
                sample_ww = alloc_wwchar(sample_length + 1);

                sample(sample_length, sample_ww, input_data[p]);
                sample_ww[sample_length] = 0;
                
                sample_len_utf8 = wwutf8len(sample_ww);
                sample_utf8 = check_alloc(sample_len_utf8 + 1);
                
                if (!wwtoutf8(sample_utf8, sample_ww)) fail("failed to convert to utf-8");
                
                fwrite(sample_utf8, 1, sample_len_utf8 + 1, stdout);
                fflush(stdout);
                
                free(sample_ww);
                free(sample_utf8);
                break;
            case 'i':
                fwrite(&n, sizeof(size_t), 1, stdout);
                fwrite(&smooth_loss, sizeof(double), 1, stdout);
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
