#ifndef MAP_H
#define MAP_H

#include <stdint.h>
#include <stdlib.h>

#include "rnn.h"
#include "unicode.h"

#define MAP_NODE_BITS 4

typedef union map_tree_node {
    union map_tree_node *subtree;
    struct {
        bool present;
        union {
            symbol_t symbol;
            wwchar_t codepoint;
        };
    };
} map_tree_node;

void map_new(map_tree_node *map);

void map_free(map_tree_node *map, size_t key_size);

void map_put_symbol(map_tree_node *map, wwchar_t key, symbol_t symbol);

bool map_get_symbol(const map_tree_node *map, wwchar_t key, symbol_t *value);

bool map_get_or_set_symbol(map_tree_node *map, wwchar_t key, symbol_t *value, symbol_t symbol);

void map_put_codepoint(map_tree_node *map, symbol_t key, wwchar_t codepoint);

bool map_get_codepoint(const map_tree_node *map, symbol_t key, wwchar_t *value);

bool map_get_or_set_codepoint(map_tree_node *map, symbol_t key, wwchar_t *value, wwchar_t codepoint);

#endif
