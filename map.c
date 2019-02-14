#include "map.h"

void map_new(map_tree_node *map) {
    map->subtree = NULL;
}

static void map_free_inner(map_tree_node *node, size_t level) {
    if (level == 0 || node->subtree == NULL) return;
    for (size_t i = 0; i < (1 << MAP_NODE_BITS); i++) {
        map_free_inner(&node->subtree[i], level - 1);
    }
    free(node->subtree);
}

void map_free(map_tree_node *map, size_t key_len) {
    map_free_inner(map, key_len * 8 / MAP_NODE_BITS);
}

void map_put_symbol(map_tree_node *map, wwchar_t key, symbol_t symbol) {
    map_tree_node *node = map;
    size_t index, i;
    for (i = 0; i < sizeof(wwchar_t) * 8; i += MAP_NODE_BITS) {
        if (node->subtree == NULL) {
            node->subtree = calloc(1 << MAP_NODE_BITS, sizeof(map_tree_node));
            if (node->subtree == NULL) fail("failed to allocate map node");
        }
        index = key & ((1 << MAP_NODE_BITS) - 1);
        key >>= MAP_NODE_BITS;
        node = &node->subtree[index];
    }
    node->symbol = symbol;
    node->present = true;
}

void map_put_codepoint(map_tree_node *map, symbol_t key, wwchar_t codepoint) {
    map_tree_node *node = map;
    size_t index, i;
    for (i = 0; i < sizeof(symbol_t) * 8; i += MAP_NODE_BITS) {
        if (node->subtree == NULL) {
            node->subtree = calloc(1 << MAP_NODE_BITS, sizeof(map_tree_node));
            if (node->subtree == NULL) fail("failed to allocate map node");
        }
        index = key & ((1 << MAP_NODE_BITS) - 1);
        key >>= MAP_NODE_BITS;
        node = &node->subtree[index];
    }
    node->codepoint = codepoint;
    node->present = true;
}

bool map_get_symbol(const map_tree_node *map, wwchar_t key, symbol_t *value) {
    const map_tree_node *node = map;
    size_t index, i;
    for (i = 0; i < sizeof(symbol_t) * 8; i += MAP_NODE_BITS) {
        if (node->subtree == NULL) return false;
        index = key & ((1 << MAP_NODE_BITS) - 1);
        key >>= MAP_NODE_BITS;
        node = &node->subtree[index];
    }
    if (!node->present) return false;
    *value = node->codepoint;
    return true;
}

bool map_get_codepoint(const map_tree_node *map, symbol_t key, wwchar_t *value) {
    const map_tree_node *node = map;
    size_t index, i;
    for (i = 0; i < sizeof(symbol_t) * 8; i += MAP_NODE_BITS) {
        if (node->subtree == NULL) return false;
        index = key & ((1 << MAP_NODE_BITS) - 1);
        key >>= MAP_NODE_BITS;
        node = &node->subtree[index];
    }
    if (!node->present) return false;
    *value = node->codepoint;
    return true;
}

bool map_get_or_set_symbol(map_tree_node *map, wwchar_t key, symbol_t *value, symbol_t symbol) {
    map_tree_node *node = map;
    size_t index, i;
    for (i = 0; i < sizeof(wwchar_t) * 8; i += MAP_NODE_BITS) {
        if (node->subtree == NULL) {
            node->subtree = calloc(1 << MAP_NODE_BITS, sizeof(map_tree_node));
            if (node->subtree == NULL) fail("failed to allocate map node");
        }
        index = key & ((1 << MAP_NODE_BITS) - 1);
        key >>= MAP_NODE_BITS;
        node = &node->subtree[index];
    }
    if (!node->present) {
        *value = node->symbol = symbol;
        node->present = true;
        return true;
    }
    *value = node->symbol;
    return false;
}

bool map_get_or_set_codepoint(map_tree_node *map, symbol_t key, wwchar_t *value, wwchar_t codepoint) {
    map_tree_node *node = map;
    size_t index, i;
    for (i = 0; i < sizeof(wwchar_t) * 8; i += MAP_NODE_BITS) {
        if (node->subtree == NULL) {
            node->subtree = calloc(1 << MAP_NODE_BITS, sizeof(map_tree_node));
            if (node->subtree == NULL) fail("failed to allocate map node");
        }
        index = key & ((1 << MAP_NODE_BITS) - 1);
        key >>= MAP_NODE_BITS;
        node = &node->subtree[index];
    }
    if (!node->present) {
        *value = node->codepoint = codepoint;
        node->present = true;
        return true;
    }
    *value = node->codepoint;
    return false;
}
