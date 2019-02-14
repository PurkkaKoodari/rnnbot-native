#ifndef UNICODE_H
#define UNICODE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef uint32_t wwchar_t;

/// gets the length of str, assuming valid UTF-8
size_t utf8strlen(const char *str);

/// gets the length of str
size_t wwstrlen(const wwchar_t *str);

/// gets the length of str when converted to UTF-8
size_t wwutf8len(const wwchar_t *str);

/// converts src from UTF-8, returns if valid
bool utf8toww(wwchar_t *dst, const char *src);

/// converts src to UTF-8, returns if valid
bool wwtoutf8(char *dst, const wwchar_t *src);

#endif
