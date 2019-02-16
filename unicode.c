#include "unicode.h"

size_t utf8strlen(const char *str) {
    size_t i = 0;
    char ch;
    while ((ch = *str++)) {
        if ((ch & 0x80) && *str) {
            str++;
            if ((ch & 0x20) && *str) {
                str++;
                if ((ch & 0x10) && *str) {
                    str++;
                }
            }
        }
        i++;
    }
    return i;
}

size_t wwstrlen(const wwchar_t *str) {
    size_t i = 0;
    while (*str++) i++;
    return i;
}

size_t wwutf8len(const wwchar_t *str) {
    size_t i = 0;
    wwchar_t ch;
    while ((ch = *str++)) i += 1 + !!(ch & ~0x7f) + !!(ch & ~0x7ff) + !!(ch & ~0xffff);
    return i;
}

bool utf8toww(wwchar_t *dst, const char *src) {
    wwchar_t wch;
    char ch, cnt;
    while ((ch = *src++)) {
        wch = ch;
        if (ch & 0x80) {
            if (ch & 0x40) {
                if (((cnt = *src++) & 0xc0) != 0x80) return false;
                wch = ((wch & 0x1f) << 6) | (cnt & 0x3f);
                if (ch & 0x20) {
                    if (((cnt = *src++) & 0xc0) != 0x80) return false;
                    wch = ((wch & 0x7ff) << 6) | (cnt & 0x3f);
                    if (ch & 0x10) {
                        if (((cnt = *src++) & 0xc0) != 0x80) return false;
                        wch = ((wch & 0xffff) << 6) | (cnt & 0x3f);
                        if (!(wch & ~0xffff)) return false;
                    } else if (!(wch & 0x7ff)) return false;
                } else if (!(wch & 0x7f)) return false;
            } else return false;
        }
        *dst++ = wch;
    }
    return true;
}

bool wwtoutf8(char *dst, const wwchar_t *src) {
    wwchar_t ch;
    while ((ch = *src++)) {
        if (ch > 0x10ffff) return false;
        if (ch & ~0xffff) {
            *dst++ = 0xf0 | (ch >> 18);
            *dst++ = 0x80 | ((ch >> 12) & 0x3f);
            *dst++ = 0x80 | ((ch >> 6) & 0x3f);
            *dst++ = 0x80 | (ch & 0x3f);
        } else if (ch & ~0x7ff) {
            *dst++ = 0xe0 | (ch >> 12);
            *dst++ = 0x80 | ((ch >> 6) & 0x3f);
            *dst++ = 0x80 | (ch & 0x3f);
        } else if (ch & ~0x7f) {
            *dst++ = 0xc0 | (ch >> 6);
            *dst++ = 0x80 | (ch & 0x3f);
        } else {
            *dst++ = ch;
        }
    }
    *dst++ = 0;
    return true;
}
