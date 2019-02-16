#ifndef RNN_H
#define RNN_H

typedef size_t symbol_t;

__attribute((noreturn)) void _fail(const char *reason, const char *file, uint32_t line);

#define fail(reason) _fail(reason, __FILE__, __LINE__)

#endif
