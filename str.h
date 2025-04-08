/**
 * lightweight string implementation in C that supports c_like strings.
 */
#pragma once

#include "common.h"

typedef void* str;

CTENSOR_API str    str_create(size_t capacity);
CTENSOR_API str    str_create_fixed(size_t capacity);
CTENSOR_API str    str_from_cstr(const char* cstr);

CTENSOR_API size_t str_available(str s);
CTENSOR_API size_t str_len(str s);
CTENSOR_API str    str_set_fixed(str s, bool value);

CTENSOR_API str    str_append_char(str s, char c);
CTENSOR_API str    str_append_string(str s, const char* c, size_t len);
CTENSOR_API str    str_append_cstr(str s, const char* c);
CTENSOR_API str    str_clear(str s);
CTENSOR_API void   str_free(str* s);

CTENSOR_API const char* str_cstr(str str);


#ifdef STR_IMPL

#define STR_INSTANCE(s) ((str_impl*)s)

typedef struct str_st str_impl;

struct str_st {
    u8* buff;
    size_t len;
    size_t capacity;
    bool fixed_cap;
};

static bool _str_needs_realloc(str_impl* str, size_t size);

CTENSOR_API str str_create(size_t capacity) {
    str_impl* s = malloc(sizeof(str_impl));
    s->buff     = NULL;
    s->len      = 0;
    s->capacity = 0;
    s->fixed_cap = false;

    if (capacity < 1)
        return s;

    s->buff = calloc(capacity + 1, sizeof(*s->buff));
    if (s->buff)
        s->capacity = capacity;

    return s;
}

CTENSOR_API str str_create_fixed(size_t capacity) {
    str_impl* s = str_create(capacity);
    s->fixed_cap = true;
    return s;
}

CTENSOR_API str str_clone(str s) {
    if (!s) return NULL;

    str_impl* src        = s;
    str_impl* s_instance = str_create(str_len(s));

    if (s_instance->buff) {
        memcpy(s_instance->buff, src->buff, src->capacity * sizeof(*src->buff));
        s_instance->len = src->len;
    }
    
    return s_instance;
}

CTENSOR_API str str_from_cstr(const char* cstr) {
    str_impl* s = str_create(strlen(cstr));
    str_append_cstr(s, cstr);
    return s;
}

CTENSOR_API str str_clear(str s) {
    str_impl* s_inst = s;
    if (s_inst->buff != NULL)
        s_inst->buff[0] = '\0';

    s_inst->len = 0;
    return s;
}

CTENSOR_API void   str_free(str* s) {
    if (!s || !(*s)) return;

    str_impl* s_impl = *s;
    if (s_impl->buff != NULL)
        free(s_impl->buff);

    free(*s);
    *s = NULL;
}


CTENSOR_API const char* str_cstr(str s) {
    return STR_INSTANCE(s)->buff;
}

CTENSOR_API size_t str_len(str s) {
    return s ? STR_INSTANCE(s)->len : 0;
}

CTENSOR_API str str_set_fixed(str s, bool value) {
    if (!s) return NULL;
    STR_INSTANCE(s)->fixed_cap = value;
    return s;
}

CTENSOR_API size_t str_available(str s) {
    str_impl* s_inst = s;
    ssize_t l = s_inst->capacity - s_inst->len;
    if (l < 0) return 0;
    return l;
}

CTENSOR_API str str_append_char(str s, char c) {
    if (!s) return NULL;
    if (!_str_needs_realloc(s, 1))
        return s;

    str_impl* s_inst = s;

    s_inst->buff[s_inst->len++] = c;
    return s;
}

CTENSOR_API str str_append_string(str s, const char* src, size_t len) {    
    if (!s) return NULL;

    if(!_str_needs_realloc(s, len))
        return s;
    
    str_impl* s_inst = s;

    memcpy(s_inst->buff + s_inst->len, src, len);
    s_inst->len += len;
    
    return s;
}

CTENSOR_API str str_append_cstr(str s, const char* src) {
    return str_append_string(s, src, strlen(src));
}

static bool _str_needs_realloc(str_impl* str, size_t size) {
    if (str->buff == NULL) {
        str->buff     = calloc(size + 1, sizeof(*(str->buff)));
        str->capacity = size;

        return str->buff != NULL;
    }

    if (str->fixed_cap && str_available(str) < size)
        return false;

    if (str_available(str) < size) {
        size_t curr_cap = str->capacity + 1;
        u8* buff = realloc(str->buff, curr_cap + size + 1);
        if (buff) {
            buff[curr_cap + size] = '\0'; // support for c_str as well.
            str->buff             = buff;
            str->capacity         = curr_cap + size;
            return true;
        }

        return false;
    }

    return true;
}

#endif // ifdef STR_IMPL