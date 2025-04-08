#pragma once

#include "common.h"
#include "str.h"
#include "vector.h"
#include <stdarg.h>


typedef struct tensor_st * tensor;
typedef float tensor_element_t;


struct tensor_st {
    tensor_element_t*  data;
    str     label;
    vector  shape;
    
    size_t  entries;
    vector  strides;

    str _dimBuff;
    str _dataBuff;
};

CTENSOR_API tensor tensor_create(vector shape);
CTENSOR_API bool tensor_is_null(tensor t);

CTENSOR_API tensor tensor_set_label(tensor t, str label);
CTENSOR_API tensor tensor_like(tensor t);
CTENSOR_API tensor tensor_scalar(tensor_element_t n);

CTENSOR_API void             tensor_set(tensor t, size_t indx, tensor_element_t value);
CTENSOR_API tensor_element_t tensor_get(tensor t, size_t indx);

CTENSOR_API size_t           tensor_entry(tensor t, vector tuple);
CTENSOR_API tensor_element_t tensor_entry_get(tensor t, vector tuple);
CTENSOR_API void             tensor_entry_set(tensor t, vector tuple, tensor_element_t v);


// basic shape to index mapping:
// i = k1*s[n-1]*...*s[1] + k2*s[n-1]*...*s[2] + ... + k(n-1)*s[n-1] + kn
// n in [0, shape.size - 1]

// converts one-dimensional tensor index into multi-dimensional index.
CTENSOR_API void tensor_index_to_shape(vector s, size_t ind);
CTENSOR_API tensor tensor_broadcast(tensor t, int dim, int n);

// tensor operations
CTENSOR_API tensor tensor_add(tensor a, tensor b);
CTENSOR_API tensor tensor_sub(tensor a, tensor b);
// CTENSOR_API void tensor_print(tensor t);

#define SHAPE(...) _tensor_shape(0, __VA_ARGS__, -2)
#define TUPLE(...) _tensor_shape(0, __VA_ARGS__, -2)

#define SCALAR(n)  tensor_scalar(n)

#ifdef TENSOR_IMPL

#define DIMLOG_BUFF_SIZE  512
#define DATALOG_BUFF_SIZE 1024

static void _tensor_prepare_dimStr(tensor t);
static void _tensor_prepare_dataStr(tensor t, char* dest, size_t size);
static size_t _vector_reduce_mul(vector v, size_t index);

CTENSOR_API vector _tensor_shape(int n, ...) {
    (void) n;
    vector shape = vector_init();

    va_list vl;
    va_start(vl, 0);
    
    while (true) {
        int x = va_arg(vl, int);
        if (x <= -2) break;
        vector_push_back(shape, &x, sizeof(x));
    }

    return shape;
}


CTENSOR_API tensor tensor_create(vector shape) {
    tensor t = NULL;
    t        = malloc(sizeof(*t));

    t->shape     = shape;
    t->label     = NULL;
    t->strides   = NULL;
    t->_dimBuff  = str_create_fixed(DIMLOG_BUFF_SIZE);
    t->_dataBuff = str_create_fixed(DATALOG_BUFF_SIZE);
    
    if (!t->shape)
        t->shape = vector_init();

    size_t dims    = vector_size(t->shape);
    size_t entries = 1;
    if (dims > 0) {
        for (int i = 0; i < dims; ++i)
            entries *= *(int*)vector_get(t->shape, i);
    }

    t->data    = calloc(entries, sizeof(*t->data));
    t->entries = entries;

    _tensor_prepare_dimStr(t);

    return t;
}

CTENSOR_API tensor tensor_scalar(float n) {
    tensor t = tensor_create(NULL);
    if (tensor_is_null(t)) 
        return NULL;
    
    *(t->data) = n;
    return t;
}

CTENSOR_API bool tensor_is_null(tensor t) {
    return t == NULL || t->data == NULL;
}

CTENSOR_API void tensor_set(tensor t, size_t indx, tensor_element_t value) {
    if (tensor_is_null(t))
        return;

    if (indx >= t->entries)
        return;

    t->data[indx] = value;
}

CTENSOR_API tensor_element_t tensor_get(tensor t, size_t indx) {
    if (tensor_is_null(t))
        return -1;

    if (indx >= t->entries)
        return -1;

    return t->data[indx];
}


static size_t _vector_reduce_mul(vector v, size_t index) {
    size_t n    = vector_size(v);
    size_t p    = 1;
    if (index >= n)
        return p;

    for (int i = index; i < n; ++i) 
        p *= *(size_t*)vector_get(v, i);
    
    return p;
}

// This function defines this: (example provided in pytorch just to get the point)
//      t = torch.tensor(...)
//      t[i1, i2, i3, ..., i_n] === tensor_entry(t, TUPLE(i1, i2, i3, ..., i_n))
//
// Implementation details: 
//    Since the underlaying tensor data is stored as a linear
//    array of floating-point numbers, this function maps a multi-dimensional
//    indexing tuple into a single index of the form:
//    indx = i1*m1 + i2*m2 + ..., i_n*1
//    where each m_i is defined as the products of all the shapes of tensor t
//    starting from i + 1, and these m values are referred to "strides" in the code.
//    This function only computes the strides if t->strides is set to NULL, then
//    they will be reused.
CTENSOR_API size_t tensor_entry(tensor t, vector tuple) {
    if (tensor_is_null(t)) return -1;

    size_t indx = 0;

    size_t tuple_size = vector_size(tuple);
    size_t shape_size = vector_size(t->shape);

    if (shape_size == 0)
        return *t->data;

    if (tuple_size != shape_size)
        return -1;

    if (!t->strides) {
        size_t strides[tuple_size];
        // generate strides coefficients m1, m2, ..., m_n.   
        for (int i = 0; i < tuple_size; ++i)
            strides[i] = _vector_reduce_mul(t->shape, i + 1);
        
        t->strides = vector_from_array(strides, sizeof(size_t), tuple_size);
    }

    // where m's are stride values.
    // TODO: Optimize this algorithm.
    for (int i = 0; i < tuple_size; ++i)
        indx += (*(int*)vector_get(tuple, i)) *  (*(int*)vector_get(t->strides, i));

    return indx;
}

CTENSOR_API tensor_element_t tensor_entry_get(tensor t, vector tuple) {
    return tensor_get(t, tensor_entry(t, tuple));
}

CTENSOR_API void tensor_entry_set(tensor t, vector tuple, tensor_element_t v) {
    return tensor_set(t, tensor_entry(t, tuple), v);
}


CTENSOR_API tensor tensor_broadcast(tensor t, int dim, int n) {
    // int* c = vector_get(t->shape, dim);
    // if (c == NULL) return NULL;

    // if (*c != 1)
    //     return NULL;

    // vector shape = vector_clone(t->shape);
    // vector_set(shape, dim, &n, sizeof(n));

    // tensor out  = tensor_create(shape);

    // vector indx_src = vector_clone(shape);
    // vector indx_out = vector_clone(shape);

    /*
    
    void tensor_set(tensor t, size_t indx, tensor_element_t value);
    tensor_element_t tensor_get(tensor t, size_t indx);
    tensor_index_to_shape
    */

    // int zero = 0;
    // for(int i = 0; i < out->entries; ++i) {
    //     tensor_index_to_shape(indx_out, i);
    //     vector_copy(indx_src, indx_out);
    //     // indx_src[dim] = 0
    //     vector_set(indx_src, dim, &zero, sizeof(zero));

    //     tensor_set(out, i, tensor_entry(t, indx_src));
    // }

    // vector_free(indx_src);
    // vector_free(indx_out);

    // return out;

    return NULL;
}

CTENSOR_API tensor tensor_like(tensor t) {
    vector shape = vector_clone(t->shape);
    if (!shape) return NULL;

    tensor out = tensor_create(shape);
    out->label = str_clone(t->label);
    if (out->label)
        str_append_cstr(out->label, " (Copy)");
    
    return out;
}

CTENSOR_API tensor tensor_set_label(tensor t, str label) {
    if (tensor_is_null(t)) return t;
    t->label = str_from_cstr(label);

    return t;
}

CTENSOR_API void tensor_print(tensor t) {
    if (tensor_is_null(t)) {
        printf("null\n");
        return;
    }

    str dims     = NULL;
        
    _tensor_prepare_dimStr(t);
    // _tensor_prepare_dataStr(t, data_string, sizeof(data_string));
    if (t->label)
        printf("tensor(name: \"%s\", shape: %s, data: %s)\n", str_cstr(t->label), str_cstr(t->_dimBuff), NULL);
    else
        printf("tensor(shape: %s, data: %s)\n", str_cstr(t->_dimBuff), NULL);
}

CTENSOR_API bool tensor_shape_match(tensor a, tensor b) {
    if (!a || !b) 
        return false;

    size_t size_a = vector_size(a->shape);
    size_t size_b = vector_size(b->shape);

    if (size_a != size_b)
        return false;

    for (int i = 0; i < size_a; ++i) {
        if(*(int*)vector_get(a->shape, i) != *(int*)vector_get(b->shape, i))
            return false;
    }

    return true;
}

CTENSOR_API tensor tensor_add(tensor a, tensor b) {
    if (!tensor_shape_match(a, b)) {
        printf("tensor shape mismatch: expected %s, got %s\n", str_cstr(a->_dimBuff), str_cstr(b->_dimBuff));
        return NULL;
    }

    tensor r = tensor_like(a);
    for (int i = 0; i < r->entries; ++i)
        r->data[i] = a->data[i] + b->data[i];

    return r;
}

CTENSOR_API tensor tensor_sub(tensor a, tensor b) {
    if (!tensor_shape_match(a, b))
        return NULL;

    tensor r = tensor_like(a);
    for (int i = 0; i < r->entries; ++i)
        r->data[i] = a->data[i] - b->data[i];

    return r;
}

static void _tensor_prepare_dimStr(tensor t) {
    size_t shape_dims = vector_size(t->shape);

    str_clear(t->_dimBuff);
    str_append_char(t->_dimBuff, '[');

    char nbuff[64] = { 0 };
    
    for (int i = 0; i < shape_dims; ++i) {
        _itoa_s(*(int*)vector_get(t->shape, i), nbuff, sizeof(nbuff), 10);
        size_t digits = strlen(nbuff);

        str_append_string(t->_dimBuff, nbuff, digits);
        
        if (i < shape_dims - 1) {
            str_append_cstr(t->_dimBuff, ", ");
        }
    }

    str_append_char(t->_dimBuff, ']');
}


// static void _tensor_prepare_dataStr(tensor t, char* dest, size_t size) {
//     printf("_tensor_prepare_dataStr(): Not Implemented\n");
// }

#endif // ifdef TENSOR_IMPL