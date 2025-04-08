#include <stdio.h>

#define ARRAY_IMPL
#define TENSOR_IMPL
#define STR_IMPL

#define TEST_PRINT(msg) printf("-- %s --\n", (msg))

#define VECTOR_IMPLEMENTATION
#   include "vector.h"
#include "tensor.h"


void test_tensor() {
    TEST_PRINT("Testing Tensor");
    tensor t1  = tensor_create(SHAPE(32, 100, 20, 30, 12));
    
    tensor_entry_set(t1, TUPLE(0, 99, 0, 2, 0), 21.0f);

    float v = tensor_entry_get(t1, TUPLE(0, 99, 0, 2, 0));
    printf("%.3f\n", v);
}

int main() {
    test_tensor();
    return 0;
}