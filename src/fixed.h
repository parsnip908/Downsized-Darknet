#ifndef FIXED_H
#define FIXED_H
#include <stdint.h>

#define QBITS 24
#define fixed_t int32_t

// Convert float to fixed_t
inline fixed_t float_to_fixed(float fp)
{
    return (fixed_t)(fp * (1 << QBITS));
}

// Shift by n qbits
// inline fixed_t qshift(fixed_t fixed_num, int n)
// {
//     return fixed_num >> (n * QBITS);
// }

// Use this any time you want to multiply two fixed_t
inline fixed_t fixed_mul(fixed_t a, fixed_t b)
{
    // return (fixed_t) (((int64_t)a * (int64_t)b) >> QBITS); //for arbitrary qbits
    // return (a * b) >> QBITS; //qbits <=12
    // return ((a>>8) * (b>>8)) >> 4; // for 20 qbits. inexact mult
    return (a>>(QBITS/2)) * (b>>(QBITS/2)); // for 24 qbits. inexact mult
}

// Convert fixed_t number back to float
inline float fixed_to_float(fixed_t fixed_num)
{
    return ((float)fixed_num) / (1 << QBITS);
}

void arr_float_to_fixed(float* fp_arr, fixed_t* fixed_arr, int N);

void arr_fixed_to_float(fixed_t* fixed_arr, float* fp_arr, int N);

#endif