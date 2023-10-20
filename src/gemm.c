#include "gemm.h"
#include "utils.h"
#include "im2col.h"
#include "dark_cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdint.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_MSC_VER)
#if defined(_M_ARM) || defined(_M_ARM64)
static inline uint32_t popcnt(uint32_t v) {
  v = v - ((v >> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
  return ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
}
#define POPCNT(x) popcnt((x))
#define POPCNT64(x) (popcnt((unsigned)(x)) + popcnt((unsigned)((uint64_t)(x) >> 32)))
#else
#include <intrin.h>
#ifdef _WIN64
#define POPCNT(x) __popcnt(x)
#define POPCNT64(x) __popcnt64(x)
#else
static inline int popcnt_64(uint64_t val64) {
  int tmp_count = __popcnt(val64);
  tmp_count += __popcnt(val64 >> 32);
  return tmp_count;
}
#define POPCNT(x) __popcnt(x)
#define POPCNT64(x) popcnt_64(x)
#endif
#endif
#elif defined(__GNUC__)
#define POPCNT(x) __builtin_popcount(x)
#define POPCNT64(x) __builtin_popcountll(x)
#endif

#define TILE_M 4 // 4 ops
#define TILE_N 16 // AVX2 = 2 ops * 8 floats
#define TILE_K 16 // loop
#ifdef __cplusplus
#define PUT_IN_REGISTER
#else
#define PUT_IN_REGISTER register
#endif

void gemm_bin(int M, int N, int K, float ALPHA,
        char  *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float* m = (float*)xcalloc(rows * cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

int is_avx() {
    return 0;
}

int is_fma_avx2() {
    return 0;
}

void gemm_nn(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}

void activate_array_cpu_custom(float *x, const int n, const ACTIVATION a)
{
    int i;
    if (a == LINEAR)
    {
    }
    else if (a == LEAKY)
    {
        for (i = 0; i < n; ++i) {
            x[i] = (x[i]>0) ? x[i] : .1*x[i];
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}

void float_to_bit(float *src, unsigned char *dst, size_t size)
{
    size_t dst_size = size / 8 + 1;
    memset(dst, 0, dst_size);

    size_t i;
    char* byte_arr = (char*)xcalloc(size, sizeof(char));
    for (i = 0; i < size; ++i) {
        if (src[i] > 0) byte_arr[i] = 1;
    }

    //for (i = 0; i < size; ++i) {
    //    dst[i / 8] |= byte_arr[i] << (i % 8);
    //}

    for (i = 0; i < size; i += 8) {
        char dst_tmp = 0;
        dst_tmp |= byte_arr[i + 0] << 0;
        dst_tmp |= byte_arr[i + 1] << 1;
        dst_tmp |= byte_arr[i + 2] << 2;
        dst_tmp |= byte_arr[i + 3] << 3;
        dst_tmp |= byte_arr[i + 4] << 4;
        dst_tmp |= byte_arr[i + 5] << 5;
        dst_tmp |= byte_arr[i + 6] << 6;
        dst_tmp |= byte_arr[i + 7] << 7;
        dst[i / 8] = dst_tmp;
    }
    free(byte_arr);
}

static inline void transpose_scalar_block(float *A, float *B, const int lda, const int ldb, const int block_size)
{
    int i;
    //#pragma omp parallel for
    for (i = 0; i<block_size; i++) {
        int j;
        for (j = 0; j<block_size; j++) {
            B[j*ldb + i] = A[i*lda + j];
        }
    }
}

void transpose_block_SSE4x4(float *A, float *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i += block_size) {
        int j, i2, j2;
        for (j = 0; j < m; j += block_size) {
            int max_i2 = i + block_size < n ? i + block_size : n;
            int max_j2 = j + block_size < m ? j + block_size : m;
            for (i2 = i; i2 < max_i2; ++i2) {
                for (j2 = j; j2 < max_j2; ++j2) {
                    B[j2*ldb + i2] = A[i2*lda + j2];
                }
                }
            }
        }
}

void forward_maxpool_layer_avx(float *src, float *dst, int *indexes, int size, int w, int h, int out_w, int out_h, int c,
    int pad, int stride, int batch)
{
    int b, k;
    const int w_offset = -pad / 2;
    const int h_offset = -pad / 2;

    for (b = 0; b < batch; ++b) {
        #pragma omp parallel for
        for (k = 0; k < c; ++k) {
            int i, j, m, n;
            for (i = 0; i < out_h; ++i) {
                for (j = 0; j < out_w; ++j) {
                    int out_index = j + out_w*(i + out_h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for (n = 0; n < size; ++n) {
                        for (m = 0; m < size; ++m) {
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int index = cur_w + w*(cur_h + h*(k + b*c));
                            int valid = (cur_h >= 0 && cur_h < h &&
                                cur_w >= 0 && cur_w < w);
                            float val = (valid != 0) ? src[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max = (val > max) ? val : max;
                        }
                    }
                    dst[out_index] = max;
                    if (indexes) indexes[out_index] = max_i;
                }
            }
        }
    }
}

// 32 channels -> 1 channel (with 32 floats)
// 256 channels -> 8 channels (with 32 floats)
void repack_input(float *input, float *re_packed_input, int w, int h, int c)
{
    const int items_per_channel = w * h;
    int chan, i;
    for (chan = 0; chan < c; chan += 32)
    {
        for (i = 0; i < items_per_channel; ++i)
        {
            int c_pack;
            for (c_pack = 0; c_pack < 32; ++c_pack) {
                float src = input[(chan + c_pack)*items_per_channel + i];

                re_packed_input[chan*items_per_channel + i * 32 + c_pack] = src;
            }
        }
    }
}

void transpose_uint32(uint32_t *src, uint32_t *dst, int src_h, int src_w, int src_align, int dst_align)
{
    //l.bit_align - algined (n) by 32
    //new_ldb - aligned (k) by 256

    int i;
    //#pragma omp parallel for
    for (i = 0; i < src_h; i += 1)  // l.size*l.size*l.c;
    {
        int j;
        for (j = 0; j < src_w; j += 1)  // out_h*out_w;
        {
            ((uint32_t *)dst)[j*dst_align / 32 + i] = ((uint32_t *)src)[i*src_align + j];
        }
    }
}

void gemm_nn_bin_transposed_32bit_packed(int M, int N, int K, float ALPHA,
    uint32_t *A, int lda,
    uint32_t *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < M; ++i) {   // l.n
        int j, s;
        float mean_val = mean_arr[i];
        for (j = 0; j < N; ++j) // out_h*out_w;
        {
            float val = 0;
            for (s = 0; s < K; ++s) // l.size*l.size*l.c/32  or (l.size*l.size*l.c)
            {
                PUT_IN_REGISTER uint32_t A_PART = ((uint32_t*)A)[i*lda + s];
                PUT_IN_REGISTER uint32_t B_PART = ((uint32_t*)B)[j * ldb + s];
                uint32_t xnor_result = ~(A_PART ^ B_PART);
                int32_t count = POPCNT(xnor_result);  // must be Signed int

                val += (2 * count - 32) * mean_val;
            }
            C[i*ldc + j] += val;
        }
    }
}

void convolution_repacked(uint32_t *packed_input, uint32_t *packed_weights, float *output,
    int w, int h, int c, int n, int size, int pad, int new_lda, float *mean_arr)
{
    int fil;
    // filter index
    #pragma omp parallel for
    for (fil = 0; fil < n; ++fil) {
        float mean_val = mean_arr[fil];
        int chan, y, x, f_y, f_x;   // c_pack
        // channel index
        for (chan = 0; chan < c / 32; ++chan)
            //for (chan = 0; chan < l.c; chan += 32)
            //for (c_pack = 0; c_pack < 32; ++c_pack)
            // input - y
            for (y = 0; y < h; ++y)
                // input - x
                for (x = 0; x < w; ++x)
                {
                    int const output_index = fil*w*h + y*w + x;
                    float sum = 0;

                    // filter - y
                    for (f_y = 0; f_y < size; ++f_y)
                    {
                        int input_y = y + f_y - pad;
                        // filter - x
                        for (f_x = 0; f_x < size; ++f_x)
                        {
                            int input_x = x + f_x - pad;
                            if (input_y < 0 || input_x < 0 || input_y >= h || input_x >= w) continue;

                            // normal
                            //float input = state.input[(chan + c_pack)*l.w*l.h + input_y*l.w + input_x];
                            //float weight = l.weights[fil*l.c*l.size*l.size + (chan + c_pack)*l.size*l.size + f_y*l.size + f_x];

                            // packed
                            //float input = re_packed_input[chan*l.w*l.h + (input_y*l.w + input_x) * 32 + c_pack];
                            //float weight = l.weights[fil*l.c*l.size*l.size + chan*l.size*l.size + (f_y*l.size + f_x) * 32 + c_pack];
                            //sum += input * weight;

                            //float input = re_packed_input[chan*l.w*l.h + (input_y*l.w + input_x) * 32 + c_pack];
                            //float weight = l.weights[fil*l.c*l.size*l.size + chan*l.size*l.size + (f_y*l.size + f_x) * 32 + c_pack];
                            //uint32_t bit1 = input > 0;
                            //uint32_t bit2 = weight > 0;
                            //uint32_t count = (~(bit1 ^ bit2)) & 1;
                            //float result = (2 * (float)count - 1) * mean_val;
                            //printf("\n mul = %f, bit1 = %d, bit2 = %d, count = %d, mean = %f, result = %f  ", input*weight, bit1, bit2, count, mean_val, result);
                            //sum += result;

                            uint32_t input = ((uint32_t *)packed_input)[chan*w*h + input_y*w + input_x];
                            //uint32_t weight = ((uint32_t *)l.align_bit_weights)[fil*l.c*l.size*l.size/32 + chan*l.size*l.size + f_y*l.size + f_x];
                            uint32_t weight = ((uint32_t *)packed_weights)[fil*new_lda / 32 + chan*size*size + f_y*size + f_x];

                            uint32_t xnor_result = ~(input ^ weight);
                            int32_t count = POPCNT(xnor_result); // mandatory Signed int
                            sum += (2 * count - 32) * mean_val;
                        }
                    }
                    // l.output[filters][width][height] +=
                    //        state.input[channels][width][height] *
                    //        l.weights[filters][channels][filter_width][filter_height];
                    output[output_index] += sum;
                }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            PUT_IN_REGISTER float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            PUT_IN_REGISTER float A_PART = ALPHA * A[k * lda + i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            PUT_IN_REGISTER float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    if (BETA != 1){
        assert(0);
    }
    int t;
    for (t = 0; t < M; ++t) {
        if (!TA && !TB)
            gemm_nn(1, N, K, ALPHA, A + t*lda, lda, B, ldb, C + t*ldc, ldc);
        else if (TA && !TB)
            gemm_tn(1, N, K, ALPHA, A + t, lda, B, ldb, C + t*ldc, ldc);
        else if (!TA && TB)
            gemm_nt(1, N, K, ALPHA, A + t*lda, lda, B, ldb, C + t*ldc, ldc);
        else
            gemm_tt(1, N, K, ALPHA, A + t, lda, B, ldb, C + t*ldc, ldc);
    }
}

#ifdef GPU

#endif

void init_cpu() {
    is_avx();
    is_fma_avx2();
}
