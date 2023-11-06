#include "fixed.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

void arr_float_to_fixed(float* fp_arr, fixed_t* fixed_arr, int N)
{
    int i = 0;
#if defined(__ARM_NEON)
    float32_t scale = (float32_t) (1 << QBITS);
    int vector_N = N - (N % 16);
    for (i = 0; i < vector_N; i += 16)
    {
        float32x4_t fp0 = vld1q_f32(fp_arr + i + 0*4);
        float32x4_t fp1 = vld1q_f32(fp_arr + i + 1*4);
        float32x4_t fp2 = vld1q_f32(fp_arr + i + 2*4);
        float32x4_t fp3 = vld1q_f32(fp_arr + i + 3*4);

        int32x4_t  fixed0 = vcvtq_s32_f32(vmulq_n_f32 (fp0, scale));
        int32x4_t  fixed1 = vcvtq_s32_f32(vmulq_n_f32 (fp1, scale));
        int32x4_t  fixed2 = vcvtq_s32_f32(vmulq_n_f32 (fp2, scale));
        int32x4_t  fixed3 = vcvtq_s32_f32(vmulq_n_f32 (fp3, scale));

        vst1q_s32(fixed_arr + i + 0*4, fixed0);
        vst1q_s32(fixed_arr + i + 1*4, fixed1);
        vst1q_s32(fixed_arr + i + 2*4, fixed2);
        vst1q_s32(fixed_arr + i + 3*4, fixed3);
    }
    for (i = vector_N; i < N; i++)
        fixed_arr[i] = float_to_fixed(fp_arr[i]);
#else
	for (i = 0; i < N; i++)
        fixed_arr[i] = float_to_fixed(fp_arr[i]);
#endif
}

void arr_fixed_to_float(fixed_t* fixed_arr, float* fp_arr, int N)
{
    int i = 0;
#if defined(__ARM_NEON)
    float32_t scale = 1.0 / ((float32_t) (1 << QBITS));
    int vector_N = N - (N % 16);
    for (i = 0; i < vector_N; i += 16)
    {
        int32x4_t fixed0 = vld1q_s32(fixed_arr + i + 0*4);
        int32x4_t fixed1 = vld1q_s32(fixed_arr + i + 1*4);
        int32x4_t fixed2 = vld1q_s32(fixed_arr + i + 2*4);
        int32x4_t fixed3 = vld1q_s32(fixed_arr + i + 3*4);

        float32x4_t fp0 = vmulq_n_f32 (vcvtq_f32_s32(fixed0), scale);
        float32x4_t fp1 = vmulq_n_f32 (vcvtq_f32_s32(fixed1), scale);
        float32x4_t fp2 = vmulq_n_f32 (vcvtq_f32_s32(fixed2), scale);
        float32x4_t fp3 = vmulq_n_f32 (vcvtq_f32_s32(fixed3), scale);

        vst1q_f32(fp_arr + i + 0*4, fp0);
        vst1q_f32(fp_arr + i + 1*4, fp1);
        vst1q_f32(fp_arr + i + 2*4, fp2);
        vst1q_f32(fp_arr + i + 3*4, fp3);
    }
    for (i = vector_N; i < N; i++)
        fp_arr[i] = fixed_to_float(fixed_arr[i]);
#else
    for (i = 0; i < N; i++)
        fp_arr[i] = fixed_to_float(fixed_arr[i]);
#endif
}
