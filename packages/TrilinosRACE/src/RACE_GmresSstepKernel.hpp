#ifndef _RACE_GmresSstepKernel_KERNEL_H
#define _RACE_GmresSstepKernel_KERNEL_H

#include "RACE_CRS_raw.hpp"
#include <RACE/interface.h>
#include "TrilinosRACE_config.h"

namespace RACE {

    template <typename packtype>
    struct kernelArgGmresSstep : public kernelArgSpMV<packtype>
    {
        using complex_type = typename packtype::complex_type;
        std::vector<complex_type> theta;
    };

//convenience macros
#define RACE_ENCODE_TO_VOID_GmresSstepKernel(A_en, x_en, theta_en, arr_offset_en)\
    using arg_type = kernelArgGmresSstep<packtype>;\
    arg_type *arg_encode = new arg_type;\
    arg_encode->A = A_en;\
    arg_encode->x = x_en;\
    arg_encode->theta = theta_en;\
    arg_encode->arr_offset = arr_offset_en;\
    void* voidArg = (void*) arg_encode;\

#define RACE_GmresSstepKernel_setTunedPower(tunedPower_en)\
    arg_encode->tunedPower = tunedPower_en;\

#define RACE_GmresSstepKernel_setOffset(arr_offset_en)\
    arg_encode->arr_offset = arr_offset_en;\

#define RACE_DELETE_ARG_GmresSstepKernel()\
    delete arg_encode;\

#define RACE_DECODE_FROM_VOID_GmresSstepKernel(voidArg)\
    using arg_type = kernelArgGmresSstep<packtype>;\
    using CRS_raw_type = typename packtype::CRS_raw_type;\
    using array_type = typename packtype::marray_type;\
    using Scalar = typename packtype::SC;\
    using LocalOrdinal = typename packtype::LO;\
    using complex_type = typename packtype::complex_type;\
    arg_type* arg_decode = (arg_type*) voidArg;\
    CRS_raw_type* A = arg_decode->A;\
    array_type* x = arg_decode->x;\
    std::vector<complex_type> theta = arg_decode->theta;\
    int arr_offset = arg_decode->arr_offset;\

#define BASE_GmresSstepKernel_KERNEL_IN_ROW\
    Scalar tmp = 0;\
    _Pragma("nounroll")\
    _Pragma("omp simd simdlen(VECTOR_LENGTH) reduction(+:tmp)")\
    for(int idx=(int)A->rowPtr[row]; idx<(int)A->rowPtr[row+1]; ++idx)\
    {\
        tmp += A->val[idx]*((*x)[cur_offset][A->col[idx]]);\
    }\

    /// \brief Compute <tt>x[i+1] = theta_i*x[i] + theta_r*x[i]</tt>, where
    template <typename packtype>
    inline void RACE_GmresSstepKernel_KERNEL(std::true_type, int start, int end, int pow, int numa_domain, void* args)
    {
        RACE_DECODE_FROM_VOID_GmresSstepKernel(args);
        const int cur_offset = (pow-1)+arr_offset;
        const int next_offset = pow+arr_offset;
        const int prev_offset = (pow-2)+arr_offset;

        if(theta[cur_offset] == 0)
        {
            for(LocalOrdinal row=start; row<end; ++row)
            {
                BASE_GmresSstepKernel_KERNEL_IN_ROW;
                (*x)[next_offset][row] = tmp;
            }
        }
        else
        {
            for(LocalOrdinal row=start; row<end; ++row)
            {
                BASE_GmresSstepKernel_KERNEL_IN_ROW;
                (*x)[next_offset][row] = tmp - theta[cur_offset]*(*x)[cur_offset][row];
            }
        }

    }

    /// \brief Compute <tt>x[i+1] = theta_i*x[i] + theta_r*x[i]</tt>, where
    template <typename packtype>
    inline void RACE_GmresSstepKernel_KERNEL(std::false_type, int start, int end, int pow, int numa_domain, void* args)
    {
        RACE_DECODE_FROM_VOID_GmresSstepKernel(args);
        const int cur_offset = (pow-1)+arr_offset;
        const int next_offset = pow+arr_offset;
        const int prev_offset = (pow-2)+arr_offset;
        Scalar theta_r = theta[cur_offset].real();
        Scalar theta_i = theta[cur_offset].imag();
        Scalar theta_i_sq = theta_i*theta_i;

        if(theta_i == 0)
        {
            for(LocalOrdinal row=start; row<end; ++row)
            {
                BASE_GmresSstepKernel_KERNEL_IN_ROW;
                (*x)[next_offset][row] = tmp - theta_r*(*x)[cur_offset][row];
            }
        }
        else
        {
            for(LocalOrdinal row=start; row<end; ++row)
            {
                BASE_GmresSstepKernel_KERNEL_IN_ROW;
                (*x)[next_offset][row] = tmp - theta_r*(*x)[cur_offset][row] + theta_i_sq*(*x)[prev_offset][row];
            }
        }

    }


    template<typename packtype>
        inline void RACE_GmresSstepKernel_KERNEL(int start, int end, int pow, int numa_domain, void* args)
        {
            RACE_GmresSstepKernel_KERNEL<packtype>(std::integral_constant<bool, packtype::STS::isComplex>{}, start, end, pow, numa_domain, args);
        }

} // namespace RACE

#endif
