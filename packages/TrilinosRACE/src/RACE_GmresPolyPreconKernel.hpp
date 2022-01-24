#ifndef _RACE_GmresPolyPreconKernel_KERNEL_H
#define _RACE_GmresPolyPreconKernel_KERNEL_H

#include "RACE_CRS_raw.hpp"
#include "RACE_SpMV.hpp"
#include <RACE/interface.h>
#include "TrilinosRACE_config.h"

namespace RACE {

    template <typename packtype>
    struct kernelArgGmresPolyPrecon : public kernelArgSpMV<packtype>
    {
        using complex_type = typename packtype::complex_type;
        using array_type = typename packtype::marray_type;
        array_type* y;
        std::vector<complex_type> theta;
        int maxSteps;
    };

//convenience macros
#define RACE_ENCODE_TO_VOID_GmresPolyPreconKernel(A_en, x_en, y_en, theta_en, arr_offset_en, maxSteps_en, tunedPower_en)\
    using arg_type = kernelArgGmresPolyPrecon<packtype>;\
    arg_type *arg_encode = new arg_type;\
    arg_encode->A = A_en;\
    arg_encode->x = x_en;\
    arg_encode->y = y_en;\
    arg_encode->theta = theta_en;\
    arg_encode->arr_offset = arr_offset_en;\
    arg_encode->maxSteps = maxSteps_en;\
    arg_encode->tunedPower = tunedPower_en;\
    void* voidArg = (void*) arg_encode;\

#define RACE_GmresPolyPreconKernel_setTunedPower(tunedPower_en)\
    arg_encode->tunedPower = tunedPower_en;\

#define RACE_GmresPolyPreconKernel_setOffset(arr_offset_en)\
    arg_encode->arr_offset = arr_offset_en;\

#define RACE_DELETE_ARG_GmresPolyPreconKernel()\
    delete arg_encode;\

#define RACE_DECODE_FROM_VOID_GmresPolyPreconKernel(voidArg)\
    using arg_type = kernelArgGmresPolyPrecon<packtype>;\
    using CRS_raw_type = typename packtype::CRS_raw_type;\
    using array_type = typename packtype::marray_type;\
    using Scalar = typename packtype::SC;\
    using LocalOrdinal = typename packtype::LO;\
    using complex_type = typename packtype::complex_type;\
    arg_type* arg_decode = (arg_type*) voidArg;\
    CRS_raw_type* A = arg_decode->A;\
    array_type* x = arg_decode->x;\
    array_type* y = arg_decode->y;\
    std::vector<complex_type> theta = arg_decode->theta;\
    int arr_offset = arg_decode->arr_offset;\
    int maxSteps = arg_decode->maxSteps;\
    int tunedPower = arg_decode->tunedPower;\

#define BASE_GmresPolyPreconKernel_KERNEL_IN_ROW\
    Scalar tmp = 0;\
    _Pragma("nounroll")\
    _Pragma("omp simd simdlen(VECTOR_LENGTH) reduction(+:tmp)")\
    for(int idx=(int)A->rowPtr[row]; idx<(int)A->rowPtr[row+1]; ++idx)\
    {\
        tmp += A->val[idx]*((*x)[cur_offset][A->col[idx]]);\
    }\

    //x stores prod
    template <typename packtype>
    inline void RACE_GmresPolyPreconKernel_KERNEL(int start, int end, int pow, int numa_domain, void* args)
    {
        RACE_DECODE_FROM_VOID_GmresPolyPreconKernel(args);
        //wrap around offset so we can reuse prod (x), and keep max. col size of
        //prod with maxSteps
        const int cur_offset = ((pow-1)+arr_offset)%(tunedPower+1);
        const int next_offset = (pow+arr_offset)%(tunedPower+1);
        const int cur_offset_wo_wrapping = (pow-1)+arr_offset;
        Scalar theta_r = theta[cur_offset_wo_wrapping].real();
        Scalar theta_r_inv = 1.0/theta_r;
        Scalar theta_i = theta[cur_offset_wo_wrapping].imag();

        if((theta_i == 0) || (packtype::STS::isComplex))
        {
            for(LocalOrdinal row=start; row<end; ++row)
            {
                (*y)[0][row] = (*y)[0][row] + theta_r_inv*(*x)[cur_offset][row];
                BASE_GmresPolyPreconKernel_KERNEL_IN_ROW;
                (*x)[next_offset][row] = (*x)[cur_offset][row] - theta_r_inv*tmp;
            }
        }
        else //if choosing this branch ensure total power is even
        {

            Scalar mod = theta_r*theta_r + theta_i*theta_i;
            Scalar mod_inv = 1/mod;
            const int prev_offset = ((pow-2)+arr_offset)%(tunedPower+1);

            if(pow % 2)
            {
                for(LocalOrdinal row=start; row<end; ++row)
                {
                    BASE_GmresPolyPreconKernel_KERNEL_IN_ROW;
                    (*x)[next_offset][row] = 2*theta_r*(*x)[cur_offset][row] - tmp;
                    (*y)[0][row] = (*y)[0][row] + mod_inv*(*x)[next_offset][row];
                }

            }
            else if((pow+arr_offset) != maxSteps)
            {
                for(LocalOrdinal row=start; row<end; ++row)
                {
                    BASE_GmresPolyPreconKernel_KERNEL_IN_ROW;
                    (*x)[next_offset][row] = (*x)[prev_offset][row] - mod_inv*tmp;
                }
            }
        }
    }

} // namespace RACE

#endif
