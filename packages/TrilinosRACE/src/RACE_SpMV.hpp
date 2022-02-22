#ifndef _RACE_SpMV_KERNEL_H
#define _RACE_SpMV_KERNEL_H

#include "RACE_CRS_raw.hpp"
#include <RACE/interface.h>

//naming convention is very important for macro generation of code
//i.e., oif using SpMV then use SpMV (and not SPMV) all over as seen in this template
namespace RACE {

    template <typename packtype>
    struct kernelArgSpMV
    {
        //single vector case only implemented, since Tpetra solvers use them and not multivector
        using array_type = typename packtype::marray_type;
        using CRS_raw_type = typename packtype::CRS_raw_type;
        using Scalar = typename packtype::SC;

        CRS_raw_type* A;
        array_type* x;
        Scalar alpha;
        Scalar beta;

        int arr_offset;
        int tunedPower;

   };

//convenience macros
#define RACE_ENCODE_TO_VOID_SpMV(A_en, x_en, alpha_en, beta_en, arr_offset_en)\
    using arg_type = kernelArgSpMV<packtype>;\
    arg_type *arg_encode = new arg_type;\
    arg_encode->A = A_en;\
    arg_encode->x = x_en;\
    arg_encode->alpha = alpha_en;\
    arg_encode->beta = beta_en;\
    arg_encode->arr_offset = arr_offset_en;\
    void* voidArg = (void*) arg_encode;\

#define RACE_SpMV_setTunedPower(tunedPower_en)\
    arg_encode->tunedPower = tunedPower_en;\

#define RACE_SpMV_setOffset(arr_offset_en)\
    arg_encode->arr_offset = arr_offset_en;\

#define RACE_DELETE_ARG_SpMV()\
    delete arg_encode;\

#define RACE_DECODE_FROM_VOID_SpMV(voidArg)\
    using arg_type = kernelArgSpMV<packtype>;\
    using CRS_raw_type = typename packtype::CRS_raw_type;\
    using array_type = typename packtype::marray_type;\
    using Scalar = typename packtype::SC;\
    using LocalOrdinal = typename packtype::LO;\
    arg_type* arg_decode = (arg_type*) voidArg;\
    CRS_raw_type* A = arg_decode->A;\
    array_type* x = arg_decode->x;\
    Scalar alpha = arg_decode->alpha;\
    Scalar beta = arg_decode->beta;\
    int arr_offset = arg_decode->arr_offset;\

#define BASE_SpMV_KERNEL_IN_ROW(_A_, _x_vec_)\
    Scalar tmp = 0;\
    _Pragma("nounroll")\
    _Pragma("omp simd simdlen(VECTOR_LENGTH) reduction(+:tmp)")\
    for(int idx=(int)_A_->rowPtr[row]; idx<(int)_A_->rowPtr[row+1]; ++idx)\
    {\
        tmp += _A_->val[idx]*((_x_vec_)[_A_->col[idx]]);\
    }\

    /// \brief Compute <tt>x[i+1] = beta*x[i] + alpha*x[i]</tt>, where
    template <typename packtype>
    inline void RACE_SpMV_KERNEL(int start, int end, int pow, int subPow, int numa_domain, void* args)
    {
    /*    using Scalar double;
        using LocalOrdinal int;
        using GlobalOrdinal long long;
        using Node */
        RACE_DECODE_FROM_VOID_SpMV(args);
        const int cur_offset = (pow-1)+arr_offset;
        const int next_offset = pow+arr_offset;

        //printf("tid = %d/%d, pow = %d, start = %d, end = %d\n", omp_get_thread_num(), omp_get_num_threads(), pow, start, end);
        if((alpha == 0) && (beta == 0))
        {
            for(LocalOrdinal row=start; row<end; ++row)
            {
                (*x)[next_offset][row] = 0;
            }
        }
        else if(alpha == 0)
        {
            for(LocalOrdinal row=start; row<end; ++row)
            {
                (*x)[next_offset][row] = beta*(*x)[next_offset][row];
            }
        }
        else if(beta == 0)
        {
            for(LocalOrdinal row=start; row<end; ++row)
            {
                BASE_SpMV_KERNEL_IN_ROW(A, (*x)[cur_offset]);
                (*x)[next_offset][row] = alpha*tmp;
            }
        }
        else
        {
            for(LocalOrdinal row=start; row<end; ++row)
            {
                BASE_SpMV_KERNEL_IN_ROW(A, (*x)[cur_offset]);
                (*x)[next_offset][row] = beta*(*x)[next_offset][row] + alpha*tmp;
            }
        }

    }


} // namespace RACE

#endif
