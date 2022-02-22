#ifndef _RACE_Precon_KERNEL_H
#define _RACE_Precon_KERNEL_H

#include "RACE_CRS_raw.hpp"
#include <RACE/interface.h>
#include "TrilinosRACE_config.h"
#include "RACE_SpMV.hpp"
#include "RACE_Precon.hpp"

namespace RACE {

    template <typename packtype>
    struct kernelArgPrecon : public kernelArgSpMV<packtype>
    {
        using complex_type = typename packtype::complex_type;
        using array_type = typename packtype::marray_type;

        array_type* b;
        std::string preconType;
    };


/////////////////////// BASE PRECONDITIONER KERNELS /////////////////////

//none- no preconditioning
#define BASE_PRECON_NONE_KERNEL_IN_ROW(_A_, _b_vec_, _xInit_vec_)\
    Scalar tmp=_xInit_vec_[row];

//plain Jacobi kernel
#define BASE_PRECON_JACOBI_KERNEL_IN_ROW(_A_, _x_in_vec_, _x_out_vec_)\
    Scalar tmp = 0;\
    tmp = (_A_->invDiag[row])* (_x_in_vec_)[row];\

//For Jacobi the preconditioner and SpMV are fused together
//therefore we have to distinguish right and left precon here
#define BASE_PRECON_JACOBI_w_SpMV_RIGHT_KERNEL_IN_ROW(_A_, _x_in_vec_, _x_out_vec_)\
    Scalar tmp = 0;\
    _Pragma("nounroll")\
    _Pragma("omp simd simdlen(VECTOR_LENGTH) reduction(+:tmp)")\
    for(int idx=(int)_A_->rowPtr[row]; idx<(int)_A_->rowPtr[row+1]; ++idx)\
    {\
        int col_idx = _A_->col[idx];\
        tmp += (_A_->val[idx])*(_A_->invDiag[col_idx])*((_x_in_vec_)[col_idx]);\
    }\


//For Jacobi the preconditioner and SpMV are fused together
//therefore we have to distinguish right and left precon here
#define BASE_PRECON_JACOBI_w_SpMV_LEFT_KERNEL_IN_ROW(_A_, _x_in_vec_, _x_out_vec_)\
    Scalar tmp = 0;\
    _Pragma("nounroll")\
    _Pragma("omp simd simdlen(VECTOR_LENGTH) reduction(+:tmp)")\
    for(int idx=(int)_A_->rowPtr[row]; idx<(int)_A_->rowPtr[row+1]; ++idx)\
    {\
        int col_idx = _A_->col[idx];\
        tmp += (_A_->val[idx])*((_x_in_vec_)[col_idx]);\
    }\
    tmp = _A_->invDiag[row]*tmp;\


//plain Gauss-Seidel kernel (compatible only for serial code)
#define BASE_PRECON_GAUSS_SEIDEL_KERNEL_IN_ROW(_A_, _b_vec_, _xInit_vec_)\
    BASE_SpMV_KERNEL_IN_ROW(_A_, _xInit_vec_);\
    tmp = _xInit_vec_[row] + (_A_->invDiag[row]) * (_b_vec_[row] - tmp);\



//this kernel is a hybrid Jacobi+GS. It does Jacobi within a level where
//conflicts can happen and GS for the rest. The kernel supports multi-threading
#define BASE_PRECON_JACOBI_GAUSS_SEIDEL_KERNEL_IN_ROW(_A_, _b_vec_, _xInit_vec_)\
    Scalar tmp = 0;\
    _Pragma("nounroll")\
    _Pragma("omp simd simdlen(VECTOR_LENGTH) reduction(+:tmp)")\
    for(int idx=(int)_A_->rowPtr[row]; idx<(int)_A_->rowPtr[row+1]; ++idx)\
    {\
        int col_idx = _A_->col[idx];\
        Scalar value = ((col_idx >= start) && (col_idx < end))?(_A_->val[idx]*((_xInit_vec_)[col_idx])):0;\
        tmp += value;\
    }\
    tmp = _xInit_vec_[row] + (_A_->invDiag[row]) * (_b_vec_[row] - tmp);\



//convenience macros
#define RACE_ENCODE_TO_VOID_Precon(A_en, b_en, x_en, arr_offset_en, preconType_en)\
    using arg_type = kernelArgPrecon<packtype>;\
    arg_type *arg_encode = new arg_type;\
    arg_encode->A = A_en;\
    arg_encode->x = x_en;\
    arg_encode->b = b_en;\
    arg_encode->arr_offset = arr_offset_en;\
    arg_encode->preconType = preconType_en;\
    void* voidArg = (void*) arg_encode;\

#define RACE_Precon_setTunedPower(tunedPower_en)\
    arg_encode->tunedPower = tunedPower_en;\

#define RACE_Precon_setOffset(arr_offset_en)\
    arg_encode->arr_offset = arr_offset_en;\

#define RACE_DELETE_ARG_Precon()\
    delete arg_encode;\

#define RACE_DECODE_FROM_VOID_Precon(voidArg)\
    using arg_type = kernelArgPrecon<packtype>;\
    using CRS_raw_type = typename packtype::CRS_raw_type;\
    using array_type = typename packtype::marray_type;\
    using Scalar = typename packtype::SC;\
    using LocalOrdinal = typename packtype::LO;\
    using complex_type = typename packtype::complex_type;\
    arg_type* arg_decode = (arg_type*) voidArg;\
    CRS_raw_type* A = arg_decode->A;\
    array_type* x = arg_decode->x;\
    array_type* b = arg_decode->b;\
    int arr_offset = arg_decode->arr_offset;\
    std::string preconType = arg_decode->preconType;\

    //the _VA_ARGS_ are added for the update kernel
#define BASE_Precon_PRECON(_precon_, ...)\
    for(LocalOrdinal row=start; row<end; ++row)\
    {\
        BASE_PRECON_ ## _precon_ ## _KERNEL_IN_ROW(A, (*preconInArray)[precon_cur_offset], (*preconOutArray)[precon_next_offset]);\
        (*preconOutArray)[precon_next_offset][row] = tmp __VA_ARGS__;\
    }\


#define GENERATE_KERNEL_Precon(_precon_)\
    template <typename packtype>\
    inline void RACE_Precon_ ## _precon_ ## _KERNEL(int start, int end, int pow, int subPow, int numa_domain, void* args)\
    {\
        RACE_DECODE_FROM_VOID_Precon(args);\
        const int cur_offset = (pow-1)+arr_offset;\
        if(preconType == "NONE")\
        {\
            /*Nothing to do*/\
        }\
        else\
        {\
            array_type* preconInArray = b;\
            int precon_cur_offset = cur_offset;\
            array_type* preconOutArray = x;\
            int precon_next_offset = cur_offset;\
            BASE_Precon_PRECON(_precon_);\
        }\
    }\


    //generate actual kernels
    GENERATE_KERNEL_Precon(JACOBI);
    GENERATE_KERNEL_Precon(JACOBI_GAUSS_SEIDEL);
    GENERATE_KERNEL_Precon(GAUSS_SEIDEL);
    GENERATE_KERNEL_Precon(NONE);

    //dispatcher function
    template<typename packtype>
        inline void RACE_Precon_KERNEL(int start, int end, int pow, int subPow, int numa_domain, void* args)
        {
            using arg_type = kernelArgPrecon<packtype>;
            arg_type* arg_decode = (arg_type*) args;
            std::string preconType = arg_decode->preconType;

            //RACE_Precon_KERNEL<packtype>(std::integral_constant<bool, packtype::STS::isComplex>{}, start, end, pow, subPow, numa_domain, args);
            if(preconType == "JACOBI")
            {
                RACE_Precon_JACOBI_KERNEL<packtype>(start, end, pow, subPow, numa_domain, args);
            }
            else if(preconType == "GAUSS-SEIDEL")
            {
                RACE_Precon_GAUSS_SEIDEL_KERNEL<packtype>(start, end, pow, subPow, numa_domain, args);
            }
            else if(preconType == "JACOBI-GAUSS-SEIDEL")
            {
                RACE_Precon_JACOBI_GAUSS_SEIDEL_KERNEL<packtype>(start, end, pow, subPow, numa_domain, args);
            }
            else if(preconType == "NONE")
            {
                RACE_Precon_NONE_KERNEL<packtype>(start, end, pow, subPow, numa_domain, args);
            }
            else
            {
                ERROR_PRINT("%s preconditioner not available\n", preconType.c_str());
            }
        }

} // namespace RACE

#endif
