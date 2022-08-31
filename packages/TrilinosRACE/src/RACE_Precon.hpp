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
        using const_array_type = typename packtype::const_marray_type;

        bool initVecIsZero;
        double innerDamping;
        array_type* preconTmp;
        const_array_type* b;
        std::string preconType;
        bool fwdDir;
    };

    //convenience macros
#define RACE_ENCODE_TO_VOID_Precon(A_en, L_en, U_en, b_en, x_en, initVecIsZero_en, innerDamping_en, arr_offset_en, totSubPower_en, preconType_en, fwdDir_en, preconTmp_en)\
    using arg_type = kernelArgPrecon<packtype>;\
    arg_type *arg_encode = new arg_type;\
    arg_encode->A = A_en;\
    arg_encode->L = L_en;\
    arg_encode->U = U_en;\
    arg_encode->x = x_en;\
    arg_encode->b = b_en;\
    arg_encode->arr_offset = arr_offset_en;\
    arg_encode->totSubPower = totSubPower_en;\
    arg_encode->preconType = preconType_en;\
    arg_encode->fwdDir = fwdDir_en;\
    arg_encode->preconTmp = preconTmp_en;\
    arg_encode->innerDamping = innerDamping_en;\
    arg_encode->initVecIsZero = initVecIsZero_en;\
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
    using const_array_type = typename packtype::const_marray_type;\
    using Scalar = typename packtype::SC;\
    using LocalOrdinal = typename packtype::LO;\
    using complex_type = typename packtype::complex_type;\
    arg_type* arg_decode = (arg_type*) voidArg;\
    CRS_raw_type* A = arg_decode->A;\
    CRS_raw_type* L = arg_decode->L;\
    CRS_raw_type* U = arg_decode->U;\
    array_type* x = arg_decode->x;\
    const_array_type* b = arg_decode->b;\
    int arr_offset = arg_decode->arr_offset;\
    int totSubPower = arg_decode->totSubPower;\
    array_type* preconTmp = arg_decode->preconTmp;\
    std::string preconType = arg_decode->preconType;\
    bool fwdDir = arg_decode->fwdDir;\
    double innerDamping = arg_decode->innerDamping;\
    bool initVecIsZero = arg_decode->initVecIsZero;\


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
        Scalar value = ((col_idx >= start) && (col_idx < row))?(_A_->val[idx]*((_xInit_vec_)[col_idx])):0;\
        tmp = tmp+value;\
    }\
    tmp = _xInit_vec_[row] + (_A_->invDiag[row]) * (_b_vec_[row] - tmp);\

//this kernel is a hybrid Jacobi Richardson + GS.
//It does Jacobi Richardson to solve triangular solver
//It solves Mg=r, normally M is L or U
//g(k+1)=D^{-1}(r-Mg(k)), compact form for two-step GS
//or g(k+1)=g0-D^{-1}Mg_k
/*
#define BASE_PRECON_JACOBI_RICHARDSON_KERNEL_IN_ROW(_M_, _g0_vec_, _g_vec_)\
    Scalar tmp = 0;\
    _Pragma("nounroll")\
    _Pragma("omp simd simdlen(VECTOR_LENGTH) reduction(+:tmp)")\
    for(int idx=(int)_M_->rowPtr[row]; idx<(int)_M_->rowPtr[row+1]; ++idx)\
    {\
        int col_idx = _M_->col[idx];\
        tmp += (_M_->val[idx])*((_g_vec_)[col_idx]);\
    }\
    tmp = _g0_vec_[row] - (_M_->invDiag[row]) * (tmp);\
*/

#define BASE_PRECON_MPK_lower(_scale_, ...)\
    for(LocalOrdinal row=start; row<end; ++row)\
    {\
        BASE_SpMV_KERNEL_IN_ROW(L, (*mpkInArray)[mpk_cur_offset]);\
        (*mpkOutArray)[mpk_next_offset][row] = _scale_*tmp  __VA_ARGS__;\
    }\

#define BASE_PRECON_MPK_upper(_scale_, ...)\
    for(LocalOrdinal row=start; row<end; ++row)\
    {\
        BASE_SpMV_KERNEL_IN_ROW(U, (*mpkInArray)[mpk_cur_offset]);\
        (*mpkOutArray)[mpk_next_offset][row] = _scale_*tmp  __VA_ARGS__;\
    }\


#define BASE_PRECON_MPK_splitUL(_scale_, ...)\
    for(LocalOrdinal row=start; row<end; ++row)\
    {\
        BASE_SpMV_KERNEL_IN_ROW(U, (*mpkInArray)[mpk_cur_offset]);\
        (*mpkOutArray)[mpk_next_offset][row] = tmp + /*U->diag[row]*/1*(*mpkInArray)[mpk_cur_offset][row];\
    }\
    for(LocalOrdinal row=start; row<end; ++row)\
    {\
        BASE_SpMV_KERNEL_IN_ROW(L, (*mpkInArray)[mpk_cur_offset]);\
        (*mpkOutArray)[mpk_next_offset][row] = _scale_*((*mpkOutArray)[mpk_next_offset][row]+tmp)  __VA_ARGS__;\
    }\



#define BASE_PRECON_MPK_splitLU(_scale_, ...)\
    for(LocalOrdinal row=start; row<end; ++row)\
    {\
        BASE_SpMV_KERNEL_IN_ROW(L, (*mpkInArray)[mpk_cur_offset]);\
        (*mpkOutArray)[mpk_next_offset][row] = tmp + /*L->diag[row]*/1*(*mpkInArray)[mpk_cur_offset][row];\
    }\
    for(LocalOrdinal row=start; row<end; ++row)\
    {\
        BASE_SpMV_KERNEL_IN_ROW(U, (*mpkInArray)[mpk_cur_offset]);\
        (*mpkOutArray)[mpk_next_offset][row] = _scale_*((*mpkOutArray)[mpk_next_offset][row]+tmp)  __VA_ARGS__;\
    }\

#define BASE_PRECON_MPK(_scale_, ...)\
    for(LocalOrdinal row=start; row<end; ++row)\
    {\
        BASE_SpMV_KERNEL_IN_ROW(A, (*mpkInArray)[mpk_cur_offset]);\
        (*mpkOutArray)[mpk_next_offset][row] = _scale_*tmp __VA_ARGS__;\
    }\



    //the _VA_ARGS_ are added for the update kernel
#define BASE_Precon_PRECON(_precon_, ...)\
    if(preconType == "TWO-STEP-GAUSS-SEIDEL")\
    {\
        if(fwdDir == true)\
        {\
            BASE_Precon_TWO_STEP_GAUSS_SEIDEL_FWD(__VA_ARGS__);\
        }\
        else\
        {\
            BASE_Precon_TWO_STEP_GAUSS_SEIDEL_BWD(__VA_ARGS__);\
        }\
    }\
    else\
    {\
        for(LocalOrdinal row=start; row<end; ++row)\
        {\
            BASE_PRECON_ ## _precon_ ## _KERNEL_IN_ROW(A, (*preconInArray)[precon_cur_offset], (*preconOutArray)[precon_next_offset]);\
            (*preconOutArray)[precon_next_offset][row] = tmp __VA_ARGS__;\
        }\
    }

//for TWO-STEP-GAUSS-SEIDEL you can't write row-wise formulation
#define BASE_Precon_TWO_STEP_GAUSS_SEIDEL_FWD()\
    array_type* mpkInArray;\
    array_type* mpkOutArray;\
    int mpk_cur_offset;\
    int mpk_next_offset;\
    double gamma = innerDamping;\
    if(subPow == startSubPow) /*right-prec*/\
    {\
        if(initVecIsZero) \
        {\
            /*g=D^{-1}b*/\
            /*find residual*/\
            /*r=b*/\
            mpkInArray = preconTmp;\
            mpk_cur_offset = subPow-1;\
            /* g_0 = gamma*D^{-1}(b) */\
            if(startSubPow == endSubPow)\
            {\
                mpkOutArray = preconOutArray;\
                mpk_next_offset = precon_next_offset;\
            }\
            else\
            {\
                mpkOutArray = preconTmp;\
                mpk_next_offset = 1;\
            }\
            for(LocalOrdinal row=start; row<end; ++row)\
            {\
                (*mpkOutArray)[mpk_next_offset][row] = (gamma*U->invDiag[row]*(*preconInArray)[precon_cur_offset][row]);\
            }\
        }\
        else\
        {\
            /*find residual*/\
            /*This branch is currently useless, because initVec is always 0*/\
            mpkInArray = preconTmp;\
            mpk_cur_offset = subPow-1;\
            /* x_k+1 = x_k + gamma*D^{-1}(b-Ax_k) */\
            if(startSubPow == endSubPow)\
            {\
                mpkOutArray = preconOutArray;/*happens if inner iter=0*/\
                mpk_next_offset = precon_next_offset;\
                /*TODO: Storing (*mpkInArray)[0][row] xInit will not work in case of
                 * LEFT and in case of right tmp has to be initialized*/\
                BASE_PRECON_MPK_splitUL(-gamma, + (gamma*U->invDiag[row]*(*preconInArray)[precon_cur_offset][row]) + (*mpkInArray)[mpk_cur_offset][row]);\
            }\
            /* g_0 = gamma*D^{-1}(b-Ax_k) */\
            else\
            {\
                mpkOutArray = preconTmp;\
                mpk_next_offset = 1;\
                BASE_PRECON_MPK_splitUL(-gamma, + (gamma*U->invDiag[row]*(*preconInArray)[precon_cur_offset][row]));\
            }\
        }\
    }\
    else if(subPow <= endSubPow)\
    {\
        if(initVecIsZero) /*Remember incase initVec is 0, then one less subPow is only needed, but this is not exploited now*/\
        {\
            double gamma2 = 1-gamma;\
            /*This can be done on two arrays too, by swapping but we avoid it*/\
            mpkInArray = preconTmp;\
            mpkOutArray = preconTmp;\
            mpk_cur_offset = subPow-1;\
            mpk_next_offset = subPow;\
            /*Jacobi-Richardson iteration*/\
            /*g_k+1 = g_0 - D^{-1}Lg_k*/\
            if(subPow == (endSubPow))\
            {\
                mpkOutArray = preconOutArray;\
                mpk_next_offset = precon_next_offset;\
                BASE_PRECON_MPK_lower(-gamma, + (*mpkInArray)[1][row] + gamma2*(*mpkInArray)[mpk_cur_offset][row]);\
            }\
            else\
            {\
                BASE_PRECON_MPK_lower(-gamma, + (*mpkInArray)[1][row] + gamma2*(*mpkInArray)[mpk_cur_offset][row]);\
            }\
        }\
        else\
        {\
            double gamma2 = 1-gamma;\
            /*This can be done on two arrays too, by swapping but we avoid it*/\
            mpkInArray = preconTmp;\
            mpkOutArray = preconTmp;\
            mpk_cur_offset = subPow-1;\
            mpk_next_offset = subPow;\
            /*Jacobi-Richardson iteration*/\
            /*g_k+1 = g_0 - D^{-1}Lg_k*/\
            if(subPow == (endSubPow))\
            {\
                mpkOutArray = preconOutArray;\
                mpk_next_offset = precon_next_offset;\
                BASE_PRECON_MPK_lower(-gamma, + (*mpkInArray)[1][row] + gamma2*(*mpkInArray)[mpk_cur_offset][row] + (*mpkInArray)[0][row]);/*do update too*/\
            }\
            else\
            {\
                BASE_PRECON_MPK_lower(-gamma, + (*mpkInArray)[1][row] + gamma2*(*mpkInArray)[mpk_cur_offset][row]);\
            }\
        }\
    }\


//for TWO-STEP-GAUSS-SEIDEL you can't write row-wise formulation
#define BASE_Precon_TWO_STEP_GAUSS_SEIDEL_BWD()\
    array_type* mpkInArray;\
    array_type* mpkOutArray;\
    int mpk_cur_offset;\
    int mpk_next_offset;\
    double gamma = innerDamping;\
    if(subPow == startSubPow) /*right-prec*/\
    {\
        if(initVecIsZero) \
        {\
            /*g=D^{-1}b*/\
            /*find residual*/\
            /*r=b*/\
            mpkInArray = preconTmp;\
            mpk_cur_offset = subPow-1;\
            /* g_0 = gamma*D^{-1}(b) */\
            if(startSubPow == endSubPow)\
            {\
                mpkOutArray = preconOutArray;\
                mpk_next_offset = precon_next_offset;\
            }\
            else\
            {\
                mpkOutArray = preconTmp;\
                mpk_next_offset = 1;\
            }\
            for(LocalOrdinal row=start; row<end; ++row)\
            {\
                (*mpkOutArray)[mpk_next_offset][row] = (gamma*U->invDiag[row]*(*preconInArray)[precon_cur_offset][row]);\
            }\
        }\
        else\
        {\
            /*find residual*/\
            /*This branch is currently useless, because initVec is always 0*/\
            mpkInArray = preconTmp;\
            mpk_cur_offset = subPow-1;\
            /* x_k+1 = x_k + gamma*D^{-1}(b-Ax_k) */\
            if(startSubPow == endSubPow)\
            {\
                mpkOutArray = preconOutArray;/*happens if inner iter=0*/\
                mpk_next_offset = precon_next_offset;\
                /*TODO: Storing (*mpkInArray)[0][row] xInit will not work in case of
                 * LEFT and in case of right tmp has to be initialized*/\
                BASE_PRECON_MPK_splitLU(-gamma, + (gamma*U->invDiag[row]*(*preconInArray)[precon_cur_offset][row]) + (*mpkInArray)[mpk_cur_offset][row]);\
            }\
            /* g_0 = gamma*D^{-1}(b-Ax_k) */\
            else\
            {\
                mpkOutArray = preconTmp;\
                mpk_next_offset = 1;\
                BASE_PRECON_MPK_splitLU(-gamma, + (gamma*U->invDiag[row]*(*preconInArray)[precon_cur_offset][row]));\
            }\
        }\
    }\
    else if(subPow <= endSubPow)\
    {\
        if(initVecIsZero) /*Remember incase initVec is 0, then one less subPow is only needed*/\
        {\
            double gamma2 = 1-gamma;\
            /*This can be done on two arrays too, by swapping but we avoid it*/\
            mpkInArray = preconTmp;\
            mpkOutArray = preconTmp;\
            mpk_cur_offset = subPow-1;\
            mpk_next_offset = subPow;\
            /*Jacobi-Richardson iteration*/\
            /*g_k+1 = g_0 - D^{-1}Lg_k*/\
            if(subPow == (endSubPow))\
            {\
                mpkOutArray = preconOutArray;\
                mpk_next_offset = precon_next_offset;\
                BASE_PRECON_MPK_upper(-gamma, + (*mpkInArray)[1][row] + gamma2*(*mpkInArray)[mpk_cur_offset][row]);\
            }\
            else\
            {\
                BASE_PRECON_MPK_upper(-gamma, + (*mpkInArray)[1][row] + gamma2*(*mpkInArray)[mpk_cur_offset][row]);\
            }\
        }\
        else\
        {\
            double gamma2 = 1-gamma;\
            /*This can be done on two arrays too, by swapping but we avoid it*/\
            mpkInArray = preconTmp;\
            mpkOutArray = preconTmp;\
            mpk_cur_offset = subPow-1;\
            mpk_next_offset = subPow;\
            /*Jacobi-Richardson iteration*/\
            /*g_k+1 = g_0 - D^{-1}Lg_k*/\
            if(subPow == (endSubPow))\
            {\
                mpkOutArray = preconOutArray;\
                mpk_next_offset = precon_next_offset;\
                BASE_PRECON_MPK_upper(-gamma, + (*mpkInArray)[1][row] + gamma2*(*mpkInArray)[mpk_cur_offset][row] + (*mpkInArray)[0][row]);/*do update too*/\
            }\
            else\
            {\
                BASE_PRECON_MPK_upper(-gamma, + (*mpkInArray)[1][row] + gamma2*(*mpkInArray)[mpk_cur_offset][row]);\
            }\
        }\
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
            const_array_type* preconInArray = b;\
            int precon_cur_offset = cur_offset;\
            array_type* preconOutArray = x;\
            int precon_next_offset = cur_offset;\
            int startSubPow = 1;\
            int endSubPow = totSubPower;\
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
            else if((preconType == "TWO-STEP-GAUSS-SEIDEL") || (preconType == "NONE"))
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
