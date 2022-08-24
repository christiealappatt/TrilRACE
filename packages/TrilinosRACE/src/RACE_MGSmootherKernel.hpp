#ifndef _RACE_MGSmootherKernel_KERNEL_H
#define _RACE_MGSmootherKernel_KERNEL_H

#include "RACE_CRS_raw.hpp"
#include "RACE_SpMV.hpp"
#include "RACE_Precon.hpp"
#include <RACE/interface.h>
#include "TrilinosRACE_config.h"

namespace RACE {

    template <typename packtype>
    struct kernelArgMGSmoother : public kernelArgPrecon<packtype>
    {
        using complex_type = typename packtype::complex_type;
        using array_type = typename packtype::marray_type;
        using const_array_type = typename packtype::const_marray_type;
        array_type* r;
        std::vector<double>* scaleCoeff;
        std::vector<double>* scale2Coeff;
        //array_type* preconTmp;

        std::string preconType;
        int maxSteps;
    };


//convenience macros
#define RACE_ENCODE_TO_VOID_MGSmootherKernel(A_en, L_en, U_en, x_en, b_en, r_en, initVecIsZero_en, innerDamping_en, scaleCoeff_en, scale2Coeff_en, arr_offset_en, maxSteps_en, tunedPower_en, totSubPower_en, preconType_en, fwdDir_en,  gmresPolyPreconTmp_en)\
    using arg_type = kernelArgMGSmoother<packtype>;\
    arg_type *arg_encode = new arg_type;\
    arg_encode->A = A_en;\
    arg_encode->L = L_en;\
    arg_encode->U = U_en;\
    arg_encode->x = x_en;\
    arg_encode->b = b_en;\
    arg_encode->r = r_en;\
    arg_encode->arr_offset = arr_offset_en;\
    arg_encode->maxSteps = maxSteps_en;\
    arg_encode->totSubPower = totSubPower_en;\
    arg_encode->tunedPower = tunedPower_en;\
    arg_encode->preconType = preconType_en;\
    arg_encode->fwdDir = fwdDir_en;\
    arg_encode->preconTmp = gmresPolyPreconTmp_en;\
    arg_encode->innerDamping = innerDamping_en;\
    arg_encode->scaleCoeff = scaleCoeff_en;\
    arg_encode->scale2Coeff = scale2Coeff_en;\
    arg_encode->initVecIsZero = initVecIsZero_en;\
    void* voidArg = (void*) arg_encode;\

#define RACE_MGSmootherKernel_setTunedPower(tunedPower_en)\
    arg_encode->tunedPower = tunedPower_en;\

#define RACE_MGSmootherKernel_setOffset(arr_offset_en)\
    arg_encode->arr_offset = arr_offset_en;\

#define RACE_DELETE_ARG_MGSmootherKernel()\
    delete arg_encode;\

#define RACE_DECODE_FROM_VOID_MGSmootherKernel(voidArg)\
    using arg_type = kernelArgMGSmoother<packtype>;\
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
    array_type* r = arg_decode->r;\
    int arr_offset = arg_decode->arr_offset;\
    int maxSteps = arg_decode->maxSteps;\
    int totSubPower = arg_decode->totSubPower;\
    int tunedPower = arg_decode->tunedPower;\
    std::string preconType = arg_decode->preconType;\
    bool fwdDir = arg_decode->fwdDir;\
    array_type* gmresPolyPreconTmp = arg_decode->preconTmp;\
    double innerDamping = arg_decode->innerDamping;\
    std::vector<double>* scaleCoeff = arg_decode->scaleCoeff;\
    std::vector<double>* scale2Coeff = arg_decode->scale2Coeff;\
    bool initVecIsZero = arg_decode->initVecIsZero;\

//define the same for fused residual kernels too
#define RACE_ENCODE_TO_VOID_MGSmootherKernel_w_Residual(A_en, L_en, U_en, x_en, b_en, r_en, initVecIsZero_en, innerDamping_en, scaleCoeff_en, scale2Coeff_en, arr_offset_en, maxSteps_en, tunedPower_en, totSubPower_en, preconType_en, fwdDir_en, gmresPolyPreconTmp_en)\
    RACE_ENCODE_TO_VOID_MGSmootherKernel(A_en, L_en, U_en, x_en, b_en, r_en, initVecIsZero_en, innerDamping_en, scaleCoeff, scale2Coeff, arr_offset_en, maxSteps_en, tunedPower_en, totSubPower_en, preconType_en,  fwdDir_en, gmresPolyPreconTmp_en)

#define RACE_MGSmootherKernel_w_Residual_setTunedPower(tunedPower_en)\
    RACE_MGSmootherKernel_setTunedPower(tunedPower_en)

#define RACE_MGSmootherKernel_w_Residual_setOffset(arr_offset_en)\
    RACE_MGSmootherKernel_setOffset(arr_offset_en)

#define RACE_DELETE_ARG_MGSmootherKernel_w_Residual()\
    RACE_DELETE_ARG_MGSmootherKernel()

#define RACE_DECODE_FROM_VOID_MGSmootherKernel_w_Residual(voidArg)\
    RACE_DECODE_FROM_VOID_MGSmootherKernel(voidArg)



#define BASE_Chebyshev_MPK()\
    if(next_pow == 1)\
    {\
        double scale = (*scaleCoeff)[cur_pow];\
        if(initVecIsZero)\
        {\
            for(LocalOrdinal row=start; row<end; ++row)\
            {\
                (*preconTmp)[2][row] = scale*A->invDiag[row]*((*preconInArray)[precon_cur_offset][row]);\
                (*preconOutArray)[precon_next_offset][row] = (*preconTmp)[2][row];\
            }\
        }\
        else\
        {\
            for(LocalOrdinal row=start; row<end; ++row)\
            {\
                BASE_SpMV_KERNEL_IN_ROW(A, (*preconTmp)[cur_pow_w_wrap]);\
                (*preconTmp)[2][row] = scale*A->invDiag[row]*((*preconInArray)[precon_cur_offset][row] - tmp);/*W is stored in (*preconTmp)[2]*/\
                (*preconOutArray)[precon_next_offset][row] = (*preconTmp)[cur_pow_w_wrap][row] + (*preconTmp)[2][row];\
            }\
        }\
    }\
    else\
    {\
        double scale = (*scaleCoeff)[cur_pow];\
        double scale2 = (*scale2Coeff)[cur_pow];\
        for(LocalOrdinal row=start; row<end; ++row)\
        {\
            BASE_SpMV_KERNEL_IN_ROW(A, (*preconTmp)[cur_pow_w_wrap]);\
            (*preconTmp)[2][row] = scale*A->invDiag[row]*((*preconInArray)[precon_cur_offset][row] - tmp) + scale2*(*preconTmp)[2][row];/*W is stored in (*preconTmp)[2]*/\
            (*preconOutArray)[precon_next_offset][row] = (*preconTmp)[cur_pow_w_wrap][row] + (*preconTmp)[2][row];\
        }\
    }\



//one power is one sweep
#define BASE_MGSmootherKernel_MPK(_Dir_, ...)\
    if(preconType == "TWO-STEP-GAUSS-SEIDEL") \
    {\
        const_array_type* preconInArray;\
        array_type* preconOutArray;\
        int precon_cur_offset=0;\
        int precon_next_offset=0;\
        int startSubPow = 1;\
        int endSubPow = totSubPower-1;\
        preconInArray = b;\
        precon_cur_offset = 0;\
        preconOutArray = gmresPolyPreconTmp;\
        precon_next_offset = totSubPower-1;\
        /*xInit should be in preconTmp[0]*/\
        array_type* preconTmp = gmresPolyPreconTmp;\
        BASE_Precon_TWO_STEP_GAUSS_SEIDEL_ ## _Dir_ ();\
        /*Copy cur x to xInit for next sweep*/\
        if(subPow == totSubPower)\
        {\
            for(LocalOrdinal row=start; row<end; ++row)\
            {\
                (*preconTmp)[0][row] = (*preconTmp)[totSubPower-1][row];\
            }\
        }\
    }\
    else if(preconType == "CHEBYSHEV") \
    {\
        const_array_type* preconInArray;\
        array_type* preconOutArray;\
        preconInArray = b;\
        int precon_cur_offset = 0;\
        preconOutArray = gmresPolyPreconTmp;\
        int precon_next_offset = next_pow_w_wrap;\
        /*xInit should be in preconTmp[0]*/\
        array_type* preconTmp = gmresPolyPreconTmp;\
        BASE_Chebyshev_MPK();\
    }\

//final sweep with residual
#define BASE_MGSmootherKernel_MPK_final_w_RESIDUAL(_Dir_, ...)\
    if(preconType == "TWO-STEP-GAUSS-SEIDEL") \
    {\
        const_array_type* preconInArray;\
        array_type* preconOutArray;\
        int precon_cur_offset=0;\
        int precon_next_offset=0;\
        int startSubPow = 1;\
        int endSubPow = totSubPower-1;\
        preconInArray = b;\
        precon_cur_offset = 0;\
        preconOutArray = x;\
        precon_next_offset = 0;\
        /*xInit should be in preconTmp[0]*/\
        array_type* preconTmp = gmresPolyPreconTmp;\
        BASE_Precon_TWO_STEP_GAUSS_SEIDEL_ ## _Dir_ ();\
        if(subPow == totSubPower) /* MPK kernel*/\
        {\
            array_type* mpkInArray = x;\
            int mpk_cur_offset = 0;\
            array_type* mpkOutArray = r;\
            int mpk_next_offset = 0;\
            BASE_PRECON_MPK_splitLU(-U->diag[row], +(*b)[0][row]);\
            /*BASE_PRECON_MPK(-1, +(*b)[0][row]);*/\
        }\
    }\
    else if(preconType == "CHEBYSHEV") \
    {\
        /*In this case find residual only, because no subPower in case of
         * Cheyshev, so sweep has to be adjusted accordingly when calling the
         * kernel*/\
        array_type* mpkInArray = x;\
        int mpk_cur_offset = 0;\
        array_type* mpkOutArray = r;\
        int mpk_next_offset = 0;\
        BASE_PRECON_MPK(-1, +(*b)[0][row]);\
    }\



//final sweep
#define BASE_MGSmootherKernel_MPK_final(_Dir_, ...)\
    if(preconType == "TWO-STEP-GAUSS-SEIDEL") \
    {\
        const_array_type* preconInArray;\
        array_type* preconOutArray;\
        int precon_cur_offset=0;\
        int precon_next_offset=0;\
        int startSubPow = 1;\
        int endSubPow = totSubPower-1;\
        preconInArray = b;\
        precon_cur_offset = 0;\
        preconOutArray = x;\
        precon_next_offset = 0;\
        /*xInit should be in preconTmp[0]*/\
        array_type* preconTmp = gmresPolyPreconTmp;\
        BASE_Precon_TWO_STEP_GAUSS_SEIDEL_ ## _Dir_ ();\
    }\
    else if(preconType == "CHEBYSHEV") \
    {\
        const_array_type* preconInArray;\
        array_type* preconOutArray;\
        preconInArray = b;\
        int precon_cur_offset = 0;\
        preconOutArray = x;\
        int precon_next_offset = 0;\
        /*xInit should be in preconTmp[0]*/\
        array_type* preconTmp = gmresPolyPreconTmp;\
        BASE_Chebyshev_MPK();\
   }\


#define GENERATE_KERNEL_MGSmootherKernel_w_RESIDUAL(_Dir_)\
    template <typename packtype>\
    inline void RACE_MGSmootherKernel_ ## _Dir_ ## _KERNEL_w_RESIDUAL(int start, int end, int pow, int subPow, int numa_domain, void* args)\
    {\
        RACE_DECODE_FROM_VOID_MGSmootherKernel_w_Residual(args);\
        int cur_pow = ((pow-1)+arr_offset);\
        int next_pow = (pow+arr_offset);\
        int wrapFactor = 2;\
        int cur_pow_w_wrap = cur_pow%wrapFactor;\
        int next_pow_w_wrap = next_pow%wrapFactor;\
        if(next_pow > 1)\
        {\
            initVecIsZero = false;\
        }\
        if(next_pow==maxSteps)\
        {\
            BASE_MGSmootherKernel_MPK_final_w_RESIDUAL(_Dir_);\
        }\
        else\
        {\
            if(preconType == "CHEBYSHEV" && (next_pow==maxSteps-1)) \
            {\
                BASE_MGSmootherKernel_MPK_final(_Dir_);\
            }\
            else\
            {\
                BASE_MGSmootherKernel_MPK(_Dir_);\
            }\
        }\
    }\


#define GENERATE_KERNEL_MGSmootherKernel(_Dir_)\
    template <typename packtype>\
    inline void RACE_MGSmootherKernel_  ## _Dir_ ## _KERNEL(int start, int end, int pow, int subPow, int numa_domain, void* args)\
    {\
        RACE_DECODE_FROM_VOID_MGSmootherKernel(args);\
        int cur_pow = ((pow-1)+arr_offset);\
        int next_pow = (pow+arr_offset);\
        int wrapFactor = 2;\
        int cur_pow_w_wrap = cur_pow%wrapFactor;\
        int next_pow_w_wrap = next_pow%wrapFactor;\
        if(next_pow > 1)\
        {\
            initVecIsZero = false;\
        }\
        if(next_pow==maxSteps)\
        {\
            BASE_MGSmootherKernel_MPK_final(_Dir_);\
        }\
        else\
        {\
            BASE_MGSmootherKernel_MPK(_Dir_);\
        }\
    }\

    GENERATE_KERNEL_MGSmootherKernel_w_RESIDUAL(FWD);
    GENERATE_KERNEL_MGSmootherKernel(FWD);

    GENERATE_KERNEL_MGSmootherKernel_w_RESIDUAL(BWD);
    GENERATE_KERNEL_MGSmootherKernel(BWD);



       //dispatcher function
    template<typename packtype>
        inline void RACE_MGSmootherKernel_w_Residual_KERNEL(int start, int end,int pow, int subPow, int numa_domain, void* args)
        {
            using arg_type = kernelArgMGSmoother<packtype>;
            arg_type* arg_decode = (arg_type*) args;
            std::string preconType = arg_decode->preconType;

            //only two-step implemented now
            if(arg_decode->fwdDir)
            {
                RACE_MGSmootherKernel_FWD_KERNEL_w_RESIDUAL<packtype>( start, end, pow, subPow, numa_domain, args);
            }
            else
            {
                RACE_MGSmootherKernel_BWD_KERNEL_w_RESIDUAL<packtype>( start, end, pow, subPow, numa_domain, args);
            }
        }

    template<typename packtype>
        inline void RACE_MGSmootherKernel_KERNEL(int start, int end,int pow, int subPow, int numa_domain, void* args)
        {
            using arg_type = kernelArgMGSmoother<packtype>;
            arg_type* arg_decode = (arg_type*) args;
            std::string preconType = arg_decode->preconType;

            //only two-step implemented now
            if(arg_decode->fwdDir)
            {
                RACE_MGSmootherKernel_FWD_KERNEL<packtype>( start, end, pow, subPow, numa_domain, args);
            }
            else
            {
                RACE_MGSmootherKernel_BWD_KERNEL<packtype>( start, end, pow, subPow, numa_domain, args);
            }
        }


} // namespace RACE

#endif
