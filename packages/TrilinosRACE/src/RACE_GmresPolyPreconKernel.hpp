#ifndef _RACE_GmresPolyPreconKernel_KERNEL_H
#define _RACE_GmresPolyPreconKernel_KERNEL_H

#include "RACE_CRS_raw.hpp"
#include "RACE_SpMV.hpp"
#include "RACE_Precon.hpp"
#include <RACE/interface.h>
#include "TrilinosRACE_config.h"

namespace RACE {

    template <typename packtype>
    struct kernelArgGmresPolyPrecon : public kernelArgPrecon<packtype>
    {
        using complex_type = typename packtype::complex_type;
        using array_type = typename packtype::marray_type;
        array_type* y;
        //array_type* preconTmp;
        std::vector<complex_type> theta;

        std::string preconType;
        std::string preconSide;
        int maxSteps;
    };

//convenience macros
#define RACE_ENCODE_TO_VOID_GmresPolyPreconKernel(A_en, L_en, U_en, x_en, y_en, initVecIsZero_en, theta_en,innerDamping_en,  arr_offset_en, maxSteps_en, tunedPower_en, totSubPower_en, preconType_en, preconSide_en, gmresPolyPreconTmp_en)\
    using arg_type = kernelArgGmresPolyPrecon<packtype>;\
    arg_type *arg_encode = new arg_type;\
    arg_encode->A = A_en;\
    arg_encode->L = L_en;\
    arg_encode->U = U_en;\
    arg_encode->x = x_en;\
    arg_encode->y = y_en;\
    arg_encode->theta = theta_en;\
    arg_encode->arr_offset = arr_offset_en;\
    arg_encode->maxSteps = maxSteps_en;\
    arg_encode->totSubPower = totSubPower_en;\
    arg_encode->tunedPower = tunedPower_en;\
    arg_encode->preconType = preconType_en;\
    arg_encode->preconSide = preconSide_en;\
    arg_encode->preconTmp = gmresPolyPreconTmp_en;\
    arg_encode->innerDamping = innerDamping_en;\
    arg_encode->initVecIsZero = initVecIsZero_en;\
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
    CRS_raw_type* L = arg_decode->L;\
    CRS_raw_type* U = arg_decode->U;\
    array_type* x = arg_decode->x;\
    array_type* y = arg_decode->y;\
    std::vector<complex_type> theta = arg_decode->theta;\
    int arr_offset = arg_decode->arr_offset;\
    int maxSteps = arg_decode->maxSteps;\
    int totSubPower = arg_decode->totSubPower;\
    int tunedPower = arg_decode->tunedPower;\
    std::string preconType = arg_decode->preconType;\
    std::string preconSide = arg_decode->preconSide;\
    array_type* gmresPolyPreconTmp = arg_decode->preconTmp;\
    double innerDamping = arg_decode->innerDamping;\
    bool initVecIsZero = arg_decode->initVecIsZero;\



#if 0
    //x stores prod
    template <typename packtype>
    inline void RACE_GmresPolyPreconKernel_KERNEL(int start, int end, int pow, int subPow, int numa_domain, void* args)
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
                BASE_SpMV_KERNEL_IN_ROW(A, (*x)[cur_offset]);
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
                    BASE_SpMV_KERNEL_IN_ROW(A, (*x)[cur_offset]);;
                    (*x)[next_offset][row] = 2*theta_r*(*x)[cur_offset][row] - tmp;
                    (*y)[0][row] = (*y)[0][row] + mod_inv*(*x)[next_offset][row];
                }

            }
            else if((pow+arr_offset) != maxSteps)
            {
                for(LocalOrdinal row=start; row<end; ++row)
                {
                    BASE_SpMV_KERNEL_IN_ROW(A, (*x)[cur_offset]);
                    (*x)[next_offset][row] = (*x)[prev_offset][row] - mod_inv*tmp;
                }
            }
        }
    }
#endif

#define BASE_GmresPolyPreconKernel_MPK_w_PRECON(_precon_, ...)\
    if(preconType == "NONE")\
    {\
        array_type* mpkInArray = x;\
        array_type* mpkOutArray = x;\
        int mpk_cur_offset = cur_offset;\
        int mpk_next_offset = next_offset;\
        for(LocalOrdinal row=start; row<end; ++row)\
        {\
            BASE_SpMV_KERNEL_IN_ROW(A, (*mpkInArray)[mpk_cur_offset]);\
            (*mpkOutArray)[mpk_next_offset][row] = tmp;\
            __VA_ARGS__;\
        }\
    }\
    else if(preconType =="JACOBI")\
    {\
        array_type* preconInArray = x;\
        array_type* preconOutArray = x;\
        int precon_cur_offset = cur_offset;\
        int precon_next_offset = next_offset;\
        if(preconSide == "RIGHT")\
        {\
            for(LocalOrdinal row=start; row<end; ++row)\
            {\
                BASE_PRECON_JACOBI_w_SpMV_RIGHT_KERNEL_IN_ROW(A, (*preconInArray)[precon_cur_offset], (*preconOutArray)[precon_next_offset]);\
                (*preconOutArray)[precon_next_offset][row] = tmp;\
                __VA_ARGS__;\
            }\
        }\
        else\
        {\
            for(LocalOrdinal row=start; row<end; ++row)\
            {\
                BASE_PRECON_JACOBI_w_SpMV_LEFT_KERNEL_IN_ROW(A, (*preconInArray)[precon_cur_offset], (*preconOutArray)[precon_next_offset]);\
                (*preconOutArray)[precon_next_offset][row] = tmp;\
                __VA_ARGS__;\
            }\
        }\
    }\
    else if((preconType == "GAUSS-SEIDEL") || (preconType == "JACOBI-GAUSS-SEIDEL"))\
    {\
        if(preconSide == "RIGHT")\
        {\
            array_type* preconInArray = x;\
            array_type* preconOutArray = gmresPolyPreconTmp;\
            int precon_cur_offset = cur_offset;\
            int precon_next_offset = cur_offset;\
            array_type* mpkInArray = gmresPolyPreconTmp;\
            array_type* mpkOutArray = x;\
            int mpk_cur_offset = cur_offset;\
            int mpk_next_offset = next_offset;\
            if((subPow%2) == 1)\
            {\
                for(LocalOrdinal row=start; row<end; ++row)\
                {\
                    BASE_PRECON_ ##_precon_## _KERNEL_IN_ROW(A, (*preconInArray)[precon_cur_offset], (*preconOutArray)[precon_next_offset]);\
                    (*preconOutArray)[precon_next_offset][row] = tmp;\
                }\
            }\
            else\
            {\
                for(LocalOrdinal row=start; row<end; ++row)\
                {\
                    BASE_SpMV_KERNEL_IN_ROW(A, (*mpkInArray)[mpk_cur_offset]);\
                    (*mpkOutArray)[mpk_next_offset][row] = tmp;\
                    __VA_ARGS__;\
                }\
            }\
        }\
        else\
        {\
            array_type* mpkInArray = x;\
            array_type* mpkOutArray = gmresPolyPreconTmp;\
            int mpk_cur_offset = cur_offset;\
            int mpk_next_offset = cur_offset;\
            array_type* preconInArray = gmresPolyPreconTmp;\
            array_type* preconOutArray = x;\
            int precon_cur_offset = cur_offset;\
            int precon_next_offset = next_offset;\
            if((subPow%2) == 1)\
            {\
                for(LocalOrdinal row=start; row<end; ++row)\
                {\
                    BASE_SpMV_KERNEL_IN_ROW(A, (*mpkInArray)[mpk_cur_offset]);\
                    (*mpkOutArray)[mpk_next_offset][row] = tmp;\
                }\
                for(LocalOrdinal row=start; row<end; ++row)\
                {\
                    BASE_PRECON_ ##_precon_## _KERNEL_IN_ROW(A, (*preconInArray)[precon_cur_offset], (*preconOutArray)[precon_next_offset]);\
                    (*preconOutArray)[precon_next_offset][row] = tmp;\
                }\
            }\
            else\
            {\
                for(LocalOrdinal row=start; row<end; ++row)\
                {\
                    __VA_ARGS__;\
                }\
            }\
        }\
    }\
    else if(preconType == "TWO-STEP-GAUSS-SEIDEL") \
    {\
        array_type* mpkInArray;\
        array_type* mpkOutArray;\
        array_type* preconInArray;\
        array_type* preconOutArray;\
        int mpk_cur_offset=0;\
        int mpk_next_offset=0;\
        int precon_cur_offset=0;\
        int precon_next_offset=0;\
        if(preconSide == "RIGHT")\
        {\
            int startSubPow = 1;\
            int endSubPow = totSubPower-1;\
            preconInArray = x;\
            precon_cur_offset = cur_offset;\
            preconOutArray = gmresPolyPreconTmp;\
            precon_next_offset = totSubPower-1;\
            array_type* preconTmp = gmresPolyPreconTmp;\
            BASE_Precon_TWO_STEP_GAUSS_SEIDEL_FWD();\
            if(subPow == totSubPower) /* MPK kernel*/\
            {\
                mpkInArray = gmresPolyPreconTmp;\
                mpk_cur_offset = totSubPower-1;\
                mpkOutArray = x;\
                mpk_next_offset = next_offset;\
                BASE_PRECON_MPK_splitLU(U->diag[row],; __VA_ARGS__);\
            }\
        }\
        else\
        {\
            if(subPow==1)\
            {\
                mpkInArray = x;\
                mpk_cur_offset = cur_offset;\
                mpkOutArray = gmresPolyPreconTmp;\
                mpk_next_offset = subPow-1;\
                BASE_PRECON_MPK_splitUL(U->diag[row]);\
            }\
            /*This works only if initVec == 0*/\
            int startSubPow = 1;\
            int endSubPow = totSubPower-1;\
            preconInArray = gmresPolyPreconTmp;\
            precon_cur_offset = subPow-1;\
            preconOutArray = x;\
            precon_next_offset = next_offset;\
            array_type* preconTmp = gmresPolyPreconTmp;\
            BASE_Precon_TWO_STEP_GAUSS_SEIDEL_FWD();\
            if(subPow == totSubPower)\
            {\
                for(LocalOrdinal row=start; row<end; ++row)\
                {\
                    __VA_ARGS__;\
                }\
            }\
        }\
    }\


#define GENERATE_KERNEL_GmresPolyPreconKernel(_precon_)\
    template <typename packtype>\
    inline void RACE_GmresPolyPreconKernel_ ## _precon_ ## _KERNEL(int start, int end, int pow, int subPow, int numa_domain, void* args)\
    {\
        RACE_DECODE_FROM_VOID_GmresPolyPreconKernel(args);\
        int wrap_fac = std::max(3, tunedPower+1);/*min of 3 to ensure prev_offset access is stored*/\
        /*wrap around offset so we can reuse prod (x),*/\
        /*and keep max. col size of prod with maxSteps*/\
        const int cur_offset = ((pow-1)+arr_offset)%(wrap_fac);\
        const int next_offset = (pow+arr_offset)%(wrap_fac);\
        const int cur_offset_wo_wrapping = (pow-1)+arr_offset;\
        Scalar theta_r = theta[cur_offset_wo_wrapping].real();\
        Scalar theta_r_inv = 1.0/theta_r;\
        Scalar theta_i = theta[cur_offset_wo_wrapping].imag();\
        if((theta_i == 0) || (packtype::STS::isComplex))\
        {\
            BASE_GmresPolyPreconKernel_MPK_w_PRECON(_precon_,\
                    (*y)[0][row] = (*y)[0][row] + theta_r_inv*(*x)[cur_offset][row];\
                    (*x)[next_offset][row] = (*x)[cur_offset][row] - theta_r_inv*(*x)[next_offset][row];\
                    );\
        }\
        else \
        {\
            Scalar mod = theta_r*theta_r + theta_i*theta_i;\
            Scalar mod_inv = 1/mod;\
            bool isConj = false;\
            if(cur_offset_wo_wrapping != 0)\
            {\
                complex_type prev_theta = theta[cur_offset_wo_wrapping-1];\
                if(std::conj(theta[cur_offset_wo_wrapping-1]) == theta[cur_offset_wo_wrapping])\
                {\
                    isConj=true;\
                }\
            }\
            const int prev_offset = ((pow-2)+arr_offset)%(wrap_fac);\
            /*If previous was not conjugate, else should take the other branch*/\
            if(!isConj)\
            {\
                BASE_GmresPolyPreconKernel_MPK_w_PRECON(_precon_,\
                        (*x)[next_offset][row] = 2*theta_r*(*x)[cur_offset][row] - (*x)[next_offset][row];\
                        (*y)[0][row] = (*y)[0][row] + mod_inv*(*x)[next_offset][row];\
                        );\
            }\
            else /*if((pow+arr_offset) < maxSteps)*/\
            {\
                BASE_GmresPolyPreconKernel_MPK_w_PRECON(_precon_,\
                        (*x)[next_offset][row] = (*x)[prev_offset][row] - mod_inv*(*x)[next_offset][row];\
                        );\
            }\
        }\
    }\


    GENERATE_KERNEL_GmresPolyPreconKernel(NONE);
    GENERATE_KERNEL_GmresPolyPreconKernel(GAUSS_SEIDEL);
    GENERATE_KERNEL_GmresPolyPreconKernel(JACOBI_GAUSS_SEIDEL);


       //dispatcher function
    template<typename packtype>
        inline void RACE_GmresPolyPreconKernel_KERNEL(int start, int end,int pow, int subPow, int numa_domain, void* args)
        {
            using arg_type = kernelArgGmresPolyPrecon<packtype>;
            arg_type* arg_decode = (arg_type*) args;
            std::string preconType = arg_decode->preconType;
            std::string preconSide = arg_decode->preconSide;

            if(preconType == "JACOBI")
            {
                RACE_GmresPolyPreconKernel_NONE_KERNEL<packtype>( start, end, pow, subPow, numa_domain, args);
            }
            else if(preconType == "GAUSS-SEIDEL")
            {
                RACE_GmresPolyPreconKernel_GAUSS_SEIDEL_KERNEL<packtype>( start, end, pow, subPow, numa_domain, args);
            }
            else if(preconType == "JACOBI-GAUSS-SEIDEL")
            {
                RACE_GmresPolyPreconKernel_JACOBI_GAUSS_SEIDEL_KERNEL<packtype>( start, end, pow, subPow, numa_domain, args);
            }
            else
            {
                RACE_GmresPolyPreconKernel_NONE_KERNEL<packtype>( start, end, pow, subPow, numa_domain, args);
            }
        }


} // namespace RACE

#endif
