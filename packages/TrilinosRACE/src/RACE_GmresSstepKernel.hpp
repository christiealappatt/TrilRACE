#ifndef _RACE_GmresSstepKernel_KERNEL_H
#define _RACE_GmresSstepKernel_KERNEL_H

#include "RACE_CRS_raw.hpp"
#include <RACE/interface.h>
#include "TrilinosRACE_config.h"
#include "RACE_SpMV.hpp"
#include "RACE_Precon.hpp"

namespace RACE {

    template <typename packtype>
    struct kernelArgGmresSstep : public kernelArgSpMV<packtype>
    {
        using complex_type = typename packtype::complex_type;
        using array_type = typename packtype::marray_type;

        array_type* preconTmp;
        std::vector<complex_type> theta;

        std::string preconType;
        std::string preconSide;

        int glb_arr_offset;
    };

//convenience macros
#define RACE_ENCODE_TO_VOID_GmresSstepKernel(A_en, x_en, theta_en, arr_offset_en, glb_arr_offset_en, preconType_en, preconSide_en, preconTmp_en)\
    using arg_type = kernelArgGmresSstep<packtype>;\
    arg_type *arg_encode = new arg_type;\
    arg_encode->A = A_en;\
    arg_encode->x = x_en;\
    arg_encode->theta = theta_en;\
    arg_encode->arr_offset = arr_offset_en;\
    arg_encode->glb_arr_offset = glb_arr_offset_en;\
    arg_encode->preconType = preconType_en;\
    arg_encode->preconSide = preconSide_en;\
    arg_encode->preconTmp = preconTmp_en;\
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
    int glb_arr_offset = arg_decode->glb_arr_offset;\
    std::string preconType = arg_decode->preconType;\
    std::string preconSide = arg_decode->preconSide;\
    array_type* preconTmp = arg_decode->preconTmp;\

 //the _VA_ARGS_ are added for the update kernel
#define BASE_GmresSstepKernel_MPK(...)\
    for(LocalOrdinal row=start; row<end; ++row)\
    {\
        BASE_SpMV_KERNEL_IN_ROW(A, (*mpkInArray)[mpk_cur_offset]);\
        (*mpkOutArray)[mpk_next_offset][row] = tmp  __VA_ARGS__;\
    }\


    //the _VA_ARGS_ are added for the update kernel
#define BASE_GmresSstepKernel_PRECON(_precon_, ...)\
    for(LocalOrdinal row=start; row<end; ++row)\
    {\
        BASE_PRECON_ ## _precon_ ## _KERNEL_IN_ROW(A, (*preconInArray)[precon_cur_offset], (*preconOutArray)[precon_next_offset]);\
        (*preconOutArray)[precon_next_offset][row] = tmp __VA_ARGS__;\
    }\

    //TODO:complex variant not tested yet
#define GENERATE_KERNEL_GmresSstepKernel_true_type(_precon_)\
    template <typename packtype>\
    inline void RACE_GmresSstepKernel_ ## _precon_ ## _KERNEL(std::true_type, int start, int end, int pow, int subPow, int numa_domain, void* args)\
    {\
        RACE_DECODE_FROM_VOID_GmresSstepKernel(args);\
        const int cur_offset = (pow-1)+arr_offset;\
        const int next_offset = pow+arr_offset;\
        array_type* mpkInArray = x;\
        array_type* mpkOutArray = x;\
        int mpk_cur_offset = cur_offset+glb_arr_offset;\
        int mpk_next_offset = next_offset+glb_arr_offset;\
        if(preconType == "NONE")\
        {\
            if(theta[cur_offset] == 0)\
            {\
                BASE_GmresSstepKernel_MPK();\
            }\
            else\
            {\
                BASE_GmresSstepKernel_MPK(- theta[cur_offset]*(*mpkOutArray)[cur_offset+glb_arr_offset][row]);\
            }\
        }\
        else if(preconType =="JACOBI")\
        {\
            array_type* preconInArray = x;\
            int precon_cur_offset = cur_offset+glb_arr_offset;\
            array_type* preconOutArray = x;\
            int precon_next_offset = next_offset+glb_arr_offset;\
            if(preconSide == "RIGHT")\
            {\
                if(theta[cur_offset] == 0)\
                {\
                    BASE_GmresSstepKernel_PRECON(JACOBI_w_SpMV_RIGHT);\
                }\
                else\
                {\
                    BASE_GmresSstepKernel_PRECON(JACOBI_w_SpMV_RIGHT, -theta[cur_offset]*(*preconOutArray)[cur_offset+glb_arr_offset][row]);\
                }\
            }\
            else\
            {\
                if(theta[cur_offset] == 0)\
                {\
                    BASE_GmresSstepKernel_PRECON(JACOBI_w_SpMV_LEFT);\
                }\
                else\
                {\
                    BASE_GmresSstepKernel_PRECON(JACOBI_w_SpMV_LEFT, -theta*(*preconOutArray)[cur_offset+glb_arr_offset][row]);\
                }\
            }\
        }\
        else if((preconType == "GAUSS-SEIDEL") || (preconType == "JACOBI-GAUSS-SEIDEL"))\
        {\
            if(preconSide == "RIGHT")\
            {\
                array_type* preconInArray = x;\
                int precon_cur_offset = cur_offset+glb_arr_offset;\
                array_type* preconOutArray = preconTmp;\
                int precon_next_offset = cur_offset;\
                mpkInArray = preconTmp;\
                mpk_cur_offset = cur_offset;\
                mpkOutArray = x;\
                mpk_next_offset = next_offset+glb_arr_offset;\
                if((subPow%2) == 1) /*right-prec*/\
                {\
                    BASE_GmresSstepKernel_PRECON(_precon_);\
                }\
                else /*MPK kernel*/\
                {\
                    if(theta[cur_offset] == 0)\
                    {\
                        BASE_GmresSstepKernel_MPK();\
                    }\
                    else\
                    {\
                        BASE_GmresSstepKernel_MPK(-theta*(*mpkOutArray)[cur_offset+glb_arr_offset][row]);\
                    }\
                }\
            }\
            else\
            {\
                mpkInArray = x;\
                mpk_cur_offset = cur_offset+glb_arr_offset;\
                mpkOutArray = preconTmp;\
                mpk_next_offset = cur_offset;\
                array_type* preconInArray = preconTmp;\
                int precon_cur_offset = cur_offset;\
                array_type* preconOutArray = x;\
                int precon_next_offset = next_offset+glb_arr_offset;\
                if((subPow%2) == 1) /*MPK-kernel*/\
                {\
                    BASE_GmresSstepKernel_MPK();\
                    BASE_GmresSstepKernel_PRECON(_precon_);  /*for left GS Preconditioner, GS has only direct (row) dependency, but no indirect.*/\
                                                             /*This means it can be done in current phase. But update shouldn't be performed now since*/\
                                                             /*x_output is updated by GS and has indirect dependency within GS*/\
                }\
                else /*left-prec*/\
                {\
                    /*Update need not be done outside RACE, but just keep it*/\
                    /*here so no need to treat GS left as a special case*/\
                    if(theta_i != 0)\
                    {\
                        for(LocalOrdinal row=start; row<end; ++row)\
                        {\
                            (*preconOutArray)[next_offset+glb_arr_offset][row] += -theta*(*preconOutArray)[cur_offset+glb_arr_offset][row]; /*update kernel*/\
                        }\
                    }\
                }\
            }\
        }\
    }\



#define GENERATE_KERNEL_GmresSstepKernel_false_type(_precon_)\
    template <typename packtype>\
    inline void RACE_GmresSstepKernel_ ## _precon_ ## _KERNEL(std::false_type, int start, int end, int pow, int subPow, int numa_domain, void* args)\
    {\
        RACE_DECODE_FROM_VOID_GmresSstepKernel(args);\
        const int cur_offset = (pow-1)+arr_offset;\
        const int next_offset = pow+arr_offset;\
        const int prev_offset = (pow-2)+arr_offset;\
        Scalar theta_r = theta[cur_offset].real();\
        Scalar theta_i = theta[cur_offset].imag();\
        Scalar theta_i_sq = theta_i*theta_i;\
        array_type* mpkInArray = x;\
        array_type* mpkOutArray = x;\
        int mpk_cur_offset = cur_offset+glb_arr_offset;\
        int mpk_next_offset = next_offset+glb_arr_offset;\
        if(preconType == "NONE")\
        {\
            if(theta_i == 0)\
            {\
                BASE_GmresSstepKernel_MPK(-theta_r*(*mpkOutArray)[cur_offset+glb_arr_offset][row]);\
            }\
            else\
            {\
                BASE_GmresSstepKernel_MPK(-theta_r*(*mpkOutArray)[cur_offset+glb_arr_offset][row] + theta_i_sq*(*mpkOutArray)[prev_offset+glb_arr_offset][row]);\
            }\
        }\
        else if(preconType =="JACOBI")\
        {\
            array_type* preconInArray = x;\
            int precon_cur_offset = cur_offset+glb_arr_offset;\
            array_type* preconOutArray = x;\
            int precon_next_offset = next_offset+glb_arr_offset;\
            if(preconSide == "RIGHT")\
            {\
                if(theta_i == 0)\
                {\
                    BASE_GmresSstepKernel_PRECON(JACOBI_w_SpMV_RIGHT, -theta_r*(*preconOutArray)[cur_offset+glb_arr_offset][row]);\
                }\
                else\
                {\
                    BASE_GmresSstepKernel_PRECON(JACOBI_w_SpMV_RIGHT, -theta_r*(*preconOutArray)[cur_offset+glb_arr_offset][row] + theta_i_sq*(*preconOutArray)[prev_offset+glb_arr_offset][row]);\
                }\
            }\
            else\
            {\
                if(theta_i == 0)\
                {\
                    BASE_GmresSstepKernel_PRECON(JACOBI_w_SpMV_LEFT, -theta_r*(*preconOutArray)[cur_offset+glb_arr_offset][row]);\
                }\
                else\
                {\
                    BASE_GmresSstepKernel_PRECON(JACOBI_w_SpMV_LEFT, -theta_r*(*preconOutArray)[cur_offset+glb_arr_offset][row] + theta_i_sq*(*preconOutArray)[prev_offset+glb_arr_offset][row]);\
                }\
            }\
        }\
        else if((preconType == "GAUSS-SEIDEL") || (preconType == "JACOBI-GAUSS-SEIDEL"))\
        {\
            if(preconSide == "RIGHT")\
            {\
                array_type* preconInArray = x;\
                int precon_cur_offset = cur_offset+glb_arr_offset;\
                array_type* preconOutArray = preconTmp;\
                int precon_next_offset = cur_offset;\
                mpkInArray = preconTmp;\
                mpk_cur_offset = cur_offset;\
                mpkOutArray = x;\
                mpk_next_offset = next_offset+glb_arr_offset;\
                if((subPow%2) == 1) /*right-prec*/\
                {\
                    BASE_GmresSstepKernel_PRECON(_precon_);\
                }\
                else /*MPK kernel*/\
                {\
                    if(theta_i == 0)\
                    {\
                        BASE_GmresSstepKernel_MPK(-theta_r*(*mpkOutArray)[cur_offset+glb_arr_offset][row]);\
                    }\
                    else\
                    {\
                        BASE_GmresSstepKernel_MPK(-theta_r*(*mpkOutArray)[cur_offset+glb_arr_offset][row] + theta_i_sq*(*mpkOutArray)[prev_offset+glb_arr_offset][row]);\
                    }\
                }\
            }\
            else\
            {\
                mpkInArray = x;\
                mpk_cur_offset = cur_offset+glb_arr_offset;\
                mpkOutArray = preconTmp;\
                mpk_next_offset = cur_offset;\
                array_type* preconInArray = preconTmp;\
                int precon_cur_offset = cur_offset;\
                array_type* preconOutArray = x;\
                int precon_next_offset = next_offset+glb_arr_offset;\
                if((subPow%2) == 1) /*MPK-kernel*/\
                {\
                    BASE_GmresSstepKernel_MPK();\
                    BASE_GmresSstepKernel_PRECON(_precon_);  /*for left GS Preconditioner, GS has only direct (row) dependency, but no indirect.*/\
                                                             /*This means it can be done in current phase. But update shouldn't be performed now since*/\
                                                             /*x_output is updated by GS and has indirect dependency within GS*/\
                }\
                else /*left-prec*/\
                {\
                    /*Update need not be done outside RACE, but just keep it*/\
                    /*here so no need to treat GS left as a special case*/\
                    if(theta_i == 0)\
                    {\
                        for(LocalOrdinal row=start; row<end; ++row)\
                        {\
                            (*preconOutArray)[next_offset+glb_arr_offset][row] += -theta_r*(*preconOutArray)[cur_offset+glb_arr_offset][row]; /*Update kernel*/\
                        }\
                    }\
                    else\
                    {\
                        for(LocalOrdinal row=start; row<end; ++row)\
                        {\
                            (*preconOutArray)[next_offset+glb_arr_offset][row] += -theta_r*(*preconOutArray)[cur_offset+glb_arr_offset][row] + theta_i_sq*(*preconOutArray)[prev_offset+glb_arr_offset][row]; /*Update kernel*/\
                        }\
                    }\
                }\
            }\
        }\
    }\


    //generate actual kernels
    //false-type
    GENERATE_KERNEL_GmresSstepKernel_false_type(NONE);
    GENERATE_KERNEL_GmresSstepKernel_false_type(GAUSS_SEIDEL);
    GENERATE_KERNEL_GmresSstepKernel_false_type(JACOBI_GAUSS_SEIDEL);
    //true-type
    GENERATE_KERNEL_GmresSstepKernel_true_type(NONE);
    GENERATE_KERNEL_GmresSstepKernel_true_type(GAUSS_SEIDEL);
    GENERATE_KERNEL_GmresSstepKernel_true_type(JACOBI_GAUSS_SEIDEL);


    //dispatcher function
    template<typename packtype>
        inline void RACE_GmresSstepKernel_KERNEL(int start, int end,int pow, int subPow, int numa_domain, void* args)
        {
            using arg_type = kernelArgGmresSstep<packtype>;
            arg_type* arg_decode = (arg_type*) args;
            std::string preconType = arg_decode->preconType;
            std::string preconSide = arg_decode->preconSide;

            //RACE_GmresSstepKernel_KERNEL<packtype>(std::integral_constant<bool, packtype::STS::isComplex>{}, start, end, pow, subPow, numa_domain, args);
            if(preconType == "JACOBI")
            {
                RACE_GmresSstepKernel_NONE_KERNEL<packtype>(std::integral_constant<bool, packtype::STS::isComplex>{}, start, end, pow, subPow, numa_domain, args);
            }
            else if(preconType == "GAUSS-SEIDEL")
            {
                RACE_GmresSstepKernel_GAUSS_SEIDEL_KERNEL<packtype>(std::integral_constant<bool, packtype::STS::isComplex>{}, start, end, pow, subPow, numa_domain, args);
            }
            else if(preconType == "JACOBI-GAUSS-SEIDEL")
            {
                RACE_GmresSstepKernel_JACOBI_GAUSS_SEIDEL_KERNEL<packtype>(std::integral_constant<bool, packtype::STS::isComplex>{}, start, end, pow, subPow, numa_domain, args);
            }
            else
            {
                RACE_GmresSstepKernel_NONE_KERNEL<packtype>(std::integral_constant<bool, packtype::STS::isComplex>{}, start, end, pow, subPow, numa_domain, args);
            }
        }

} // namespace RACE

#endif
