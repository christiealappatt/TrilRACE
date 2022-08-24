#ifndef _RACE_KERNELS_H_
#define _RACE_KERNELS_H_

#include "Tpetra_CrsMatrix.hpp"
#include <RACE/interface.h>
#include "TrilinosRACE_config.h"
#include "RACE_SpMV.hpp"
#include "RACE_Precon.hpp"
#include "RACE_GmresSstepKernel.hpp"
#include "RACE_GmresPolyPreconKernel.hpp"
#include "RACE_MGSmootherKernel.hpp"
#include <string.h>
#include <vector>

namespace RACE {

    template <typename packtype>
        class kernels
        {
            private:
            using CrsMatrixType = typename packtype::CRS_type;
            using CRS_raw_type = typename packtype::CRS_raw_type;
            using Scalar = typename packtype::SC;
            using complex_type = typename packtype::complex_type;
            using array_type = typename packtype::marray_type;
            using const_array_type = typename packtype::const_marray_type;
            using vec_type = typename packtype::mvec_type;

            preProcess<packtype>* pre_process;
            CRS_raw_type* A; //matrix
            CRS_raw_type* L; //matrix
            CRS_raw_type* U; //matrix
            std::string precType;
            std::string precSide;
            int subPow;
            double gamma;
            double lambdaMin;
            double lambdaMax;
            double eigRatio;
            //currently M has to have same (sub-)sparsity that of A
            //TODO: support preconditioners having different sparsity
            //as that of A.
            //This means M has to be visible in preprocessing stage
            //M is required only if the preconditioning matrix is specified
            //explicilty like in SPAI preconditioners
            CRS_raw_type* M;//preconditioner matrix
            RACE::Interface *ce;

            vec_type* workspacePolyPrecon;
            vec_type* workspacePolyPrecon2;
            vec_type* workspaceGmresSstep;
            vec_type* workspacePrecon;

            public:
            kernels(preProcess<packtype>* _preProcess_, Teuchos::RCP<CrsMatrixType> _M_=Teuchos::null): pre_process(_preProcess_), A(NULL), L(NULL), U(NULL), M(NULL), ce(NULL), workspacePolyPrecon(NULL), workspacePolyPrecon2(NULL), workspaceGmresSstep(NULL), workspacePrecon(NULL), precType("NONE"), precSide("NONE"), subPow(1), gamma(1), lambdaMin(std::nan("")), lambdaMax(std::nan("")), eigRatio(std::nan("")), paramUptodate(false)
            {

                init(_M_);
            }
            bool paramUptodate;

            /*Disable empty constructor
             * kernels(): A(NULL), L(NULL), U(NULL), M(NULL), ce(NULL), workspacePolyPrecon(NULL), workspacePolyPrecon2(NULL), workspaceGmresSstep(NULL), workspacePrecon(NULL), precType("NONE"), precSide("NONE"), subPow(1), gamma(1), lambdaMin(std::nan("")), lambdaMax(std::nan("")), eigRatio(std::nan(""))
            {
            }*/

            void init(Teuchos::RCP<CrsMatrixType> _M_=Teuchos::null)
            {
                ce = pre_process->get_RACE_engine();
                if(ce == NULL)
                {
                    ERROR_PRINT("RACE engine not found");
                }
                Teuchos::RCP<CrsMatrixType> permA = pre_process->getPermutedMatrix();
                if(permA == Teuchos::null)
                {
                    ERROR_PRINT("Matrix not passed to RACE kernels");
                }
                A=new CRS_raw_type(permA);
                std::string precon_type = pre_process->getPreconType();

                if(precon_type == "TWO-STEP-GAUSS-SEIDEL")
                {
                    A->splitMatrixToLU(&L, &U, true);
                }
                std::string precon_side = "NONE";
                precon_side = pre_process->getPreconSide();
                //std::transform(precon_side.begin(), precon_side.end(), precon_side.begin(), ::toupper);

                precType = precon_type;
                precSide = precon_side;

                if((precType != "NONE") && (_M_ != Teuchos::null))
                {
                    M=new CRS_raw_type(_M_);
                }

                subPow = pre_process->getInnerPower();

                setupParams();
                //determin subPow
                /*if((precType != "NONE") && (precType != "JACOBI"))
                {
                    subPow = 2;
                }*/

            }

            void setupParams()
            {
                if(precType == "CHEBYSHEV")
                {
                    lambdaMin = pre_process->getLambdaMin();
                    lambdaMax = pre_process->getLambdaMax();
                    eigRatio = pre_process->getEigRatio();
                }
                if(precType == "TWO-STEP-GAUSS-SEIDEL")
                {
                    gamma = pre_process->getInnerDamping();
                }
                paramUptodate = true;
            }

            ~kernels()
            {
                if(A)
                {
                    delete A;
                }
                if(M)
                {
                    delete M;
                }
                if(L)
                {
                    delete L;
                }
                if(U)
                {
                    delete U;
                }
                if(workspacePolyPrecon)
                {
                    delete workspacePolyPrecon;
                }
                if(workspacePolyPrecon2)
                {
                    delete workspacePolyPrecon2;
                }
                if(workspaceGmresSstep)
                {
                    delete workspaceGmresSstep;
                }
                if(workspacePrecon)
                {
                    delete workspacePrecon;
                }
            }
            std::string getPrecType()
            {
                return precType;
            }
            void createPrecon();


//The third VA_ARGS is for passing arguments to ENCODE operator
#define TrilinosRACE_MPK_KERNEL_BODY(_NAME_, ZERO_OUT_CODE, ...)\
        {\
            /*pass only pointer of x_arr, else copying this would mean each*/\
            /*thread will have a private copy of x_arr in the function and*/\
            /*it will destroy it by all threads, causing segfault*/\
            RACE_ENCODE_TO_VOID_ ## _NAME_(__VA_ARGS__);\
            /*setting power to -1 here because I am not going to run with*/\
            /*it and I am resetting it next*/\
            int race_power_id = ce->registerFunction(&RACE_ ## _NAME_ ## _KERNEL<packtype>, voidArg, -1, subPow);\
            if(forceSubPower > 0)\
            {\
                ce->setSubPower(race_power_id, forceSubPower);\
            }\
            if(precType == "GAUSS-SEIDEL")\
            {\
                ce->setSerial(race_power_id); /*do only in serial for GAUSS-SEIDEL*/\
            }\
            int bestPow = tunedPow;\
            if(bestPow == -1)\
            {\
                /*tuneFunction sets to bestPow*/\
                bestPow = ce->tuneFunction(race_power_id);\
            }\
            tunedPow=bestPow;\
            RACE_ ## _NAME_ ## _setTunedPower(bestPow);\
            if(power <= bestPow)\
            {\
                ce->setPower(race_power_id, power);\
                ce->executeFunction(race_power_id);\
            }\
            else /*need multiple iterations with bestPow*/\
            {\
                ce->setPower(race_power_id, bestPow);\
                int powBlocks = static_cast<int>(power/((double)bestPow));\
                int remPow = power%bestPow;\
                for(int block=0; block<powBlocks; ++block)\
                {\
                    RACE_ ## _NAME_ ## _setOffset(block*bestPow);/*offset array access for the iteration*/\
                    ce->executeFunction(race_power_id);\
                    ZERO_OUT_CODE;\
                }\
                if(remPow > 0)\
                {\
                    RACE_ ## _NAME_ ## _setOffset(powBlocks*bestPow);/*offset array access for the iteration*/\
                    ce->setPower(race_power_id, remPow);\
                    ce->executeFunction(race_power_id);\
                }\
            }\
            RACE_DELETE_ARG_ ## _NAME_();\
            if(precType == "GAUSS-SEIDEL")\
            {\
                ce->unsetSerial(race_power_id); /*do only in serial for GAUSS-SEIDEL*/\
            }\
        }\


            //MPK performs for i=1,...,power {x[i+1]=beta*x[i+1]+alpha*A*x[i]}; where x[i] is a vector of length nrows. Assumes x[0] is the initial input vector
            int MPK(int power, vec_type &x, Scalar alpha, Scalar beta, int tunedPow)
            {
                if(!paramUptodate)
                {
                    setupParams();
                }
                int forceSubPower = 1;//enforce subPower to 1
                if(power > 0)
                {
                    array_type x_arr = x.get2dViewNonConst();
                    TrilinosRACE_MPK_KERNEL_BODY(SpMV, , A, &x_arr, alpha, beta, 0);
                }

#if 0
                int race_power_id = ce->registerFunction(&RACE_SpMV_KERNEL<packtype>, voidArg, power);
                {
                    ce->executeFunction(race_power_id);
                }
                RACE_DELETE_ARG_SpMV();
#endif
                return tunedPow; //bestPow == tunedPow, if it is not tuning phase
            }

            //Precon kernels, just performs preconditioning
            int PreconKernel(int power, const vec_type &b, vec_type &x, bool fwdDir)
            {
                if(!paramUptodate)
                {
                    setupParams();
                }

                int tunedPow = 1; //no tuning for this kernel as it is normally called for power=1
                int forceSubPower = 1;//enforce subpower to 1

                int innerIter = subPow-1;
                if(power > 0)
                {
                    if( ((precType == "GAUSS-SEIDEL")||(precType == "JACOBI-GAUSS-SEIDEL")) || (precType == "TWO-STEP-GAUSS-SEIDEL"))        {
                        x.putScalar(0); //set to zero initial vector

                        if(precType == "TWO-STEP-GAUSS-SEIDEL")
                        {
                            forceSubPower = innerIter;
                            allocateVecWorkspace(workspacePrecon, x.getMap(), innerIter, true);
                        }
                    }
                    array_type x_arr = x.get2dViewNonConst();
                    const_array_type b_arr = b.get2dView();

                    bool initVecIsZero = true;
                    if(precType == "TWO-STEP-GAUSS-SEIDEL")
                    {
                        array_type preconTmp = workspacePrecon->get2dViewNonConst();
                        TrilinosRACE_MPK_KERNEL_BODY(Precon, , A, L, U, &b_arr, &x_arr, initVecIsZero, gamma, 0, innerIter, precType, fwdDir, &preconTmp);
                    }
                    else
                    {
                        TrilinosRACE_MPK_KERNEL_BODY(Precon, , A, L, U, &b_arr, &x_arr, initVecIsZero, gamma, 0, innerIter, precType, fwdDir, NULL);
                    }
                }
                return tunedPow;
            }

            using map_type = typename vec_type::map_type;
            //returns true if new workspace created, else
            //false
            bool allocateVecWorkspace(vec_type* &workspace, const Teuchos::RCP<const map_type>& map, int ncols, bool zeroOut=true)
            {
                bool newAllocate = false;

                if(workspace)
                {
                    //check if already available worspace is sufficient to hold
                    //the vectors
                    if(workspace->getNumVectors() < ncols)
                    {
                        //free previous workspace
                        delete workspace;
                        workspace = NULL;
                        newAllocate = true;
                    }
                }
                else
                {
                    newAllocate = true;
                }

                if(newAllocate)
                {
                    workspace = new vec_type(map, ncols, zeroOut);
                }
                else if(zeroOut)
                {
                    workspace->putScalar(0);
                }
            }

            //MPK_GmresSstepKernel performs the main apply kernel for GmresSstep
            //if tunedPow=-1 then tuning is performed, else tunedPow (last
            //argument) is taken as internal stepsize for RACE
            //Complete x (with restart number of columns passed) and iter passed
            //since we need to access prev-to-prev offset if theta_i!=0
            int MPK_GmresSstepKernel(int power, int iter, vec_type &x, std::vector<complex_type> theta, int tunedPow)
            {
                if(!paramUptodate)
                {
                    setupParams();
                }
                //Teuchos::Range1D index(iter, iter+power);
                //Teuchos::RCP<MV> Q_subview  = Q.subViewNonConst (index);

                bool needAllocation = false;
                int forceSubPower = 0;//use already set subPower
                int innerIter = subPow;
                //for preconditioner we have to create temporary work space to hold tunedPow columns
                //iterations
                if((precType != "NONE") && (precType != "JACOBI")   )
                {
                    if(precType == "TWO-STEP-GAUSS-SEIDEL")
                    {
                        allocateVecWorkspace(workspaceGmresSstep, x.getMap(), innerIter, true);
                    }
                    else
                    {
                        //can limit x column to allocCol but that would mean to
                        //reset it to zero in between, else initial guess for
                        //preconditioner will not be zero
                        allocateVecWorkspace(workspaceGmresSstep, x.getMap(), power+1, true);
                    }
                    needAllocation = true;
                }
                if(power > 0)
                {
                    //zero-out x for GS left precon, to enable starting with
                    //zero vec, leave out prev iteration
                    if( ((precType == "GAUSS-SEIDEL")||(precType == "JACOBI-GAUSS-SEIDEL")) && (precSide == "LEFT"))
                    {
                        //Teuchos::Range1D index(1, x.getNumVectors()-1);
                        Teuchos::Range1D index(iter+1, iter+power);
                        Teuchos::RCP<vec_type> x_subview  = x.subViewNonConst (index);
                        x_subview->putScalar(0);
                    }
                    array_type x_arr = x.get2dViewNonConst();
                    //ensure first iteration does not have access to prev_offset,
                    //i.e., set imaginary part of first theta to 0
                    //theta[0] = complex_type(theta[0].real(), 0);


                    bool initVecIsZero = true;
                    if(needAllocation)
                    {
                        array_type preconTmp = workspaceGmresSstep->get2dViewNonConst();
                        TrilinosRACE_MPK_KERNEL_BODY(GmresSstepKernel, , A, L, U, &x_arr, initVecIsZero, theta, gamma, 0, iter, innerIter, precType, precSide, &preconTmp);
                    }
                    else
                    {
                        TrilinosRACE_MPK_KERNEL_BODY(GmresSstepKernel, , A, L, U, &x_arr, initVecIsZero, theta, gamma, 0, iter, innerIter, precType, precSide, NULL);
                    }

                }
                return tunedPow;
            }

            //MPK_GmresPolyPreconKernel performs the main body of GMRES preconditioner
            //prod stores the final  polynomial
            int MPK_GmresPolyPreconKernel(int power, vec_type &prod, vec_type &y, std::vector<complex_type> theta, int tunedPow)
            {
                if(!paramUptodate)
                {
                    setupParams();
                }
                //for prod we have to create temporary work space to hold tunedPow columns
                //iterations
                int allocCol = tunedPow;
                bool tunePhase = false;
                int innerIter = subPow;

                if(allocCol <= 0)
                {
                    //don't perform any computation during tuning phase since,
                    //y vector is summed up
                    allocCol = ce->getHighestPower();
                    tunePhase = true;
                }
                allocCol = std::max(2, allocCol);//ensure minimum of 2, due to access of prev_offset in case of theta_i != 0
                //last argument is false, since we disable zeroOut as it is not
                //required here
                allocateVecWorkspace(workspacePolyPrecon, prod.getMap(), allocCol+1, true);

                //set first vector to prod
                //Teuchos::Range1D index(0, 0);
                //Teuchos::RCP<vec_type> prod_subview =  workspacePolyPrecon->subViewNonConst (index);
                Teuchos::RCP<vec_type> prod_subview = workspacePolyPrecon->getVectorNonConst(0);
                Tpetra::deep_copy(*prod_subview, prod);

                bool needAllocation = false;
                //for preconditioner we have to create temporary work space to hold tunedPow columns
                //iterations
                if((precType != "NONE") && (precType != "JACOBI"))
                {
                    if(precType == "TWO-STEP-GAUSS-SEIDEL")
                    {
                        allocateVecWorkspace(workspacePolyPrecon2, prod.getMap(), innerIter, true);
                    }
                    else
                    {
                        allocateVecWorkspace(workspacePolyPrecon2, prod.getMap(), allocCol+1, true);
                    }
                    needAllocation = true;
                }

                int forceSubPower = 0;//use already set subPower
                bool initVecIsZero = true;
                if(power > 0)
                {

                    array_type x_arr = workspacePolyPrecon->get2dViewNonConst();
                    array_type y_arr = y.get2dViewNonConst();
                    if(needAllocation)
                    {
                        array_type preconTmp = workspacePolyPrecon2->get2dViewNonConst();

                        TrilinosRACE_MPK_KERNEL_BODY(GmresPolyPreconKernel, workspacePolyPrecon2->putScalar(0);, A, L, U, &x_arr, &y_arr, initVecIsZero, theta, gamma, 0, power, allocCol, innerIter, precType, precSide, &preconTmp);
                    }
                    else
                    {
                        TrilinosRACE_MPK_KERNEL_BODY(GmresPolyPreconKernel, , A, L, U, &x_arr, &y_arr, initVecIsZero, theta, gamma, 0, power, allocCol, innerIter, precType, precSide, NULL);
                    }
                }

                //now in case of theta(dim-1,1)=0 or SCT::isComplex do one
                //iteration on y
                if(!tunePhase && (theta[power].imag()==0 || packtype::STS::isComplex))
                {
                    int final_col_index = power % (allocCol+1);
                    //Teuchos::Range1D index_final(final_col_index, final_col_index);
                    //prod_subview =  workspacePolyPrecon->subViewNonConst (index_final);
                    prod_subview = workspacePolyPrecon->getVectorNonConst(final_col_index);
                    //y= y+(1/theta_r)*prod
                    y.update(1.0/(theta[power].real()), *prod_subview, 1);
                }

                return tunedPow;
            }

            //MPK_MGSmoother performs 'n' sweeps of the MG smoother
            int MPK_MGSmootherKernel(int power, vec_type &x, vec_type &b, bool zeroGuess, bool fwdDir, int tunedPow)
            {
                if(!paramUptodate)
                {
                    setupParams();
                }
                //for prod we have to create temporary work space to hold tunedPow columns
                //iterations
                int allocCol = tunedPow;
                bool tunePhase = false;
                int innerIter = subPow;

                if(tunedPow <= 0)
                {
                    tunePhase = true;
                }

                bool needAllocation = false;
                std::vector<double>* scaleCoeff = new std::vector<double>(power);
                std::vector<double>* scale2Coeff = new std::vector<double>(power);
                if(precType == "TWO-STEP-GAUSS-SEIDEL")
                {
                    allocateVecWorkspace(workspacePolyPrecon, x.getMap(), innerIter, true);
                    needAllocation = true;
                }
                else if(precType == "CHEBYSHEV")
                {
                    //tmp of size 3 arrays is only required in this case
                    allocateVecWorkspace(workspacePolyPrecon, x.getMap(), 3, true);
                    needAllocation=true;

                    //compute coefficients for Chebyshev
                    double alpha = lambdaMax/eigRatio;
                    double boostFactor = 1.1;
                    double beta = boostFactor*lambdaMax;
                    double delta = 2.0/(beta-alpha);
                    double theta = (beta+alpha)/2.0;
                    double s1 = theta*delta;

                    (*scaleCoeff)[0] = 1.0/theta;
                    (*scale2Coeff)[0] = 0.0;//not used
                    double rhok = 1.0/s1;
                    for(int p=1; p<power; ++p)
                    {
                        double rhokp1 = 1.0/(2*s1-rhok);
                        (*scaleCoeff)[p] = 2.0*rhokp1*delta;
                        (*scale2Coeff)[p] = rhokp1*rhok;
                        rhok = rhokp1;
                    }
                }
                else
                {
                    printf("Error: only Two-stage Gauss-Seidel and Chebyshev implemented currently for MG smoother\n");
                }
                if(!zeroGuess && needAllocation)
                {
                    //set first vector to xInit value
                    //Teuchos::Range1D index(0, 0);
                    //Teuchos::RCP<vec_type> prod_subview =  workspacePolyPrecon->subViewNonConst (index);
                    Teuchos::RCP<vec_type> x_subview = workspacePolyPrecon->getVectorNonConst(0);
                    Tpetra::deep_copy(*x_subview, x);
                }

                int forceSubPower = 0;//use already set subPower
                bool initVecIsZero = zeroGuess;
                if(power > 0)
                {

                    array_type x_arr = x.get2dViewNonConst();
                    const_array_type b_arr = b.get2dView();
                    if(needAllocation)
                    {
                        array_type preconTmp = workspacePolyPrecon->get2dViewNonConst();

                        TrilinosRACE_MPK_KERNEL_BODY(MGSmootherKernel, ;, A, L, U, &x_arr, &b_arr, NULL, initVecIsZero, gamma, scaleCoeff, scale2Coeff, 0, power, allocCol, innerIter, precType, fwdDir, &preconTmp);
                    }
                    else
                    {

                        printf("Error: only Two-stage Gauss-Seidel implemented currently for smoother\n");
                        //TrilinosRACE_MPK_KERNEL_BODY(GmresPolyPreconKernel, , A, L, U, &x_arr, &y_arr, initVecIsZero, theta, gamma, 0, power, allocCol, innerIter, precType, NULL);
                    }
                }

                if(precType == "CHEBYSHEV")
                {
                    delete scaleCoeff;
                    delete scale2Coeff;
                }
                return tunedPow;
            }

            //MPK_MGSmoother performs 'n' sweeps of the MG smoother
            int MPK_MGSmootherKernel(int power, vec_type &x, vec_type &b, vec_type &r, bool zeroGuess, bool fwdDir, int tunedPow)
            {
                if(!paramUptodate)
                {
                    setupParams();
                }

                if(precType == "CHEBYSHEV")
                {
                    //extra sweep to calculate residual
                    //this is not required for TWO-STEP-GAUSS-SEIDEL since it is
                    //dealt internally with subPow
                    power += 1;
                }

                //for prod we have to create temporary work space to hold tunedPow columns
                //iterations
                int allocCol = tunedPow;
                bool tunePhase = false;
                int innerIter = subPow;

                if(tunedPow <= 0)
                {
                    tunePhase = true;
                }

                //power-1 since last sweep is for residual
                std::vector<double>* scaleCoeff = new std::vector<double>(power-1);
                std::vector<double>* scale2Coeff = new std::vector<double>(power-1);
                bool needAllocation = false;
                if(precType == "TWO-STEP-GAUSS-SEIDEL")
                {
                    allocateVecWorkspace(workspacePolyPrecon, x.getMap(), innerIter, true);
                    needAllocation = true;
                }
                else if(precType == "CHEBYSHEV")
                {
                    //tmp of size 3 arrays is only required in this case
                    allocateVecWorkspace(workspacePolyPrecon, x.getMap(), 3, true);
                    needAllocation=true;
                    //printf("lambdaMax = %f, eigRatio =%f\n", lambdaMax, eigRatio);
                    //compute coefficients for Chebyshev
                    double alpha = lambdaMax/eigRatio;
                    double boostFactor = 1.1;
                    double beta = boostFactor*lambdaMax;
                    double delta = 2.0/(beta-alpha);
                    double theta = (beta+alpha)/2.0;
                    double s1 = theta*delta;

                    (*scaleCoeff)[0] = 1.0/theta;
                    (*scale2Coeff)[0] = 0.0;//not used
                    double rhok = 1.0/s1;
                    for(int p=1; p<power-1; ++p)
                    {
                        double rhokp1 = 1.0/(2*s1-rhok);
                        (*scaleCoeff)[p] = 2.0*rhokp1*delta;
                        (*scale2Coeff)[p] = rhokp1*rhok;
                        rhok = rhokp1;
                    }
/*
                    for(int p=0; p<power-1; ++p)
                    {
                        printf("scale[%d] = %f\n", p, (*scaleCoeff)[p]);
                        printf("scale2[%d] = %f\n", p, (*scale2Coeff)[p]);
                    }
                    */
                }
                else
                {
                    printf("Error: only Two-stage Gauss-Seidel and Chebyshev implemented currently for MG smoother\n");
                }
                if(!zeroGuess && needAllocation)
                {
                    //set first vector to xInit value
                    //Teuchos::Range1D index(0, 0);
                    //Teuchos::RCP<vec_type> prod_subview =  workspacePolyPrecon->subViewNonConst (index);
                    Teuchos::RCP<vec_type> x_subview = workspacePolyPrecon->getVectorNonConst(0);
                    Tpetra::deep_copy(*x_subview, x);
                }

                int forceSubPower = 0;//use already set subPower
                bool initVecIsZero = zeroGuess;
                if(power > 0)
                {

                    array_type x_arr = x.get2dViewNonConst();
                    const_array_type b_arr = b.get2dView();
                    array_type r_arr = r.get2dViewNonConst();
                    if(needAllocation)
                    {
                        array_type preconTmp = workspacePolyPrecon->get2dViewNonConst();

                        TrilinosRACE_MPK_KERNEL_BODY(MGSmootherKernel_w_Residual, ;, A, L, U, &x_arr, &b_arr, &r_arr, initVecIsZero, gamma, scaleCoeff, scale2Coeff, 0, power, allocCol, innerIter, precType, fwdDir, &preconTmp);
                    }
                    else
                    {

                        printf("Error: only Two-stage Gauss-Seidel and Chebyshev implemented currently for smoother\n");
                        //TrilinosRACE_MPK_KERNEL_BODY(GmresPolyPreconKernel, , A, L, U, &x_arr, &y_arr, initVecIsZero, theta, gamma, 0, power, allocCol, innerIter, precType, NULL);
                    }
                }

                return tunedPow;
            }


        }; //class kernels

} //namespace RACE

#endif
