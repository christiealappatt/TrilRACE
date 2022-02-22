#ifndef _RACE_KERNELS_H_
#define _RACE_KERNELS_H_

#include "Tpetra_CrsMatrix.hpp"
#include <RACE/interface.h>
#include "TrilinosRACE_config.h"
#include "RACE_SpMV.hpp"
#include "RACE_Precon.hpp"
#include "RACE_GmresSstepKernel.hpp"
#include "RACE_GmresPolyPreconKernel.hpp"
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
            using vec_type = typename packtype::mvec_type;

            CRS_raw_type* A; //matrix
            std::string precType;
            std::string precSide;
            int subPow;
            //currently M has to have same (sub-)sparsity that of A
            //TODO: support preconditioners having different sparsity
            //as that of A.
            //This means M has to be visible in preprocessing stage
            //M is required only if the preconditioning matrix is specified
            //explicilty like in SPAI preconditioners
            CRS_raw_type* M;//preconditioner matrix
            RACE::Interface *ce;

            vec_type* workspacePolyPrecon;
            vec_type* workspaceGmresSstep;

            public:
            kernels(RACE::Interface *_ce_, Teuchos::RCP<CrsMatrixType> _A_, Teuchos::ParameterList& paramList, Teuchos::RCP<CrsMatrixType> _M_=Teuchos::null): A(NULL), M(NULL), ce(NULL), workspacePolyPrecon(NULL), workspaceGmresSstep(NULL), subPow(1)
            {

                init(_ce_, _A_, paramList, _M_);
            }
            kernels(): A(NULL), M(NULL), ce(NULL), workspacePolyPrecon(NULL), workspaceGmresSstep(NULL), precType("NONE"), precSide("NONE"), subPow(1)
            {
            }

            void init(RACE::Interface *_ce_, Teuchos::RCP<CrsMatrixType> _A_, Teuchos::ParameterList& paramList, Teuchos::RCP<CrsMatrixType> _M_=Teuchos::null)
            {
                ce = _ce_;
                if(ce == NULL)
                {
                    ERROR_PRINT("RACE engine not found");
                }
                if(_A_ == Teuchos::null)
                {
                    ERROR_PRINT("Matrix not passed to RACE kernels");
                }
                A=new CRS_raw_type(_A_);

                std::string precon_type = "NONE";
                precon_type = paramList.get("Preconditioner", precon_type);
                std::transform(precon_type.begin(), precon_type.end(), precon_type.begin(), ::toupper);

                std::string precon_side = "NONE";
                precon_side = paramList.get("Preconditioner side", precon_side);
                std::transform(precon_side.begin(), precon_side.end(), precon_side.begin(), ::toupper);

                precType = precon_type;
                precSide = precon_side;
                if((precType != "NONE") && (_M_ != Teuchos::null))
                {
                    M=new CRS_raw_type(_M_);
                }

                subPow = 1;
                //determin subPow
                if((precType != "NONE") && (precType != "JACOBI"))
                {
                    subPow = 2;
                }
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
                if(workspacePolyPrecon)
                {
                    delete workspacePolyPrecon;
                }
                if(workspaceGmresSstep)
                {
                    delete workspaceGmresSstep;
                }
            }
            std::string getPrecType()
            {
                return precType;
            }
            void createPrecon();


//The  second VA_ARGS is for passing arguments to ENCODE operator
#define TrilinosRACE_MPK_KERNEL_BODY(_NAME_, ...)\
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
                int forceSubPower = 1;//enforce subPower to 1
                if(power > 0)
                {
                    array_type x_arr = x.get2dViewNonConst();
                    TrilinosRACE_MPK_KERNEL_BODY(SpMV, A, &x_arr, alpha, beta, 0);
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
            int PreconKernel(int power, vec_type &b, vec_type &x)
            {
                int tunedPow = 1; //no tuning for this kernel as it is normally called for power=1
                int forceSubPower = 1;//enforce subpower to 1
                if(power > 0)
                {
                    if((precType == "GAUSS-SEIDEL")||(precType == "JACOBI-GAUSS-SEIDEL"))
                    {
                        x.putScalar(0); //set to zero initial vector
                    }
                    array_type x_arr = x.get2dViewNonConst();
                    array_type b_arr = b.get2dViewNonConst();

                    TrilinosRACE_MPK_KERNEL_BODY(Precon, A, &b_arr, &x_arr, 0, precType);
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
                    workspace = new vec_type(map, ncols, false);
                }

                if(zeroOut)
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
                //Teuchos::Range1D index(iter, iter+power);
                //Teuchos::RCP<MV> Q_subview  = Q.subViewNonConst (index);

                bool needAllocation = false;
                int forceSubPower = 0;//use already set subPower
                //for preconditioner we have to create temporary work space to hold tunedPow columns
                //iterations
                if((precType != "NONE") && (precType != "JACOBI"))
                {
                    //can limit x column to allocCol but that would mean to
                    //reset it to zero in between, else initial guess for
                    //preconditioner will not be zero
                    allocateVecWorkspace(workspaceGmresSstep, x.getMap(), power+1, true);
                    needAllocation = true;
                }
                if(power > 0)
                {
                    //zero-out x for GS left precon, to enable starting with
                    //zero vec, leave out prev and prev-prev iteration
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


                    if(needAllocation)
                    {
                        array_type preconTmp = workspaceGmresSstep->get2dViewNonConst();
                        TrilinosRACE_MPK_KERNEL_BODY(GmresSstepKernel, A, &x_arr, theta, 0, iter, precType, precSide, &preconTmp);
                    }
                    else
                    {
                        TrilinosRACE_MPK_KERNEL_BODY(GmresSstepKernel, A, &x_arr, theta, 0, iter, precType, precSide, NULL);
                    }

                }
                return tunedPow;
            }

            //MPK_GmresPolyPreconKernel performs the main body of GMRES preconditioner
            //prod stores the final  polynomial
            int MPK_GmresPolyPreconKernel(int power, vec_type &prod, vec_type &y, std::vector<complex_type> theta, int tunedPow)
            {

                //for prod we have to create temporary work space to hold tunedPow columns
                //iterations
                int allocCol = tunedPow;
                bool tunePhase = false;
                int forceSubPower = 0;//use already set subPower
                if(allocCol <= 0)
                {
                    //don't perform any computation during tuning phase since,
                    //y vector is summed up
                    allocCol = ce->getHighestPower();
                    tunePhase = true;
                }
                //last argument is false, since we disable zeroOut as it is not
                //required here
                allocateVecWorkspace(workspacePolyPrecon, prod.getMap(), allocCol+1, false);

                //set first vector to prod
                //Teuchos::Range1D index(0, 0);
                //Teuchos::RCP<vec_type> prod_subview =  workspacePolyPrecon->subViewNonConst (index);
                Teuchos::RCP<vec_type> prod_subview = workspacePolyPrecon->getVectorNonConst(0);
                Tpetra::deep_copy(*prod_subview, prod);

                if(power > 0)
                {
                    array_type x_arr = workspacePolyPrecon->get2dViewNonConst();
                    array_type y_arr = y.get2dViewNonConst();
                    TrilinosRACE_MPK_KERNEL_BODY(GmresPolyPreconKernel, A, &x_arr, &y_arr, theta, 0, power, allocCol, precType, precSide, NULL);
                }

                //now in case of theta(dim-1,1)=0 or SCT::isComplex do one
                //iteration on y
                if(!tunePhase && (theta[power].imag()==0 || packtype::STS::isComplex))
                {
                    int final_col_index = power % (tunedPow+1);
                    //Teuchos::Range1D index_final(final_col_index, final_col_index);
                    //prod_subview =  workspacePolyPrecon->subViewNonConst (index_final);
                    prod_subview = workspacePolyPrecon->getVectorNonConst(final_col_index);
                    //y= y+(1/theta_r)*prod
                    y.update(1.0/(theta[power].real()), *prod_subview, 1);
                }

                return tunedPow;
            }


        }; //class kernels

} //namespace RACE

#endif
