#ifndef _RACE_FRONTEND_H_
#define _RACE_FRONTEND_H_

#include "RACE_pre_process.hpp"
#include "RACE_kernels.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "RACE_packtype.hpp"

namespace RACE
{

    //TODO: template on CRS and MV types, so CRS and MV can have different types
//template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
    template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
        class frontend
        {
            using packtype = RACE_packtype<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
            using CrsMatrixType = typename packtype::CRS_type;
            using preProcess_type = preProcess<packtype>;
            using kernels_type = kernels<packtype>;

            // Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

            preProcess_type pre;
            kernels_type exec;

            public:
            //constructor
            frontend(Teuchos::RCP<CrsMatrixType> origA_, Teuchos::ParameterList& paramList, Teuchos::RCP<CrsMatrixType> M=Teuchos::null): pre(origA_, paramList), exec(&pre)
            {
                //Teuchos::RCP<CrsMatrixType> permA = pre.getPermutedMatrix();
                //exec.init(&pre);
            }

            Teuchos::RCP<CrsMatrixType> getPermutedMatrix()
            {
                return pre.getPermutedMatrix();
            }

            int* getPerm()
            {
                return pre.getPerm();
            }

            int* getInvPerm()
            {
                return pre.getInvPerm();
            }

            using MV = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
            //TODO: permToOrig
            void origToPerm(Teuchos::RCP<MV> &dest_vec, Teuchos::RCP<const MV> src_vec)
            {
                if(src_vec->getLocalLength() != pre.nrows)
                {
                    ERROR_PRINT("Error in dimension");
                }

                Teuchos::ArrayRCP<Teuchos::ArrayRCP<const Scalar> > src_ptr = src_vec->get2dView();
                Teuchos::ArrayRCP<Teuchos::ArrayRCP<Scalar> >      dest_ptr = dest_vec->get2dViewNonConst();


                for(size_t k=0; k < src_vec->getNumVectors(); k++)
                {
                    for(LocalOrdinal i=0; (size_t)i< src_vec->getLocalLength(); i++)
                    {
                        int perm_row = i;
                        if(pre.perm)
                        {
                            perm_row = pre.perm[i];
                        }

                        dest_ptr[k][i] = src_ptr[k][perm_row];

                    }
                }
            }

            using vec_type = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

            void updateParamList(Teuchos::ParameterList newParams)
            {
                pre.updateParamList(newParams);
                exec.paramUptodate = false;
            }

            void setupKernels()
            {
                exec.setupParams();
            }

            int apply(int power, vec_type &x, Scalar alpha=Teuchos::ScalarTraits<Scalar>::one(), Scalar beta = Teuchos::ScalarTraits<Scalar>::zero(), int tunedPow=1)
            {
                std::string precType = exec.getPrecType();
                if( (precType=="NONE") || (precType=="JACOBI") )
                {
                    return exec.MPK(power, x, alpha, beta, tunedPow);
                }
                else
                {
                    ERROR_PRINT("MPK with %s preconditioner not implemented yet in RACE", precType.c_str());
                    return -2;
                }
            }

            using complex_type = typename packtype::complex_type;

            int apply_Precon(int power, const vec_type &b, vec_type &x, bool fwdDir=true)
            {
                // timer
                Teuchos::RCP< Teuchos::Time > timer  = Teuchos::TimeMonitor::getNewCounter ("RACE::Prec-apply");
                Teuchos::TimeMonitor LocalTimer (*timer);

                std::string precType = exec.getPrecType();
                if( (precType=="NONE" || precType=="JACOBI") || (precType=="GAUSS-SEIDEL" || precType=="JACOBI-GAUSS-SEIDEL") || (precType=="TWO-STEP-GAUSS-SEIDEL") )
                {
                    return exec.PreconKernel(power, b, x, fwdDir);
                }
                else
                {
                    ERROR_PRINT("%s preconditioner kernel not implemented yet in RACE", precType.c_str());
                    return -2;
                }
            }

            int apply_GmresSstep(int power, int iter, vec_type &x, std::vector<complex_type> theta, int tunedPow=1)
            {
                // timer
                Teuchos::RCP< Teuchos::Time > timer  = Teuchos::TimeMonitor::getNewCounter ("RACE::GmresSstep kernel");
                Teuchos::TimeMonitor LocalTimer (*timer);

                std::string precType = exec.getPrecType();
                if( (precType=="NONE" || precType=="JACOBI") || (precType=="GAUSS-SEIDEL" || precType=="JACOBI-GAUSS-SEIDEL") || (precType=="TWO-STEP-GAUSS-SEIDEL") )
                {
                    return exec.MPK_GmresSstepKernel(power, iter, x, theta, tunedPow);
                }
                else
                {
                    ERROR_PRINT("GMRES-s-step with %s preconditioner not implemented yet in RACE", precType.c_str());
                    return -2;
                }
            }


            int apply_GmresPolyPrecon(int power, vec_type &prod, vec_type &y, std::vector<complex_type> theta, int tunedPow=1)
            {
                // timer
                Teuchos::RCP< Teuchos::Time > timer  = Teuchos::TimeMonitor::getNewCounter ("RACE::GmresPoly kernel");
                Teuchos::TimeMonitor LocalTimer (*timer);


                //step size and use it
                std::string precType = exec.getPrecType();
                if( (precType=="NONE" || precType=="JACOBI") || (precType=="GAUSS-SEIDEL" || precType=="JACOBI-GAUSS-SEIDEL") || (precType=="TWO-STEP-GAUSS-SEIDEL") )
                {
                    return exec.MPK_GmresPolyPreconKernel(power, prod, y, theta, tunedPow);
                }
                else
                {
                    ERROR_PRINT("GMRES polynomial preconditioner with %s preconditioner not implemented yet in RACE", precType.c_str());
                    return -2;
                }
            }

            int apply_Smoother(int sweeps, vec_type &x, vec_type &b, bool zeroGuess=false, bool fwdDir=true, int tunedPow=1)
            {
                // timer
                Teuchos::RCP< Teuchos::Time > timer  = Teuchos::TimeMonitor::getNewCounter ("RACE::MGSmoother kernel");
                Teuchos::TimeMonitor LocalTimer (*timer);


                //step size and use it
                std::string precType = exec.getPrecType();
                if( (precType=="TWO-STEP-GAUSS-SEIDEL") || (precType=="CHEBYSHEV") )
                {
                    return exec.MPK_MGSmootherKernel(sweeps, x, b, zeroGuess, fwdDir, tunedPow);
                }
                else
                {
                    ERROR_PRINT("MG Smoother with %s preconditioner not implemented yet in RACE", precType.c_str());
                    return -2;
                }
            }

            //fused with Residual computation
            int apply_Smoother(int sweeps, vec_type &x, vec_type &b, vec_type &res, bool zeroGuess=false, bool fwdDir=true, int tunedPow=1)
            {
                // timer
                Teuchos::RCP< Teuchos::Time > timer  = Teuchos::TimeMonitor::getNewCounter ("RACE::MGSmoother+residual kernel");
                Teuchos::TimeMonitor LocalTimer (*timer);


                //step size and use it
                std::string precType = exec.getPrecType();
                if( (precType=="TWO-STEP-GAUSS-SEIDEL") || (precType=="CHEBYSHEV") )
                {
                    return exec.MPK_MGSmootherKernel(sweeps, x, b, res, zeroGuess, fwdDir, tunedPow);
                }
                else
                {
                    ERROR_PRINT("MG Smoother with %s preconditioner not implemented yet in RACE", precType.c_str());
                    return -2;
                }
            }

        };//class frontend
}//namespace RACE

#endif
