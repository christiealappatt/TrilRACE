#ifndef _RACE_CRS_RAW_H
#define _RACE_CRS_RAW_H

#include "Tpetra_CrsMatrix.hpp"

namespace RACE {

    //this is just a wrapper for CRS sparsemat
    template <class packtype>
        struct CRS_raw
        {
            using Scalar = typename packtype::SC;
            using LocalOrdinal = typename packtype::LO;
            using GlobalOrdinal = typename packtype::GO;
            LocalOrdinal nrows;
            const size_t* rowPtr;
            const LocalOrdinal* col;
            const Scalar* val;
            const Scalar* invDiag;
            const Scalar* diag;
            bool self_managed_mem;
            private:
            Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal>* invDiag_vec;
            Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal>* diag_vec;
            using Vector = typename Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal>;
            public:
            using CrsMatrixType = typename packtype::CRS_type;
            CRS_raw():nrows(0),rowPtr(NULL), col(NULL), val(NULL), invDiag(NULL), self_managed_mem(true)
            {
            }
            CRS_raw(Teuchos::RCP<CrsMatrixType> A)
            {
                self_managed_mem = false;
                convertFromTpetraCrs(A);
            }

            ~CRS_raw()
            {
                if(self_managed_mem)
                {
                    if(val)
                    {
                        delete[] val;
                    }
                    if(rowPtr)
                    {
                        delete[] rowPtr;
                    }
                    if(col)
                    {
                        delete[] col;
                    }
                    //invDiag and diag are pointers, only creator destroys
                    //(i.e., self managed_mem=false)
                }
                else
                {
                    if(invDiag_vec)
                    {
                        delete invDiag_vec;
                    }
                    if(diag_vec)
                    {
                        delete diag_vec;
                    }
                }
            }

            //to transfer from a different library the data structure
            //need to be called after contructor
            void initCover(const LocalOrdinal nrows_, const Scalar *val_, const size_t* rowPtr_, const LocalOrdinal* col_, const Scalar* diag_, const Scalar* invDiag_)
            {
                nrows=nrows_;
                //nnz=nnz_;
                val=val_;
                rowPtr=rowPtr_;
                col=col_;
                diag=diag_;
                invDiag=invDiag_;
            }

            void convertFromTpetraCrs(Teuchos::RCP<CrsMatrixType> A)
            {
                invDiag = NULL;
                diag = NULL;
                if(A!=Teuchos::null)
                {
                    nrows = A->getLocalNumRows();
                    Teuchos::ArrayRCP<const size_t> rowPointers;
                    Teuchos::ArrayRCP<const LocalOrdinal> columnIndices;
                    Teuchos::ArrayRCP<const Scalar> values;
                    //A->getAllValues(rowPointers, columnIndices, values);//deprecated
                    rowPointers = Kokkos::Compat::persistingView(A->getLocalRowPtrsHost());
                    columnIndices = Kokkos::Compat::persistingView(A->getLocalIndicesHost());
                    values = Teuchos::arcp_reinterpret_cast<const Scalar>(
                        Kokkos::Compat::persistingView(A->getLocalValuesHost(
                                                        Tpetra::Access::ReadOnly)));
                    rowPtr = rowPointers.getRawPtr();
                    col = columnIndices.getRawPtr();
                    val = values.getRawPtr();
                    diag_vec = new Vector(A->getRowMap());
                    A->getLocalDiagCopy(*diag_vec);
                    diag = (diag_vec->getData()).getRawPtr();
                    invDiag_vec = new Vector(A->getRowMap());
                    invDiag_vec->reciprocal(*diag_vec);
                    invDiag = (invDiag_vec->getData()).getRawPtr();
                    /*for(int i=0; i<10; ++i)
                    {
                        printf("2.invDiag[%d] = %f\n", i, invDiag[i]);
                    }*/
                }
                else
                {
                    nrows=0;
                    rowPtr=NULL;
                    col=NULL;
                    val=NULL;
                    invDiag=NULL;
                }
            }
            //splits to strictly upper and lower triangle part
            void splitMatrixToLU(CRS_raw **L_ptr, CRS_raw  **U_ptr, bool row_normalize=false)
            {

                size_t* L_rowPtr = new size_t[nrows+1];
                size_t* U_rowPtr = new size_t[nrows+1];

                L_rowPtr[0] = 0;
                U_rowPtr[0] = 0;

                //NUMA init
#pragma omp parallel for schedule(static)
                for(int row=0; row<nrows; ++row)
                {
                    L_rowPtr[row+1] = 0;
                    U_rowPtr[row+1] = 0;
                }

                int L_nnz = 0;
                int U_nnz = 0;
                for(int row=0; row<nrows; ++row)
                {
                    for(int idx=rowPtr[row]; idx<rowPtr[row+1]; ++idx)
                    {
                        if(col[idx] > row)
                        {
                            ++U_nnz;
                        }
                        else if(col[idx] < row)
                        {
                            ++L_nnz;
                        }
                    }
                    L_rowPtr[row+1] = L_nnz;
                    U_rowPtr[row+1] = U_nnz;
                }

                Scalar* L_val = new Scalar[L_nnz];
                LocalOrdinal* L_col = new int[L_nnz];
                Scalar* U_val = new Scalar[U_nnz];
                LocalOrdinal* U_col = new int[U_nnz];

                //with NUMA init
#pragma omp parallel for schedule(static)
                for(int row=0; row<nrows; ++row)
                {
                    int L_ctr = L_rowPtr[row];
                    int U_ctr = U_rowPtr[row];
                    for(int idx=rowPtr[row]; idx<rowPtr[row+1]; ++idx)
                    {
                        double scale=1;
                        if(row_normalize)
                        {
                            scale = invDiag[row];
                        }
                        if(col[idx]>row)
                        {
                            U_col[U_ctr] = col[idx];
                            U_val[U_ctr] = val[idx]*scale;
                            ++U_ctr;
                        }
                        else if(col[idx] < row)
                        {
                            L_col[L_ctr] = col[idx];
                            L_val[L_ctr] = val[idx]*scale;
                            ++L_ctr;
                        }
                    }
                }

                (*L_ptr) = new CRS_raw;
                (*U_ptr) = new CRS_raw;
                (*L_ptr)->initCover(nrows, L_val, L_rowPtr, L_col, diag, invDiag);
                (*U_ptr)->initCover(nrows, U_val, U_rowPtr, U_col, diag, invDiag);

            }

        };
}//namespace RACE

#endif
