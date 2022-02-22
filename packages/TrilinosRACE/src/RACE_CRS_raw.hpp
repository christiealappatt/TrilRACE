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
            private:
            Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal> invDiag_vec;
            using Vector = typename Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal>;
            public:
            using CrsMatrixType = typename packtype::CRS_type;
            CRS_raw():nrows(0),rowPtr(NULL), col(NULL), val(NULL), invDiag(NULL)
            {
            }
            CRS_raw(Teuchos::RCP<CrsMatrixType> A)
            {
                convertFromTpetraCrs(A);
            }
            void convertFromTpetraCrs(Teuchos::RCP<CrsMatrixType> A)
            {
                invDiag = NULL;//TODO
                if(A!=Teuchos::null)
                {
                    nrows = A->getNodeNumRows();
                    Teuchos::ArrayRCP<const size_t> rowPointers;
                    Teuchos::ArrayRCP<const LocalOrdinal> columnIndices;
                    Teuchos::ArrayRCP<const Scalar> values;
                    A->getAllValues(rowPointers, columnIndices, values);
                    rowPtr = rowPointers.getRawPtr();
                    col = columnIndices.getRawPtr();
                    val = values.getRawPtr();
                    invDiag_vec = Vector(A->getRowMap());
                    A->getLocalDiagCopy(invDiag_vec);
                    invDiag_vec.reciprocal(invDiag_vec);
                    invDiag = (invDiag_vec.getData()).getRawPtr();
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
        };
}//namespace RACE

#endif
