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
            LocalOrdinal nrows;
            const size_t* rowPtr;
            const LocalOrdinal* col;
            const Scalar* val;
            using CrsMatrixType = typename packtype::CRS_type;
            CRS_raw():nrows(0),rowPtr(NULL), col(NULL), val(NULL)
            {
            }
            CRS_raw(Teuchos::RCP<CrsMatrixType> A)
            {
                convertFromTpetraCrs(A);
            }
            void convertFromTpetraCrs(Teuchos::RCP<CrsMatrixType> A)
            {
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
                }
                else
                {
                    nrows=0;
                    rowPtr=NULL;
                    col=NULL;
                    val=NULL;
                }
            }
        };
}//namespace RACE

#endif
