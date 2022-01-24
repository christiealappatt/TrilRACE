#ifndef _RACE_PACKTYPE_H
#define _RACE_PACKTYPE_H

#include "RACE_CRS_raw.hpp"
#include "Tpetra_CrsMatrix.hpp"

namespace RACE
{
    template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
        class RACE_packtype
        {
            public:
                typedef Scalar SC;
                typedef LocalOrdinal LO;
                typedef GlobalOrdinal GO;
                typedef Node NO;
                typedef Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> CRS_type;
                typedef CRS_raw<RACE_packtype> CRS_raw_type;
               // <Scalar, LocalOrdinal, GlobalOrdinal, Node> CRS_raw_type;
                typedef Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> mvec_type;
                typedef Teuchos::ArrayRCP<Teuchos::ArrayRCP<Scalar> > marray_type;
                typedef Teuchos::ScalarTraits<SC> STS;
                typedef typename STS::magnitudeType real_type;
                typedef typename std::complex<real_type> complex_type;
        };

}

#endif
