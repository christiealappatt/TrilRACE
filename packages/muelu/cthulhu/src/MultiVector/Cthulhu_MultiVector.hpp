#ifndef CTHULHU_MULTIVECTOR_HPP
#define CTHULHU_MULTIVECTOR_HPP

/* this file is automatically generated - do not edit (see script/interfaces.py) */

#include <Teuchos_LabeledObject.hpp>
#include <Teuchos_DataAccess.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_Range1D.hpp>
#include <Kokkos_MultiVector.hpp>
#include <Kokkos_DefaultArithmetic.hpp>
#include "Cthulhu_ConfigDefs.hpp"
#include "Cthulhu_DistObject.hpp"
#include "Cthulhu_Map.hpp"

namespace Cthulhu {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  // forward declaration of Vector, needed to prevent circular inclusions
  template<class S, class LO, class GO, class N> class Vector;
#endif

  template <class Scalar, class LocalOrdinal = int, class GlobalOrdinal = LocalOrdinal, class Node = Kokkos::DefaultNode::DefaultNodeType>
  class MultiVector
    : public DistObject< Scalar, LocalOrdinal, GlobalOrdinal, Node >
  {

  public:

    //! @name Constructor/Destructor Methods
    //@{ 

    //! Destructor.
    virtual ~MultiVector() { }

   //@}

    //! @name Post-construction modification routines
    //@{

    //! Initialize all values in a multi-vector with specified value.
    virtual void putScalar(const Scalar &value)= 0;

    //! Set multi-vector values to random numbers.
    virtual void randomize()= 0;

    //@}

    //! @name Data Copy and View get methods
    //@{

    //! 
    virtual Teuchos::ArrayRCP< const Scalar > getData(size_t j) const = 0;

    //! 
    virtual Teuchos::ArrayRCP< Scalar > getDataNonConst(size_t j)= 0;

    //@}

    //! @name Mathematical methods
    //@{

    //! Computes dot product of each corresponding pair of vectors, dots[i] = this[i].dot(A[i]).
    virtual void dot(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Teuchos::ArrayView< Scalar > &dots) const = 0;

    //! Puts element-wise absolute values of input Multi-vector in target: A = abs(this).
    virtual void abs(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A)= 0;

    //! Puts element-wise reciprocal values of input Multi-vector in target, this(i,j) = 1/A(i,j).
    virtual void reciprocal(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A)= 0;

    //! Scale the current values of a multi-vector, this = alpha*this.
    virtual void scale(const Scalar &alpha)= 0;

    //! Update multi-vector values with scaled values of A, this = beta*this + alpha*A.
    virtual void update(const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Scalar &beta)= 0;

    //! Update multi-vector with scaled values of A and B, this = gamma*this + alpha*A + beta*B.
    virtual void update(const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Scalar &beta, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &B, const Scalar &gamma)= 0;

    //! Compute 1-norm of each vector in multi-vector.
    virtual void norm1(const Teuchos::ArrayView< typename Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const = 0;

    //! Compute 2-norm of each vector in multi-vector.
    virtual void norm2(const Teuchos::ArrayView< typename Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const = 0;

    //! Compute Inf-norm of each vector in multi-vector.
    virtual void normInf(const Teuchos::ArrayView< typename Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const = 0;

    //! Compute Weighted 2-norm (RMS Norm) of each vector in multi-vector.
    virtual void normWeighted(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &weights, const Teuchos::ArrayView< typename Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const = 0;

    //! Compute mean (average) value of each vector in multi-vector.
    virtual void meanValue(const Teuchos::ArrayView< Scalar > &means) const = 0;

    //! Matrix-Matrix multiplication, this = beta*this + alpha*op(A)*op(B).
    virtual void multiply(Teuchos::ETransp transA, Teuchos::ETransp transB, const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &B, const Scalar &beta)= 0;

    //! Element-wise multiply of a Vector A with a MultiVector B.
    virtual void elementWiseMultiply(Scalar scalarAB, const Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &B, Scalar scalarThis)= 0;

    //@}

    //! @name Attribute access functions
    //@{

    //! Returns the number of vectors in the multi-vector.
    virtual size_t getNumVectors() const = 0;

    //! Returns the local vector length on the calling processor of vectors in the multi-vector.
    virtual size_t getLocalLength() const = 0;

    //! Returns the global vector length of vectors in the multi-vector.
    virtual global_size_t getGlobalLength() const = 0;

    //@}

    //! @name Overridden from Teuchos::Describable
    //@{

    //! Return a simple one-line description of this object.
    virtual std::string description() const = 0;

    //! Print the object with some verbosity level to an FancyOStream object.
    virtual void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel=Teuchos::Describable::verbLevel_default) const = 0;

    //@}

    //! @name Cthulhu specific
    //@{
 
    //! Set seed for Random function.
    virtual void setSeed(unsigned int seed)= 0;

    //! Compute max value of each vector in multi-vector.
    virtual void maxValue(const Teuchos::ArrayView< Scalar > &maxs) const= 0;

    //@}

  }; // MultiVector class

} // Cthulhu namespace

#define CTHULHU_MULTIVECTOR_SHORT
#endif // CTHULHU_MULTIVECTOR_HPP
