// @HEADER
// ************************************************************************
//
//                           Intrepid2 Package
//                 Copyright (2007) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Kyungjoo Kim  (kyukim@sandia.gov), or
//                    Mauro Perego  (mperego@sandia.gov)
//
// ************************************************************************
// @HEADER

/** \file test_01.cpp
    \brief  Unit tests for the Intrepid2::G_HEX_C1_FEM class.
    \author Created by P. Bochev, D. Ridzal, K. Peterson and Kyungjoo Kim
*/
#include "Intrepid2_config.h"

#ifdef HAVE_INTREPID2_DEBUG
#define INTREPID2_TEST_FOR_DEBUG_ABORT_OVERRIDE_TO_CONTINUE
#endif

#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"

#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_RCP.hpp"

namespace Intrepid2 {

  namespace Test {
    
#define INTREPID2_TEST_ERROR_EXPECTED( S, nthrow, ncatch )              \
    try {                                                               \
      ++nthrow;                                                         \
      S ;                                                               \
    }                                                                   \
    catch (std::logic_error err) {                                      \
      ++ncatch;                                                         \
      *outStream << "Expected Error ----------------------------------------------------------------\n"; \
      *outStream << err.what() << '\n';                                 \
      *outStream << "-------------------------------------------------------------------------------" << "\n\n"; \
    }
    
    template<typename ValueType, typename DeviceSpaceType>
    int HGRAD_HEX_C1_FEM_Test01(const bool verbose) {
      
      typedef ValueType value_type;
      
      Teuchos::RCP<std::ostream> outStream;
      Teuchos::oblackholestream bhs; // outputs nothing
      
      if (verbose)
        outStream = Teuchos::rcp(&std::cout, false);
      else
        outStream = Teuchos::rcp(&bhs,       false);
      
      Teuchos::oblackholestream oldFormatState;
      oldFormatState.copyfmt(std::cout);
      
      typedef typename
        Kokkos::Impl::is_space<DeviceSpaceType>::host_mirror_space::execution_space HostSpaceType ;
      
      *outStream << "DeviceSpace::  "; DeviceSpaceType::print_configuration(*outStream, false);
      *outStream << "HostSpace::    ";   HostSpaceType::print_configuration(*outStream, false);
      
      *outStream                                                       
        << "===============================================================================\n"
        << "|                                                                             |\n"
        << "|                 Unit Test (Basis_HGRAD_HEX_C1_FEM)                          |\n"
        << "|                                                                             |\n"
        << "|     1) Conversion of Dof tags into Dof ordinals and back                    |\n"
        << "|     2) Basis values for VALUE, GRAD, CURL, and Dk operators                 |\n"
        << "|                                                                             |\n"
        << "|  Questions? Contact  Pavel Bochev  (pbboche@sandia.gov),                    |\n"
        << "|                      Denis Ridzal  (dridzal@sandia.gov),                    |\n"
        << "|                      Kara Peterson (kjpeter@sandia.gov).                    |\n"
        << "|                                                                             |\n"
        << "|  Intrepid's website: http://trilinos.sandia.gov/packages/intrepid           |\n"
        << "|  Trilinos website:   http://trilinos.sandia.gov                             |\n"
        << "|                                                                             |\n"
        << "===============================================================================\n";

      typedef Kokkos::DynRankView<value_type,DeviceSpaceType> DynRankView;
#define ConstructWithLabel(obj, ...) obj(#obj, __VA_ARGS__)

      const value_type tol = Parameters::Tolerence;
      int errorFlag = 0;

      *outStream
        << "\n"
        << "===============================================================================\n"
        << "| TEST 1: Basis creation, exceptions tests                                    |\n"
        << "===============================================================================\n";

      ordinal_type nthrow = 0, ncatch = 0;
      try {
#ifdef HAVE_INTREPID2_DEBUG
        Basis_HGRAD_HEX_C1_FEM<DeviceSpaceType> hexBasis;
        
        // Define array containing the 8 vertices of the reference HEX, its center and 6 face centers
        DynRankView ConstructWithLabel(hexNodes, 15, 3);
        hexNodes(0,0) = -1.0;  hexNodes(0,1) = -1.0;  hexNodes(0,2) = -1.0;
        hexNodes(1,0) =  1.0;  hexNodes(1,1) = -1.0;  hexNodes(1,2) = -1.0;
        hexNodes(2,0) =  1.0;  hexNodes(2,1) =  1.0;  hexNodes(2,2) = -1.0;
        hexNodes(3,0) = -1.0;  hexNodes(3,1) =  1.0;  hexNodes(3,2) = -1.0;
        
        hexNodes(4,0) = -1.0;  hexNodes(4,1) = -1.0;  hexNodes(4,2) =  1.0;
        hexNodes(5,0) =  1.0;  hexNodes(5,1) = -1.0;  hexNodes(5,2) =  1.0;
        hexNodes(6,0) =  1.0;  hexNodes(6,1) =  1.0;  hexNodes(6,2) =  1.0;
        hexNodes(7,0) = -1.0;  hexNodes(7,1) =  1.0;  hexNodes(7,2) =  1.0;  
        
        hexNodes(8,0) =  0.0;  hexNodes(8,1) =  0.0;  hexNodes(8,2) =  0.0;
        
        hexNodes(9,0) =  1.0;  hexNodes(9,1) =  0.0;  hexNodes(9,2) =  0.0;
        hexNodes(10,0)= -1.0;  hexNodes(10,1)=  0.0;  hexNodes(10,2)=  0.0;
        
        hexNodes(11,0)=  0.0;  hexNodes(11,1)=  1.0;  hexNodes(11,2)=  0.0;
        hexNodes(12,0)=  0.0;  hexNodes(12,1)= -1.0;  hexNodes(12,2)=  0.0;
        
        hexNodes(13,0)=  0.0;  hexNodes(13,1)=  0.0;  hexNodes(13,2)=  1.0;
        hexNodes(14,0)=  0.0;  hexNodes(14,1)=  0.0;  hexNodes(14,2)= -1.0;
        
        // Generic array for the output values; needs to be properly resized depending on the operator type
        const auto numFields = hexBasis.getCardinality();
        const auto numPoints = hexNodes.dimension(0);
        const auto spaceDim  = hexBasis.getBaseCellTopology().getDimension();
        const auto D2Cardin  = getDkCardinality(OPERATOR_D2, spaceDim);

        const auto workSize  = numFields*numPoints*D2Cardin;
        DynRankView ConstructWithLabel(work, workSize);

        // resize vals to rank-2 container with dimensions
        DynRankView vals = DynRankView(work.data(), numFields, numPoints);
        {
          // exception #1: CURL cannot be applied to scalar functions in 3D
          // resize vals to rank-3 container with dimensions (num. basis functions, num. points, arbitrary)
          DynRankView tmpvals = DynRankView(work.data(), numFields, numPoints, spaceDim);
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getValues(tmpvals, hexNodes, OPERATOR_CURL), nthrow, ncatch );
        }
        {
          // exception #2: DIV cannot be applied to scalar functions in 3D
          // resize vals to rank-2 container with dimensions (num. basis functions, num. points)
          DynRankView tmpvals = DynRankView(work.data(), numFields, numPoints);
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getValues(tmpvals, hexNodes, OPERATOR_DIV), nthrow, ncatch );
        }

        // Exceptions 3-7: all bf tags/bf Ids below are wrong and should cause getDofOrdinal() and 
        // getDofTag() to access invalid array elements thereby causing bounds check exception
        {
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getDofOrdinal(3,0,0), nthrow, ncatch );
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getDofOrdinal(1,1,1), nthrow, ncatch );
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getDofOrdinal(0,4,1), nthrow, ncatch );
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getDofTag(8),         nthrow, ncatch );
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getDofTag(-1),        nthrow, ncatch );
        }

        // Exceptions 8-18 test exception handling with incorrectly dimensioned input/output arrays
        {
          // exception #8: input points array must be of rank-2
          DynRankView ConstructWithLabel(badPoints, 4, 5, 3);
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getValues(vals, badPoints, OPERATOR_VALUE), nthrow, ncatch );
        }
        {
          // exception #9 dimension 1 in the input point array must equal space dimension of the cell
          DynRankView ConstructWithLabel(badPoints, 4, 2);
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getValues(vals, badPoints, OPERATOR_VALUE), nthrow, ncatch );
        }
        {
          // exception #10 output values must be of rank-2 for OPERATOR_VALUE
          DynRankView ConstructWithLabel(badVals, 4, 3, 1);
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getValues(badVals, hexNodes, OPERATOR_VALUE), nthrow, ncatch );
        }
        {
          // exception #11 output values must be of rank-3 for OPERATOR_GRAD
          DynRankView ConstructWithLabel(badVals, 4, 3);
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getValues(badVals, hexNodes, OPERATOR_GRAD), nthrow, ncatch );

          // exception #12 output values must be of rank-3 for OPERATOR_D1
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getValues(badVals, hexNodes, OPERATOR_D1), nthrow, ncatch );

          // exception #13 output values must be of rank-3 for OPERATOR_D2
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getValues(badVals, hexNodes, OPERATOR_D2), nthrow, ncatch );
        }
        {
          // exception #14 incorrect 0th dimension of output array (must equal number of basis functions)
          DynRankView ConstructWithLabel(badVals, numFields + 1, numPoints);
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getValues(badVals, hexNodes, OPERATOR_VALUE), nthrow, ncatch );
        }
        {
          // exception #15 incorrect 1st dimension of output array (must equal number of points)
          DynRankView ConstructWithLabel(badVals, numFields, numPoints + 1);
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getValues(badVals, hexNodes, OPERATOR_VALUE), nthrow, ncatch );
        }
        {
          // exception #16: incorrect 2nd dimension of output array (must equal the space dimension)
          DynRankView ConstructWithLabel(badVals, numFields, numPoints, 4);
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getValues(badVals, hexNodes, OPERATOR_GRAD), nthrow, ncatch );
        }
        {
          // exception #17: incorrect 2nd dimension of output array (must equal D2 cardinality in 3D)
          DynRankView ConstructWithLabel(badVals, numFields, numPoints, 40);
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getValues(badVals, hexNodes, OPERATOR_D2), nthrow, ncatch );
        }
        {
          // exception #18: incorrect 2nd dimension of output array (must equal D3 cardinality in 3D)
          DynRankView ConstructWithLabel(badVals, numFields, numPoints, 50);
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getValues(badVals, hexNodes, OPERATOR_D3), nthrow, ncatch );
        }
#endif
      } catch (std::logic_error err) {
        *outStream << "UNEXPECTED ERROR !!! ----------------------------------------------------------\n";
        *outStream << err.what() << '\n';
        *outStream << "-------------------------------------------------------------------------------" << "\n\n";
        errorFlag = -1000;
      }
  
      // Check if number of thrown exceptions matches the one we expect 
      // Note Teuchos throw number will not pick up exceptions 3-7 and therefore will not match.
      if (nthrow != ncatch) {
        errorFlag++;
        *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
      }
      
      *outStream
        << "\n"
        << "===============================================================================\n"
        << "| TEST 2: correctness of tag to enum and enum to tag lookups                  |\n"
        << "===============================================================================\n";
      
      try{
        Basis_HGRAD_HEX_C1_FEM<DeviceSpaceType> hexBasis;

        const auto numFields = hexBasis.getCardinality();
        const auto allTags = hexBasis.getAllDofTags();

        // Loop over all tags, lookup the associated dof enumeration and then lookup the tag again
        const auto dofTagSize = allTags.dimension(0);
        for (auto i=0;i<dofTagSize;++i) {
          const auto bfOrd = hexBasis.getDofOrdinal(allTags(i,0), allTags(i,1), allTags(i,2));

          const auto myTag = hexBasis.getDofTag(bfOrd);
          if( !( (myTag(0) == allTags(i,0)) &&
                 (myTag(1) == allTags(i,1)) &&
                 (myTag(2) == allTags(i,2)) &&
                 (myTag(3) == allTags(i,3)) ) ) {
            errorFlag++;
            *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
            *outStream << " getDofOrdinal( {"
                       << allTags(i,0) << ", "
                       << allTags(i,1) << ", "
                       << allTags(i,2) << ", "
                       << allTags(i,3) << "}) = " << bfOrd <<" but \n";
            *outStream << " getDofTag(" << bfOrd << ") = { "
                       << myTag(0) << ", "
                       << myTag(1) << ", "
                       << myTag(2) << ", "
                       << myTag(3) << "}\n";
          }
        }
    
        // Now do the same but loop over basis functions
        for(auto bfOrd=0;bfOrd<numFields;++bfOrd) {
          const auto myTag  = hexBasis.getDofTag(bfOrd);
          const auto myBfOrd = hexBasis.getDofOrdinal(myTag(0), myTag(1), myTag(2));
          if( bfOrd != myBfOrd) {
            errorFlag++;
            *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
            *outStream << " getDofTag(" << bfOrd << ") = { "
                       << myTag(0) << ", "
                       << myTag(1) << ", "
                       << myTag(2) << ", "
                       << myTag(3) << "} but getDofOrdinal({"
                       << myTag(0) << ", "
                       << myTag(1) << ", "
                       << myTag(2) << ", "
                       << myTag(3) << "} ) = " << myBfOrd << "\n";
          }
        }
      } catch (std::logic_error err){
        *outStream << "UNEXPECTED ERROR !!! ----------------------------------------------------------\n";
        *outStream << err.what() << '\n';
        *outStream << "-------------------------------------------------------------------------------" << "\n\n";
        errorFlag = -1000; 
      };
  
      // *outStream \
      //   << "\n"
      //   << "===============================================================================\n"\
      //   << "| TEST 3: correctness of basis function values                                |\n"\
      //   << "===============================================================================\n";
  
      // outStream -> precision(20);
  
      // // VALUE: Each row gives the 8 correct basis set values at an evaluation point
      // double basisValues[] = {
      //   // bottom 4 vertices
      //   { 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0 },
      //   { 0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0 },
      //   { 0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 0.0 },
      //   { 0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 0.0 },
      //   // top 4 vertices
      //   { 0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0 },
      //   { 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0 },
      //   { 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0 },
      //   { 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0 },
      //   // center {0, 0, 0}
      //   0.125, 0.125, 0.125, 0.125,  0.125, 0.125, 0.125, 0.125,
      //   // faces { 1, 0, 0} and {-1, 0, 0}
      //   0.0,   0.25,  0.25,  0.0,    0.0,   0.25,  0.25,  0.0,
      //   0.25,  0.0,   0.0,   0.25,   0.25,  0.0,   0.0,   0.25,
      //   // faces { 0, 1, 0} and { 0,-1, 0}
      //   0.0,   0.0,   0.25,  0.25,   0.0,   0.0,   0.25,  0.25,
      //   0.25,  0.25,  0.0,   0.0,    0.25,  0.25,  0.0,   0.0,
      //   // faces {0, 0, 1} and {0, 0, -1}
      //   0.0,   0.0,   0.0,   0.0,    0.25,  0.25,  0.25,  0.25,
      //   0.25,  0.25,  0.25,  0.25,   0.0,   0.0,   0.0,   0.0,
      // };
  
      // // GRAD and D1: each row gives the 3x8 correct values of the gradients of the 8 basis functions
      // double basisGrads[] = {   
      //   // points 0-3
      //   -0.5,-0.5,-0.5,  0.5, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
      //   -0.5, 0.0, 0.0,  0.5,-0.5,-0.5,  0.0, 0.5, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
      //   0.0, 0.0, 0.0,  0.0,-0.5, 0.0,  0.5, 0.5,-0.5, -0.5, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.0, 0.0, 0.0,
      //   0.0,-0.5, 0.0,  0.0, 0.0, 0.0,  0.5, 0.0, 0.0, -0.5, 0.5,-0.5,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.5,
      //   // points 4-7
      //   0.0, 0.0,-0.5,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0, -0.5,-0.5, 0.5,  0.5, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.5, 0.0,
      //   0.0, 0.0, 0.0,  0.0, 0.0,-0.5,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0, -0.5, 0.0, 0.0,  0.5,-0.5, 0.5,  0.0, 0.5, 0.0,  0.0, 0.0, 0.0,
      //   0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,-0.5,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0,-0.5, 0.0,  0.5, 0.5, 0.5, -0.5, 0.0, 0.0,
      //   0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,-0.5,  0.0,-0.5, 0.0,  0.0, 0.0, 0.0,  0.5, 0.0, 0.0, -0.5, 0.5, 0.5,
      //   // point 8
      //   -0.125,-0.125,-0.125,  0.125,-0.125,-0.125,  0.125, 0.125,-0.125, \
      //   -0.125, 0.125,-0.125, -0.125,-0.125, 0.125,  0.125,-0.125, 0.125, \
      //   0.125, 0.125, 0.125, -0.125, 0.125, 0.125,
      //   // point 9
      //   -0.125, 0.0,   0.0,    0.125,-0.25, -0.25,   0.125, 0.25, -0.25,  -0.125, 0.0, 0.0, \
      //   -0.125, 0.0,   0.0,    0.125,-0.25,  0.25,   0.125, 0.25,  0.25,  -0.125, 0.0, 0.0,
      //   // point 10
      //   -0.125,-0.25, -0.25,   0.125, 0.0,   0.0,    0.125, 0.0,   0.0,   -0.125, 0.25, -0.25,\
      //   -0.125,-0.25,  0.25,   0.125, 0.0,   0.0,    0.125, 0.0,   0.0,   -0.125, 0.25,  0.25,
      //   // point 11
      //   0.0,  -0.125, 0.0,    0.0,  -0.125, 0.0,    0.25,  0.125,-0.25,  -0.25,  0.125,-0.25,\
      //   0.0,  -0.125, 0.0,    0.0,  -0.125, 0.0,    0.25,  0.125, 0.25,  -0.25,  0.125, 0.25,
      //   // point 12
      //   -0.25, -0.125,-0.25,   0.25, -0.125,-0.25,   0.0,   0.125, 0.0,    0.0,   0.125, 0.0, \
      //   -0.25, -0.125, 0.25,   0.25, -0.125, 0.25,   0.0,   0.125, 0.0,    0.0,   0.125, 0.0,
      //   // point 13
      //   0.0,   0.0,  -0.125,  0.0,   0.0,  -0.125,  0.0,   0.0,  -0.125,  0.0,   0.0,  -0.125, \
      //   -0.25, -0.25,  0.125,  0.25, -0.25,  0.125,  0.25,  0.25,  0.125, -0.25,  0.25,  0.125,
      //   // point 14
      //   -0.25, -0.25, -0.125,  0.25, -0.25, -0.125,  0.25,  0.25, -0.125, -0.25,  0.25, -0.125, \
      //   0.0,   0.0,   0.125,  0.0,   0.0,   0.125,  0.0,   0.0,   0.125,  0.0,   0.0,   0.125
      // };
  
      // //D2: flat array with the values of D2 applied to basis functions. Multi-index is (P,F,K)
      // double basisD2[] = {
      //   // point 0
      //   0, 0.25, 0.25, 0, 0.25, 0, 0, -0.25, -0.25, 0, 0., 0, 0, 0.25, 0., 0, \
      //   0., 0, 0, -0.25, 0., 0, -0.25, 0, 0, 0., -0.25, 0, -0.25, 0, 0, 0., \
      //   0.25, 0, 0., 0, 0, 0., 0., 0, 0., 0, 0, 0., 0., 0, 0.25, 0., \
      //   // point 1
      //   0, 0.25, 0.25, 0, 0., 0, 0, -0.25, -0.25, 0, 0.25, 0, 0, 0.25, 0., 0, \
      //   -0.25, 0, 0, -0.25, 0., 0, 0., 0, 0, 0., -0.25, 0, 0., 0, 0, 0., \
      //   0.25, 0, -0.25, 0, 0, 0., 0., 0, 0.25, 0, 0, 0., 0., 0, 0., 0., \
      //   // Point 2
      //   0, 0.25, 0., 0, 0., 0, 0, -0.25, 0., 0, 0.25, 0, 0, 0.25, -0.25, 0, \
      //   -0.25, 0, 0, -0.25, 0.25, 0, 0., 0, 0, 0., 0., 0, 0., 0, 0, 0., 0., \
      //   0, -0.25, 0, 0, 0., 0.25, 0, 0.25, 0, 0, 0., -0.25, 0, 0., 0., \
      //   // Point 3
      //   0, 0.25, 0., 0, 0.25, 0, 0, -0.25, 0., 0, 0., 0, 0, 0.25, -0.25, 0, \
      //   0., 0, 0, -0.25, 0.25, 0, -0.25, 0, 0, 0., 0., 0, -0.25, 0, 0, 0., \
      //   0., 0, 0., 0, 0, 0., 0.25, 0, 0., 0, 0, 0., -0.25, 0, 0.25, 0.,\
      //   // Point 4
      //   0, 0., 0.25, 0, 0.25, 0, 0, 0., -0.25, 0, 0., 0, 0, 0., 0., 0, 0., 0, \
      //   0, 0., 0., 0, -0.25, 0, 0, 0.25, -0.25, 0, -0.25, 0, 0, -0.25, 0.25, \
      //   0, 0., 0, 0, 0.25, 0., 0, 0., 0, 0, -0.25, 0., 0, 0.25, 0., \
      //   // Point 5
      //   0, 0., 0.25, 0, 0., 0, 0, 0., -0.25, 0, 0.25, 0, 0, 0., 0., 0, -0.25, \
      //   0, 0, 0., 0., 0, 0., 0, 0, 0.25, -0.25, 0, 0., 0, 0, -0.25, 0.25, 0, \
      //   -0.25, 0, 0, 0.25, 0., 0, 0.25, 0, 0, -0.25, 0., 0, 0., 0., \
      //   // Point 6
      //   0, 0., 0., 0, 0., 0, 0, 0., 0., 0, 0.25, 0, 0, 0., -0.25, 0, -0.25, \
      //   0, 0, 0., 0.25, 0, 0., 0, 0, 0.25, 0., 0, 0., 0, 0, -0.25, 0., 0, \
      //   -0.25, 0, 0, 0.25, 0.25, 0, 0.25, 0, 0, -0.25, -0.25, 0, 0., 0., \
      //   // Point 7
      //   0, 0., 0., 0, 0.25, 0, 0, 0., 0., 0, 0., 0, 0, 0., -0.25, 0, 0., 0, \
      //   0, 0., 0.25, 0, -0.25, 0, 0, 0.25, 0., 0, -0.25, 0, 0, -0.25, 0., 0, \
      //   0., 0, 0, 0.25, 0.25, 0, 0., 0, 0, -0.25, -0.25, 0, 0.25, 0., \
      //   // Point 8
      //   0, 0.125, 0.125, 0, 0.125, 0, 0, -0.125, -0.125, 0, 0.125, 0, 0, \
      //   0.125, -0.125, 0, -0.125, 0, 0, -0.125, 0.125, 0, -0.125, 0, 0, \
      //   0.125, -0.125, 0, -0.125, 0, 0, -0.125, 0.125, 0, -0.125, 0, 0, \
      //   0.125, 0.125, 0, 0.125, 0, 0, -0.125, -0.125, 0, 0.125, 0., \
      //   // Point 9
      //   0, 0.125, 0.125, 0, 0., 0, 0, -0.125, -0.125, 0, 0.25, 0, 0, 0.125, \
      //   -0.125, 0, -0.25, 0, 0, -0.125, 0.125, 0, 0., 0, 0, 0.125, -0.125, 0, \
      //   0., 0, 0, -0.125, 0.125, 0, -0.25, 0, 0, 0.125, 0.125, 0, 0.25, 0, 0, \
      //   -0.125, -0.125, 0, 0., 0., \
      //   // Point 10
      //   0, 0.125, 0.125, 0, 0.25, 0, 0, -0.125, -0.125, 0, 0., 0, 0, 0.125, \
      //   -0.125, 0, 0., 0, 0, -0.125, 0.125, 0, -0.25, 0, 0, 0.125, -0.125, 0, \
      //   -0.25, 0, 0, -0.125, 0.125, 0, 0., 0, 0, 0.125, 0.125, 0, 0., 0, 0, \
      //   -0.125, -0.125, 0, 0.25, 0., \
      //   // Point 11
      //   0, 0.125, 0., 0, 0.125, 0, 0, -0.125, 0., 0, 0.125, 0, 0, 0.125, \
      //   -0.25, 0, -0.125, 0, 0, -0.125, 0.25, 0, -0.125, 0, 0, 0.125, 0., 0, \
      //   -0.125, 0, 0, -0.125, 0., 0, -0.125, 0, 0, 0.125, 0.25, 0, 0.125, 0, \
      //   0, -0.125, -0.25, 0, 0.125, 0., \
      //   // Point 12
      //   0, 0.125, 0.25, 0, 0.125, 0, 0, -0.125, -0.25, 0, 0.125, 0, 0, 0.125, \
      //   0., 0, -0.125, 0, 0, -0.125, 0., 0, -0.125, 0, 0, 0.125, -0.25, 0, \
      //   -0.125, 0, 0, -0.125, 0.25, 0, -0.125, 0, 0, 0.125, 0., 0, 0.125, 0, \
      //   0, -0.125, 0., 0, 0.125, 0., \
      //   // Point 13
      //   0, 0., 0.125, 0, 0.125, 0, 0, 0., -0.125, 0, 0.125, 0, 0, 0., -0.125, \
      //   0, -0.125, 0, 0, 0., 0.125, 0, -0.125, 0, 0, 0.25, -0.125, 0, -0.125, \
      //   0, 0, -0.25, 0.125, 0, -0.125, 0, 0, 0.25, 0.125, 0, 0.125, 0, 0, \
      //   -0.25, -0.125, 0, 0.125, 0., \
      //   // Point 14
      //   0, 0.25, 0.125, 0, 0.125, 0, 0, -0.25, -0.125, 0, 0.125, 0, 0, 0.25, \
      //   -0.125, 0, -0.125, 0, 0, -0.25, 0.125, 0, -0.125, 0, 0, 0., -0.125, \
      //   0, -0.125, 0, 0, 0., 0.125, 0, -0.125, 0, 0, 0., 0.125, 0, 0.125, 0, \
      //   0, 0., -0.125, 0, 0.125, 0.
      // };
  
      // try{
        
      //   // Dimensions for the output arrays:
      //   int numFields = hexBasis.getCardinality();
      //   int numPoints = hexNodes.dimension(0);
      //   int spaceDim  = hexBasis.getBaseCellTopology().getDimension();
      //   int D2Cardin  = Intrepid2::getDkCardinality(OPERATOR_D2, spaceDim);
    
      //   // Generic array for values, grads, curls, etc. that will be properly sized before each call
      //   DynRankView ConstructWithLabel vals;
    
      //   // Check VALUE of basis functions: resize vals to rank-2 container:
      //   vals.resize(numFields, numPoints);
      //   hexBasis.getValues(vals, hexNodes, OPERATOR_VALUE);
      //   for (int i = 0; i < numFields; i++) {
      //     for (int j = 0; j < numPoints; j++) {
      //       int l =  i + j * numFields;
      //       if (std::abs(vals(i,j) - basisValues[l]) > INTREPID_TOL) {
      //         errorFlag++;
      //         *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";

      //         // Output the multi-index of the value where the error is:
      //         *outStream << " At multi-index { ";
      //         *outStream << i << " ";*outStream << j << " ";
      //         *outStream << "}  computed value: " << vals(i,j)
      //                    << " but reference value: " << basisValues[l] << "\n";
      //       }
      //     }
      //   }
    
      //   // Check GRAD of basis function: resize vals to rank-3 container
      //   vals.resize(numFields, numPoints, spaceDim);
      //   hexBasis.getValues(vals, hexNodes, OPERATOR_GRAD);
      //   for (int i = 0; i < numFields; i++) {
      //     for (int j = 0; j < numPoints; j++) {
      //       for (int k = 0; k < spaceDim; k++) {
      //         int l = k + i * spaceDim + j * spaceDim * numFields;
      //         if (std::abs(vals(i,j,k) - basisGrads[l]) > INTREPID_TOL) {
      //           errorFlag++;
      //           *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";

      //           // Output the multi-index of the value where the error is:
      //           *outStream << " At multi-index { ";
      //           *outStream << i << " ";*outStream << j << " ";*outStream << k << " ";
      //           *outStream << "}  computed grad component: " << vals(i,j,k)
      //                      << " but reference grad component: " << basisGrads[l] << "\n";
      //         }
      //       }
      //     }
      //   }
    
      //   // Check D1 of basis function (do not resize vals because it has the correct size: D1 = GRAD)
      //   hexBasis.getValues(vals, hexNodes, OPERATOR_D1);
      //   for (int i = 0; i < numFields; i++) {
      //     for (int j = 0; j < numPoints; j++) {
      //       for (int k = 0; k < spaceDim; k++) {
      //         int l = k + i * spaceDim + j * spaceDim * numFields;
      //         if (std::abs(vals(i,j,k) - basisGrads[l]) > INTREPID_TOL) {
      //           errorFlag++;
      //           *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";

      //           // Output the multi-index of the value where the error is:
      //           *outStream << " At multi-index { ";
      //           *outStream << i << " ";*outStream << j << " ";*outStream << k << " ";
      //           *outStream << "}  computed D1 component: " << vals(i,j,k)
      //                      << " but reference D1 component: " << basisGrads[l] << "\n";
      //         }
      //       }
      //     }
      //   }

    
      //   // Check D2 of basis function
      //   vals.resize(numFields, numPoints, D2Cardin);    
      //   hexBasis.getValues(vals, hexNodes, OPERATOR_D2);
      //   for (int i = 0; i < numFields; i++) {
      //     for (int j = 0; j < numPoints; j++) {
      //       for (int k = 0; k < D2Cardin; k++) {
      //         int l = k + i * D2Cardin + j * D2Cardin * numFields;
      //         if (std::abs(vals(i,j,k) - basisD2[l]) > INTREPID_TOL) {
      //           errorFlag++;
      //           *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";

      //           // Output the multi-index of the value where the error is:
      //           *outStream << " At multi-index { ";
      //           *outStream << i << " ";*outStream << j << " ";*outStream << k << " ";
      //           *outStream << "}  computed D2 component: " << vals(i,j,k)
      //                      << " but reference D2 component: " << basisD2[l] << "\n";
      //         }
      //       }
      //     }
      //   }

      //   // Check all higher derivatives - must be zero. 
      //   for(EOperator op = OPERATOR_D3; op < OPERATOR_MAX; op++) {
      
      //     // The last dimension is the number of kth derivatives and needs to be resized for every Dk
      //     int DkCardin  = Intrepid2::getDkCardinality(op, spaceDim);
      //     vals.resize(numFields, numPoints, DkCardin);    

      //     hexBasis.getValues(vals, hexNodes, op);
      //     for (int i1 = 0; i1 < numFields; i1++) 
      //       for (int i2 = 0; i2 < numPoints; i2++) 
      //         for (int i3 = 0; i3 < DkCardin; i3++) {
      //           if (std::abs(vals(i1,i2,i3)) > INTREPID_TOL) {
      //             errorFlag++;
      //             *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
          
      //             // Get the multi-index of the value where the error is and the operator order
      //             int ord = Intrepid2::getOperatorOrder(op);
      //             *outStream << " At multi-index { "<<i1<<" "<<i2 <<" "<<i3;
      //             *outStream << "}  computed D"<< ord <<" component: " << vals(i1,i2,i3) 
      //                        << " but reference D" << ord << " component:  0 \n";
      //           }
      //         }
      //   }    
      // }
  
      // // Catch unexpected errors
      // catch (std::logic_error err) {
      //   *outStream << err.what() << "\n\n";
      //   errorFlag = -1000;
      // };

      *outStream 
        << "\n"
        << "===============================================================================\n"
        << "| TEST 4: correctness of DoF locations                                        |\n"
        << "===============================================================================\n";

      try{
        Basis_HGRAD_HEX_C1_FEM<DeviceSpaceType> hexBasis;
        const auto numFields = hexBasis.getCardinality();
        const auto spaceDim  = hexBasis.getBaseCellTopology().getDimension();

        // Check exceptions.
#ifdef HAVE_INTREPID2_DEBUG
        {       
          DynRankView ConstructWithLabel(badVals, 1,2,3);
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getDofCoords(badVals), nthrow, ncatch );
        }
        {
          DynRankView ConstructWithLabel(badVals, 3,2);
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getDofCoords(badVals), nthrow, ncatch );
        }
        {
          DynRankView ConstructWithLabel(badVals, 8,2);
          INTREPID2_TEST_ERROR_EXPECTED( hexBasis.getDofCoords(badVals), nthrow, ncatch );
        }
#endif
        if (nthrow != ncatch) {
          *outStream << "UNEXPECTED ERROR !!! ----------------------------------------------------------\n";
          ++errorFlag;
        }

        DynRankView ConstructWithLabel(bvals, numFields, numFields);
        DynRankView ConstructWithLabel(cvals, numFields, spaceDim);

        // Check mathematical correctness.
        hexBasis.getDofCoords(cvals);
        hexBasis.getValues(bvals, cvals, OPERATOR_VALUE);
        for (auto i=0;i<numFields;++i) {
          for (auto j=0;j<numFields;++j) {
            if (i != j && (std::abs(bvals(i,j) - 0.0) > tol)) {
              errorFlag++;
              std::stringstream ss;
              ss << "\n Value of basis function " << i << " at (" << cvals(i,0) << ", " << cvals(i,1) << ") is " << bvals(i,j) << " but should be 0.0\n";
              *outStream << ss.str();
            }
            else if ((i == j) && (std::abs(bvals(i,j) - 1.0) > INTREPID_TOL)) {
              errorFlag++;
              std::stringstream ss;
              ss << "\n Value of basis function " << i << " at (" << cvals(i,0) << ", " << cvals(i,1) << ") is " << bvals(i,j) << " but should be 1.0\n";
              *outStream << ss.str();
            }
          }
        }

      } catch (std::logic_error err){
        *outStream << "UNEXPECTED ERROR !!! ----------------------------------------------------------\n";
        *outStream << err.what() << '\n';
        *outStream << "-------------------------------------------------------------------------------" << "\n\n";
        errorFlag = -1000;
      };
  
      if (errorFlag != 0)
        std::cout << "End Result: TEST FAILED\n";
      else
        std::cout << "End Result: TEST PASSED\n";
  
      // reset format state of std::cout
      std::cout.copyfmt(oldFormatState);

      return errorFlag;
    }
  } // end of test 
} // end of intrepid2
