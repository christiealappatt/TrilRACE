// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef MUELU_PERFMODELS_HPP
#define MUELU_PERFMODELS_HPP

#include <vector>
#include <Teuchos_DefaultComm.hpp>


namespace MueLu {

  template <class Scalar,
            class LocalOrdinal = DefaultLocalOrdinal,
            class GlobalOrdinal = DefaultGlobalOrdinal,
            class Node = DefaultNode>
  class PerfModels {
  public:
    /* Single Node tests based upon the STREAM benchmark for measuring memory
     * bandwith and computation rate. These processes compute either the addition
     * of two vectors or the multiplication of dense matrices of any given size.
     * Many iterations occur which then return a vector containing the individual
     * lengths of time per iteration.
     *
     * See further here:
     *    - https://www.cs.virginia.edu/stream/ref.html
     *    - https://github.com/UoB-HPC/BabelStream
     */
    double stream_vector_add_SC(int KERNEL_REPEATS, int VECTOR_SIZE);
    double stream_vector_add_LO(int KERNEL_REPEATS, int VECTOR_SIZE);
    double stream_vector_add_size_t(int KERNEL_REPEATS, int VECTOR_SIZE);

    std::vector<double> stream_vector_add_SC_all(int KERNEL_REPEATS, int VECTOR_SIZE);
    std::vector<double> stream_vector_add_LO_all(int KERNEL_REPEATS, int VECTOR_SIZE);
    std::vector<double> stream_vector_add_size_t_all(int KERNEL_REPEATS, int VECTOR_SIZE);

    /* A latency test between two processes based upon the MVAPICH OSU Micro-Benchmarks.
     * The sender process sends a message and then waits for confirmation of reception.
     * Many iterations occur with various message sizes and the average latency values
     * are returned within a map. Utilizes blocking send and recieve.
     *
     * See further: https://mvapich.cse.ohio-state.edu/benchmarks/
     */
    std::map<int,double> pingpong_test_host(int KERNEL_REPEATS, int MAX_SIZE, const RCP<const Teuchos::Comm<int> > &comm);
    std::map<int,double> pingpong_test_device(int KERNEL_REPEATS, int MAX_SIZE, const RCP<const Teuchos::Comm<int> > &comm);
    //  private:
    


  }; //class PerfModels

} //namespace MueLu

#endif //ifndef MUELU_PERFMODELS_HPP
