// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
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
// ************************************************************************
// @HEADER

#ifndef TPETRA_DETAILS_DISTRIBUTOR_ACTOR_HPP
#define TPETRA_DETAILS_DISTRIBUTOR_ACTOR_HPP

#include "Tpetra_Details_DistributorPlan.hpp"
#include "Tpetra_Util.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Time.hpp"

namespace Tpetra {
namespace Details {

template <class View1, class View2>
constexpr bool areKokkosViews = Kokkos::Impl::is_view<View1>::value && Kokkos::Impl::is_view<View2>::value;

class DistributorActor {

public:
  DistributorActor();
  DistributorActor(const DistributorActor& otherActor);

  template <class ExpView, class ImpView>
  typename std::enable_if_t<areKokkosViews<ExpView, ImpView>>
  doPosts(const DistributorPlan& plan,
          const ExpView& exports,
          size_t numPackets,
          const ImpView& imports);

  template <class ExpView, class ImpView>
  typename std::enable_if_t<areKokkosViews<ExpView, ImpView>>
  doPosts(const DistributorPlan& plan,
          const ExpView &exports,
          const Teuchos::ArrayView<const size_t>& numExportPacketsPerLID,
          const ImpView &imports,
          const Teuchos::ArrayView<const size_t>& numImportPacketsPerLID);

  Teuchos::Array<Teuchos::RCP<Teuchos::CommRequest<int>>> requests_;

#ifdef HAVE_TPETRA_DISTRIBUTOR_TIMINGS
  Teuchos::RCP<Teuchos::Time> timer_doPosts3TA_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts4TA_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts3KV_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts4KV_;
  Teuchos::RCP<Teuchos::Time> timer_doWaits_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts3TA_recvs_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts4TA_recvs_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts3TA_barrier_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts4TA_barrier_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts3TA_sends_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts4TA_sends_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts3TA_sends_slow_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts4TA_sends_slow_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts3TA_sends_fast_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts4TA_sends_fast_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts3KV_recvs_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts4KV_recvs_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts3KV_barrier_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts4KV_barrier_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts3KV_sends_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts4KV_sends_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts3KV_sends_slow_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts4KV_sends_slow_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts3KV_sends_fast_;
  Teuchos::RCP<Teuchos::Time> timer_doPosts4KV_sends_fast_;

  //! Make the instance's timers.  (Call only in constructor.)
  void makeTimers();
#endif // HAVE_TPETRA_DISTRIBUTOR_TIMINGS
};

template <class ExpView, class ImpView>
typename std::enable_if_t<areKokkosViews<ExpView, ImpView>>
DistributorActor::doPosts(const DistributorPlan& plan,
                          const ExpView& exports,
                          size_t numPackets,
                          const ImpView& imports)
{
  using Teuchos::Array;
  using Teuchos::as;
  using Teuchos::FancyOStream;
  using Teuchos::includesVerbLevel;
  using Teuchos::ireceive;
  using Teuchos::isend;
  using Teuchos::readySend;
  using Teuchos::send;
  using Teuchos::ssend;
  using Teuchos::TypeNameTraits;
  using Teuchos::typeName;
  using std::endl;
  using Kokkos::Compat::create_const_view;
  using Kokkos::Compat::create_view;
  using Kokkos::Compat::subview_offset;
  using Kokkos::Compat::deep_copy_offset;
  typedef Array<size_t>::size_type size_type;
  typedef ExpView exports_view_type;
  typedef ImpView imports_view_type;

#ifdef KOKKOS_ENABLE_CUDA
  static_assert
    (! std::is_same<typename ExpView::memory_space, Kokkos::CudaUVMSpace>::value &&
     ! std::is_same<typename ImpView::memory_space, Kokkos::CudaUVMSpace>::value,
     "Please do not use Tpetra::Distributor with UVM allocations.  "
     "See Trilinos GitHub issue #1088.");
#endif // KOKKOS_ENABLE_CUDA

#ifdef HAVE_TPETRA_DISTRIBUTOR_TIMINGS
  Teuchos::TimeMonitor timeMon (*timer_doPosts3KV_);
#endif // HAVE_TPETRA_DISTRIBUTOR_TIMINGS

  const int myRank = plan.comm_->getRank ();
  // Run-time configurable parameters that come from the input
  // ParameterList set by setParameterList().
  const Details::EDistributorSendType sendType = plan.sendType_;
  const bool doBarrier = plan.barrierBetweenRecvSend_;

  TEUCHOS_TEST_FOR_EXCEPTION(
      sendType == Details::DISTRIBUTOR_RSEND && ! doBarrier,
      std::logic_error,
      "Tpetra::Distributor::doPosts(3 args, Kokkos): Ready-send version "
      "requires a barrier between posting receives and posting ready sends.  "
      "This should have been checked before.  "
      "Please report this bug to the Tpetra developers.");

  size_t selfReceiveOffset = 0;

  // MPI tag for nonblocking receives and blocking sends in this
  // method.  Some processes might take the "fast" path
  // (indicesTo_.empty()) and others might take the "slow" path for
  // the same doPosts() call, so the path tag must be the same for
  // both.
  const int pathTag = 0;
  const int tag = plan.getTag(pathTag);

#ifdef HAVE_TPETRA_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION
    (requests_.size () != 0,
     std::logic_error,
     "Tpetra::Distributor::doPosts(3 args, Kokkos): Process "
     << myRank << ": requests_.size() = " << requests_.size () << " != 0.");
#endif // HAVE_TPETRA_DEBUG

  // Distributor uses requests_.size() as the number of outstanding
  // nonblocking message requests, so we resize to zero to maintain
  // this invariant.
  //
  // numReceives_ does _not_ include the self message, if there is
  // one.  Here, we do actually send a message to ourselves, so we
  // include any self message in the "actual" number of receives to
  // post.
  //
  // NOTE (mfh 19 Mar 2012): Epetra_MpiDistributor::DoPosts()
  // doesn't (re)allocate its array of requests.  That happens in
  // CreateFromSends(), ComputeRecvs_(), DoReversePosts() (on
  // demand), or Resize_().
  const size_type actualNumReceives = as<size_type> (plan.numReceives_) +
    as<size_type> (plan.sendMessageToSelf_ ? 1 : 0);
  requests_.resize (0);

  // Post the nonblocking receives.  It's common MPI wisdom to post
  // receives before sends.  In MPI terms, this means favoring
  // adding to the "posted queue" (of receive requests) over adding
  // to the "unexpected queue" (of arrived messages not yet matched
  // with a receive).
  {
#ifdef HAVE_TPETRA_DISTRIBUTOR_TIMINGS
    Teuchos::TimeMonitor timeMonRecvs (*timer_doPosts3KV_recvs_);
#endif // HAVE_TPETRA_DISTRIBUTOR_TIMINGS

    size_t curBufferOffset = 0;
    for (size_type i = 0; i < actualNumReceives; ++i) {
      const size_t curBufLen = plan.lengthsFrom_[i] * numPackets;
      if (plan.procsFrom_[i] != myRank) {
        // If my process is receiving these packet(s) from another
        // process (not a self-receive):
        //
        // 1. Set up the persisting view (recvBuf) of the imports
        //    array, given the offset and size (total number of
        //    packets from process procsFrom_[i]).
        // 2. Start the Irecv and save the resulting request.
        TEUCHOS_TEST_FOR_EXCEPTION(
            curBufferOffset + curBufLen > static_cast<size_t> (imports.size ()),
            std::logic_error, "Tpetra::Distributor::doPosts(3 args, Kokkos): "
            "Exceeded size of 'imports' array in packing loop on Process " <<
            myRank << ".  imports.size() = " << imports.size () << " < "
            "curBufferOffset(" << curBufferOffset << ") + curBufLen(" <<
            curBufLen << ").");
        imports_view_type recvBuf =
          subview_offset (imports, curBufferOffset, curBufLen);
        requests_.push_back (ireceive<int> (recvBuf, plan.procsFrom_[i],
              tag, *plan.comm_));
      }
      else { // Receiving from myself
        selfReceiveOffset = curBufferOffset; // Remember the self-recv offset
      }
      curBufferOffset += curBufLen;
    }
  }

  if (doBarrier) {
#ifdef HAVE_TPETRA_DISTRIBUTOR_TIMINGS
    Teuchos::TimeMonitor timeMonBarrier (*timer_doPosts3KV_barrier_);
#endif // HAVE_TPETRA_DISTRIBUTOR_TIMINGS

    // If we are using ready sends (MPI_Rsend) below, we need to do
    // a barrier before we post the ready sends.  This is because a
    // ready send requires that its matching receive has already
    // been posted before the send has been posted.  The only way to
    // guarantee that in this case is to use a barrier.
    plan.comm_->barrier ();
  }

#ifdef HAVE_TPETRA_DISTRIBUTOR_TIMINGS
  Teuchos::TimeMonitor timeMonSends (*timer_doPosts3KV_sends_);
#endif // HAVE_TPETRA_DISTRIBUTOR_TIMINGS

  // setup scan through procIdsToSendTo_ list starting with higher numbered procs
  // (should help balance message traffic)
  //
  // FIXME (mfh 20 Feb 2013) Why haven't we precomputed this?
  // It doesn't depend on the input at all.
  size_t numBlocks = plan.numSendsToOtherProcs_ + plan.sendMessageToSelf_;
  size_t procIndex = 0;
  while ((procIndex < numBlocks) && (plan.procIdsToSendTo_[procIndex] < myRank)) {
    ++procIndex;
  }
  if (procIndex == numBlocks) {
    procIndex = 0;
  }

  size_t selfNum = 0;
  size_t selfIndex = 0;

  if (plan.indicesTo_.empty()) {

#ifdef HAVE_TPETRA_DISTRIBUTOR_TIMINGS
    Teuchos::TimeMonitor timeMonSends2 (*timer_doPosts3KV_sends_fast_);
#endif // HAVE_TPETRA_DISTRIBUTOR_TIMINGS

    // Data are already blocked (laid out) by process, so we don't
    // need a separate send buffer (besides the exports array).
    for (size_t i = 0; i < numBlocks; ++i) {
      size_t p = i + procIndex;
      if (p > (numBlocks - 1)) {
        p -= numBlocks;
      }

      if (plan.procIdsToSendTo_[p] != myRank) {
        exports_view_type tmpSend = subview_offset(
            exports, plan.startsTo_[p]*numPackets, plan.lengthsTo_[p]*numPackets);

        if (sendType == Details::DISTRIBUTOR_SEND) {
          send<int> (tmpSend,
              as<int> (tmpSend.size ()),
              plan.procIdsToSendTo_[p], tag, *plan.comm_);
        }
        else if (sendType == Details::DISTRIBUTOR_ISEND) {
          exports_view_type tmpSendBuf =
            subview_offset (exports, plan.startsTo_[p] * numPackets,
                plan.lengthsTo_[p] * numPackets);
          requests_.push_back (isend<int> (tmpSendBuf, plan.procIdsToSendTo_[p],
                tag, *plan.comm_));
        }
        else if (sendType == Details::DISTRIBUTOR_RSEND) {
          readySend<int> (tmpSend,
              as<int> (tmpSend.size ()),
              plan.procIdsToSendTo_[p], tag, *plan.comm_);
        }
        else if (sendType == Details::DISTRIBUTOR_SSEND) {
          ssend<int> (tmpSend,
              as<int> (tmpSend.size ()),
              plan.procIdsToSendTo_[p], tag, *plan.comm_);
        } else {
          TEUCHOS_TEST_FOR_EXCEPTION(
              true,
              std::logic_error,
              "Tpetra::Distributor::doPosts(3 args, Kokkos): "
              "Invalid send type.  We should never get here.  "
              "Please report this bug to the Tpetra developers.");
        }
      }
      else { // "Sending" the message to myself
        selfNum = p;
      }
    }

    if (plan.sendMessageToSelf_) {
      // This is how we "send a message to ourself": we copy from
      // the export buffer to the import buffer.  That saves
      // Teuchos::Comm implementations other than MpiComm (in
      // particular, SerialComm) the trouble of implementing self
      // messages correctly.  (To do this right, SerialComm would
      // need internal buffer space for messages, keyed on the
      // message's tag.)
      deep_copy_offset(imports, exports, selfReceiveOffset,
          plan.startsTo_[selfNum]*numPackets,
          plan.lengthsTo_[selfNum]*numPackets);
    }
  }
  else { // data are not blocked by proc, use send buffer

#ifdef HAVE_TPETRA_DISTRIBUTOR_TIMINGS
    Teuchos::TimeMonitor timeMonSends2 (*timer_doPosts3KV_sends_slow_);
#endif // HAVE_TPETRA_DISTRIBUTOR_TIMINGS

    typedef typename ExpView::non_const_value_type Packet;
    typedef typename ExpView::array_layout Layout;
    typedef typename ExpView::device_type Device;
    typedef typename ExpView::memory_traits Mem;
    Kokkos::View<Packet*,Layout,Device,Mem> sendArray ("sendArray",
        plan.maxSendLength_ * numPackets);

    // FIXME (mfh 05 Mar 2013) This is broken for Isend (nonblocking
    // sends), because the buffer is only long enough for one send.
    TEUCHOS_TEST_FOR_EXCEPTION(
        sendType == Details::DISTRIBUTOR_ISEND,
        std::logic_error,
        "Tpetra::Distributor::doPosts(3 args, Kokkos): The \"send buffer\" code path "
        "doesn't currently work with nonblocking sends.");

    for (size_t i = 0; i < numBlocks; ++i) {
      size_t p = i + procIndex;
      if (p > (numBlocks - 1)) {
        p -= numBlocks;
      }

      if (plan.procIdsToSendTo_[p] != myRank) {
        size_t sendArrayOffset = 0;
        size_t j = plan.startsTo_[p];
        for (size_t k = 0; k < plan.lengthsTo_[p]; ++k, ++j) {
          deep_copy_offset(sendArray, exports, sendArrayOffset,
              plan.indicesTo_[j]*numPackets, numPackets);
          sendArrayOffset += numPackets;
        }
        ImpView tmpSend =
          subview_offset(sendArray, size_t(0), plan.lengthsTo_[p]*numPackets);

        if (sendType == Details::DISTRIBUTOR_SEND) {
          send<int> (tmpSend,
              as<int> (tmpSend.size ()),
              plan.procIdsToSendTo_[p], tag, *plan.comm_);
        }
        else if (sendType == Details::DISTRIBUTOR_ISEND) {
          exports_view_type tmpSendBuf =
            subview_offset (sendArray, size_t(0), plan.lengthsTo_[p] * numPackets);
          requests_.push_back (isend<int> (tmpSendBuf, plan.procIdsToSendTo_[p],
                tag, *plan.comm_));
        }
        else if (sendType == Details::DISTRIBUTOR_RSEND) {
          readySend<int> (tmpSend,
              as<int> (tmpSend.size ()),
              plan.procIdsToSendTo_[p], tag, *plan.comm_);
        }
        else if (sendType == Details::DISTRIBUTOR_SSEND) {
          ssend<int> (tmpSend,
              as<int> (tmpSend.size ()),
              plan.procIdsToSendTo_[p], tag, *plan.comm_);
        }
        else {
          TEUCHOS_TEST_FOR_EXCEPTION(
              true,
              std::logic_error,
              "Tpetra::Distributor::doPosts(3 args, Kokkos): "
              "Invalid send type.  We should never get here.  "
              "Please report this bug to the Tpetra developers.");
        }
      }
      else { // "Sending" the message to myself
        selfNum = p;
        selfIndex = plan.startsTo_[p];
      }
    }

    if (plan.sendMessageToSelf_) {
      for (size_t k = 0; k < plan.lengthsTo_[selfNum]; ++k) {
        deep_copy_offset(imports, exports, selfReceiveOffset,
            plan.indicesTo_[selfIndex]*numPackets, numPackets);
        ++selfIndex;
        selfReceiveOffset += numPackets;
      }
    }
  }
}

template <class ExpView, class ImpView>
typename std::enable_if_t<areKokkosViews<ExpView, ImpView>>
DistributorActor::
doPosts(const DistributorPlan& plan,
        const ExpView &exports,
        const Teuchos::ArrayView<const size_t>& numExportPacketsPerLID,
        const ImpView &imports,
        const Teuchos::ArrayView<const size_t>& numImportPacketsPerLID)
{
  using Teuchos::Array;
  using Teuchos::as;
  using Teuchos::ireceive;
  using Teuchos::isend;
  using Teuchos::readySend;
  using Teuchos::send;
  using Teuchos::ssend;
  using Teuchos::TypeNameTraits;
  using std::endl;
  using Kokkos::Compat::create_const_view;
  using Kokkos::Compat::create_view;
  using Kokkos::Compat::subview_offset;
  using Kokkos::Compat::deep_copy_offset;
  typedef Array<size_t>::size_type size_type;
  typedef ExpView exports_view_type;
  typedef ImpView imports_view_type;

#ifdef KOKKOS_ENABLE_CUDA
  static_assert (! std::is_same<typename ExpView::memory_space, Kokkos::CudaUVMSpace>::value &&
      ! std::is_same<typename ImpView::memory_space, Kokkos::CudaUVMSpace>::value,
      "Please do not use Tpetra::Distributor with UVM "
      "allocations.  See GitHub issue #1088.");
#endif // KOKKOS_ENABLE_CUDA

#ifdef HAVE_TPETRA_DISTRIBUTOR_TIMINGS
  Teuchos::TimeMonitor timeMon (*timer_doPosts4KV_);
#endif // HAVE_TPETRA_DISTRIBUTOR_TIMINGS

  // Run-time configurable parameters that come from the input
  // ParameterList set by setParameterList().
  const Details::EDistributorSendType sendType = plan.sendType_;
  const bool doBarrier = plan.barrierBetweenRecvSend_;

  TEUCHOS_TEST_FOR_EXCEPTION(
      sendType == Details::DISTRIBUTOR_RSEND && ! doBarrier,
      std::logic_error, "Tpetra::Distributor::doPosts(4 args, Kokkos): Ready-send "
      "version requires a barrier between posting receives and posting ready "
      "sends.  This should have been checked before.  "
      "Please report this bug to the Tpetra developers.");

  const int myProcID = plan.comm_->getRank ();
  size_t selfReceiveOffset = 0;

#ifdef HAVE_TEUCHOS_DEBUG
  // Different messages may have different numbers of packets.
  size_t totalNumImportPackets = 0;
  for (size_type ii = 0; ii < numImportPacketsPerLID.size (); ++ii) {
    totalNumImportPackets += numImportPacketsPerLID[ii];
  }
  TEUCHOS_TEST_FOR_EXCEPTION(
      imports.extent (0) < totalNumImportPackets, std::runtime_error,
      "Tpetra::Distributor::doPosts(4 args, Kokkos): The 'imports' array must have "
      "enough entries to hold the expected number of import packets.  "
      "imports.extent(0) = " << imports.extent (0) << " < "
      "totalNumImportPackets = " << totalNumImportPackets << ".");
#endif // HAVE_TEUCHOS_DEBUG

  // MPI tag for nonblocking receives and blocking sends in this
  // method.  Some processes might take the "fast" path
  // (plan.indicesTo_.empty()) and others might take the "slow" path for
  // the same doPosts() call, so the path tag must be the same for
  // both.
  const int pathTag = 1;
  const int tag = plan.getTag(pathTag);

#ifdef HAVE_TEUCHOS_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION
    (requests_.size () != 0, std::logic_error, "Tpetra::Distributor::"
     "doPosts(4 args, Kokkos): Process " << myProcID << ": requests_.size () = "
     << requests_.size () << " != 0.");
#endif // HAVE_TEUCHOS_DEBUG
  // Distributor uses requests_.size() as the number of outstanding
  // nonblocking message requests, so we resize to zero to maintain
  // this invariant.
  //
  // numReceives_ does _not_ include the self message, if there is
  // one.  Here, we do actually send a message to ourselves, so we
  // include any self message in the "actual" number of receives to
  // post.
  //
  // NOTE (mfh 19 Mar 2012): Epetra_MpiDistributor::DoPosts()
  // doesn't (re)allocate its array of requests.  That happens in
  // CreateFromSends(), ComputeRecvs_(), DoReversePosts() (on
  // demand), or Resize_().
  const size_type actualNumReceives = as<size_type> (plan.numReceives_) +
    as<size_type> (plan.sendMessageToSelf_ ? 1 : 0);
  requests_.resize (0);

  // Post the nonblocking receives.  It's common MPI wisdom to post
  // receives before sends.  In MPI terms, this means favoring
  // adding to the "posted queue" (of receive requests) over adding
  // to the "unexpected queue" (of arrived messages not yet matched
  // with a receive).
  {
#ifdef HAVE_TPETRA_DISTRIBUTOR_TIMINGS
    Teuchos::TimeMonitor timeMonRecvs (*timer_doPosts4KV_recvs_);
#endif // HAVE_TPETRA_DISTRIBUTOR_TIMINGS

    size_t curBufferOffset = 0;
    size_t curLIDoffset = 0;
    for (size_type i = 0; i < actualNumReceives; ++i) {
      size_t totalPacketsFrom_i = 0;
      for (size_t j = 0; j < plan.lengthsFrom_[i]; ++j) {
        totalPacketsFrom_i += numImportPacketsPerLID[curLIDoffset+j];
      }
      curLIDoffset += plan.lengthsFrom_[i];
      if (plan.procsFrom_[i] != myProcID && totalPacketsFrom_i) {
        // If my process is receiving these packet(s) from another
        // process (not a self-receive), and if there is at least
        // one packet to receive:
        //
        // 1. Set up the persisting view (recvBuf) into the imports
        //    array, given the offset and size (total number of
        //    packets from process procsFrom_[i]).
        // 2. Start the Irecv and save the resulting request.
        imports_view_type recvBuf =
          subview_offset (imports, curBufferOffset, totalPacketsFrom_i);
        requests_.push_back (ireceive<int> (recvBuf, plan.procsFrom_[i],
              tag, *plan.comm_));
      }
      else { // Receiving these packet(s) from myself
        selfReceiveOffset = curBufferOffset; // Remember the offset
      }
      curBufferOffset += totalPacketsFrom_i;
    }
  }

  if (doBarrier) {
#ifdef HAVE_TPETRA_DISTRIBUTOR_TIMINGS
    Teuchos::TimeMonitor timeMonBarrier (*timer_doPosts4KV_barrier_);
#endif // HAVE_TPETRA_DISTRIBUTOR_TIMINGS
    // If we are using ready sends (MPI_Rsend) below, we need to do
    // a barrier before we post the ready sends.  This is because a
    // ready send requires that its matching receive has already
    // been posted before the send has been posted.  The only way to
    // guarantee that in this case is to use a barrier.
    plan.comm_->barrier ();
  }

#ifdef HAVE_TPETRA_DISTRIBUTOR_TIMINGS
  Teuchos::TimeMonitor timeMonSends (*timer_doPosts4KV_sends_);
#endif // HAVE_TPETRA_DISTRIBUTOR_TIMINGS

  // setup arrays containing starting-offsets into exports for each send,
  // and num-packets-to-send for each send.
  Array<size_t> sendPacketOffsets(plan.numSendsToOtherProcs_,0), packetsPerSend(plan.numSendsToOtherProcs_,0);
  size_t maxNumPackets = 0;
  size_t curPKToffset = 0;
  for (size_t pp=0; pp<plan.numSendsToOtherProcs_; ++pp) {
    sendPacketOffsets[pp] = curPKToffset;
    size_t numPackets = 0;
    for (size_t j=plan.startsTo_[pp]; j<plan.startsTo_[pp]+plan.lengthsTo_[pp]; ++j) {
      numPackets += numExportPacketsPerLID[j];
    }
    if (numPackets > maxNumPackets) maxNumPackets = numPackets;
    packetsPerSend[pp] = numPackets;
    curPKToffset += numPackets;
  }

  // setup scan through procIdsToSendTo_ list starting with higher numbered procs
  // (should help balance message traffic)
  size_t numBlocks = plan.numSendsToOtherProcs_ + plan.sendMessageToSelf_;
  size_t procIndex = 0;
  while ((procIndex < numBlocks) && (plan.procIdsToSendTo_[procIndex] < myProcID)) {
    ++procIndex;
  }
  if (procIndex == numBlocks) {
    procIndex = 0;
  }

  size_t selfNum = 0;
  size_t selfIndex = 0;
  if (plan.indicesTo_.empty()) {

#ifdef HAVE_TPETRA_DISTRIBUTOR_TIMINGS
    Teuchos::TimeMonitor timeMonSends2 (*timer_doPosts4KV_sends_fast_);
#endif // HAVE_TPETRA_DISTRIBUTOR_TIMINGS

    // Data are already blocked (laid out) by process, so we don't
    // need a separate send buffer (besides the exports array).
    for (size_t i = 0; i < numBlocks; ++i) {
      size_t p = i + procIndex;
      if (p > (numBlocks - 1)) {
        p -= numBlocks;
      }

      if (plan.procIdsToSendTo_[p] != myProcID && packetsPerSend[p] > 0) {
        exports_view_type tmpSend =
          subview_offset(exports, sendPacketOffsets[p], packetsPerSend[p]);

        if (sendType == Details::DISTRIBUTOR_SEND) { // the default, so put it first
          send<int> (tmpSend,
              as<int> (tmpSend.size ()),
              plan.procIdsToSendTo_[p], tag, *plan.comm_);
        }
        else if (sendType == Details::DISTRIBUTOR_RSEND) {
          readySend<int> (tmpSend,
              as<int> (tmpSend.size ()),
              plan.procIdsToSendTo_[p], tag, *plan.comm_);
        }
        else if (sendType == Details::DISTRIBUTOR_ISEND) {
          exports_view_type tmpSendBuf =
            subview_offset (exports, sendPacketOffsets[p], packetsPerSend[p]);
          requests_.push_back (isend<int> (tmpSendBuf, plan.procIdsToSendTo_[p],
                tag, *plan.comm_));
        }
        else if (sendType == Details::DISTRIBUTOR_SSEND) {
          ssend<int> (tmpSend,
              as<int> (tmpSend.size ()),
              plan.procIdsToSendTo_[p], tag, *plan.comm_);
        }
        else {
          TEUCHOS_TEST_FOR_EXCEPTION(
              true, std::logic_error,
              "Tpetra::Distributor::doPosts(4 args, Kokkos): "
              "Invalid send type.  We should never get here.  "
              "Please report this bug to the Tpetra developers.");
        }
      }
      else { // "Sending" the message to myself
        selfNum = p;
      }
    }

    if (plan.sendMessageToSelf_) {
      deep_copy_offset(imports, exports, selfReceiveOffset,
          sendPacketOffsets[selfNum], packetsPerSend[selfNum]);
    }
  }
  else { // data are not blocked by proc, use send buffer

#ifdef HAVE_TPETRA_DISTRIBUTOR_TIMINGS
    Teuchos::TimeMonitor timeMonSends2 (*timer_doPosts4KV_sends_slow_);
#endif // HAVE_TPETRA_DISTRIBUTOR_TIMINGS

    // FIXME (mfh 05 Mar 2013) This may be broken for Isend.
    typedef typename ExpView::non_const_value_type Packet;
    typedef typename ExpView::array_layout Layout;
    typedef typename ExpView::device_type Device;
    typedef typename ExpView::memory_traits Mem;
    Kokkos::View<Packet*,Layout,Device,Mem> sendArray ("sendArray", maxNumPackets); // send buffer

    TEUCHOS_TEST_FOR_EXCEPTION(
        sendType == Details::DISTRIBUTOR_ISEND,
        std::logic_error,
        "Tpetra::Distributor::doPosts(4-arg, Kokkos): "
        "The \"send buffer\" code path may not necessarily work with nonblocking sends.");

    Array<size_t> indicesOffsets (numExportPacketsPerLID.size(), 0);
    size_t ioffset = 0;
    for (int j=0; j<numExportPacketsPerLID.size(); ++j) {
      indicesOffsets[j] = ioffset;
      ioffset += numExportPacketsPerLID[j];
    }

    for (size_t i = 0; i < numBlocks; ++i) {
      size_t p = i + procIndex;
      if (p > (numBlocks - 1)) {
        p -= numBlocks;
      }

      if (plan.procIdsToSendTo_[p] != myProcID) {
        size_t sendArrayOffset = 0;
        size_t j = plan.startsTo_[p];
        size_t numPacketsTo_p = 0;
        for (size_t k = 0; k < plan.lengthsTo_[p]; ++k, ++j) {
          numPacketsTo_p += numExportPacketsPerLID[j];
          deep_copy_offset(sendArray, exports, sendArrayOffset,
              indicesOffsets[j], numExportPacketsPerLID[j]);
          sendArrayOffset += numExportPacketsPerLID[j];
        }
        if (numPacketsTo_p > 0) {
          ImpView tmpSend =
            subview_offset(sendArray, size_t(0), numPacketsTo_p);

          if (sendType == Details::DISTRIBUTOR_RSEND) {
            readySend<int> (tmpSend,
                as<int> (tmpSend.size ()),
                plan.procIdsToSendTo_[p], tag, *plan.comm_);
          }
          else if (sendType == Details::DISTRIBUTOR_ISEND) {
            exports_view_type tmpSendBuf =
              subview_offset (sendArray, size_t(0), numPacketsTo_p);
            requests_.push_back (isend<int> (tmpSendBuf, plan.procIdsToSendTo_[p],
                  tag, *plan.comm_));
          }
          else if (sendType == Details::DISTRIBUTOR_SSEND) {
            ssend<int> (tmpSend,
                as<int> (tmpSend.size ()),
                plan.procIdsToSendTo_[p], tag, *plan.comm_);
          }
          else { // if (sendType == Details::DISTRIBUTOR_SSEND)
            send<int> (tmpSend,
                as<int> (tmpSend.size ()),
                plan.procIdsToSendTo_[p], tag, *plan.comm_);
          }
        }
      }
      else { // "Sending" the message to myself
        selfNum = p;
        selfIndex = plan.startsTo_[p];
      }
    }

    if (plan.sendMessageToSelf_) {
      for (size_t k = 0; k < plan.lengthsTo_[selfNum]; ++k) {
        deep_copy_offset(imports, exports, selfReceiveOffset,
            indicesOffsets[selfIndex],
            numExportPacketsPerLID[selfIndex]);
        selfReceiveOffset += numExportPacketsPerLID[selfIndex];
        ++selfIndex;
      }
    }
  }
}

}
}

#endif
