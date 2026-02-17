//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
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
// Questions? Contact Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_IFPACK2RACEPRECONDITIONERTPETRA_DEF_HPP
#define _FROSCH_IFPACK2RACEPRECONDITIONERTPETRA_DEF_HPP

#include <FROSch_Ifpack2RACEPreconditionerTpetra_decl.hpp>

#include "Ifpack2_Details_getCrsMatrix.hpp"
#include "Ifpack2_RILUK_decl.hpp"

#ifdef HAVE_SHYLU_DDFROSCH_ZOLTAN2
#include "Zoltan2_TpetraRowGraphAdapter.hpp"
#include "Zoltan2_OrderingProblem.hpp"
#include "Zoltan2_OrderingSolution.hpp"
#endif

namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;
#ifdef USE_RACE
    using crs_matrix_type = Tpetra::CrsMatrix<>;
    using RACE_type = RACE::frontend<crs_matrix_type::scalar_type, crs_matrix_type::local_ordinal_type,crs_matrix_type::global_ordinal_type, crs_matrix_type::node_type>;
    using Teuchos::ParameterList;
#endif

    template<class SC,class LO,class GO,class NO>
    int Ifpack2RACEPreconditionerTpetra<SC,LO,GO,NO>::initialize()
    {
#ifdef DANE_DEBUG
        std::cout << "Calling Ifpack2RACEPreconditionerTpetra<SC,LO,GO,NO>::initialize..." << std::endl;
#endif
        FROSCH_TIMER_START_SOLVER(initializeTime,"Ifpack2RACEPreconditionerTpetra::initialize");
        this->IsInitialized_ = true;
        this->IsComputed_ = false;

        Ifpack2RACEPreconditioner_->initialize();
#ifdef DANE_DEBUG
        std::cout << "initialize() done" << std::endl;
#endif
        return 0;
    }

    template<class SC,class LO,class GO,class NO>
    int Ifpack2RACEPreconditionerTpetra<SC,LO,GO,NO>::compute()
    {
#ifdef DANE_DEBUG
        std::cout << "Calling Ifpack2RACEPreconditionerTpetra<SC,LO,GO,NO>::compute()..." << std::endl;
#endif
        FROSCH_TIMER_START_SOLVER(computeTime,"Ifpack2RACEPreconditionerTpetra::compute");
        FROSCH_ASSERT(this->IsInitialized_,"FROSch::Ifpack2vPreconditionerTpetra: !this->IsInitialized_");
        
#ifdef USE_RACE
        std::cout << "Preprocessing local subdomains with RACE!" << std::endl;

        // How will this be passed between subroutines?
        // Just leave outside? Above subroutines?
        Teuchos::RCP<RACE_type> race;
        RCP<crs_matrix_type> A;
        void* raceVoidHandle = NULL;


        ParameterList RACE_params("RACE");
        // TODO: manually fill parameter list!
        //   RACE_params.set("Cache size", atof(args.RACE_cacheSize.c_str()));
        RACE_params.set("Cache size", 5.0); // Just for testing
        //   int highestPower = atoi(args.RACE_highestPower.c_str());
        //   RACE_params.set("Highest power", highestPower);
        //   RACE_params.set("Preconditioner", RACE_precon_type);
        //   RACE_params.set("Preconditioner side", args.precSide);
        //   RACE_params.set("Inner iteration", preconditionerInnerSweep);
        //   RACE_params.set("Inner damping", preconditionerInnerDamping);

       // Try to extract a Tpetra::CrsMatrix from the Ifpack2 preconditioner's matrix (non-const)
        ConstTCrsMatrixPtr constCrs = Ifpack2::Details::getCrsMatrix<SC,LO,GO,NO>(Ifpack2RACEPreconditioner_->getMatrix());
        if (constCrs.is_null()) {
            std::cerr << "Ifpack2RACEPreconditionerTpetra::initialize: could not extract a Tpetra::CrsMatrix for RACE." << std::endl;
            return -1;
        }

        // Remove constness (only safe if the underlying object is truly mutable)
        Teuchos::RCP<crs_matrix_type> crsMat = Teuchos::rcp_const_cast<crs_matrix_type>(constCrs);
        if (crsMat.is_null()) {
            std::cerr << "Ifpack2RACEPreconditionerTpetra::initialize: could not obtain non-const CrsMatrix for RACE." << std::endl;
            return -1;
        }

        // Init interface
        race = Teuchos::rcp(new RACE_type(crsMat, RACE_params));
        
        // Have RACE permute matrix
        A = race->getPermutedMatrix();
#ifdef DANE_DEBUG
        TEUCHOS_TEST_FOR_EXCEPTION(A.is_null(), std::runtime_error,
            "RACE returned null matrix.");

        std::cout << "RACE permuted matrix: "
                << " rows=" << A->getGlobalNumRows()
                << " cols=" << A->getGlobalNumCols()
                << " nnz="  << A->getGlobalNumEntries()
                << std::endl;
#endif
        if (A.is_null()) {
            std::cerr << "Ifpack2RACEPreconditionerTpetra::initialize: race returned null matrix." << std::endl;
            return -1;
        }

        // Keep interface void pointer for later
        raceVoidHandle = (void*)(race.getRawPtr());

        // Cast back to TRowMatrix, and give back to Ifpack2
        // But, since there does not exist a "setMatrix" for generic Ifpack2RACEPreconditioner_,
        // need to rebuild preconditioner
        auto rowMatConst = rcp_dynamic_cast<const TRowMatrix>(A);
        if (rowMatConst.is_null()) {
            std::cerr << "Ifpack2RACEPreconditionerTpetra::initialize: could not cast permuted matrix to Tpetra::RowMatrix." << std::endl;
            return -1;
        }

        // Get the existing parameter list and type from the current preconditioner
        std::string precType = Ifpack2Type_;
        Teuchos::ParameterList params = Ifpack2Params_;

        // Create a new preconditioner of the same type with the new (permuted) matrix
        Teuchos::RCP<Ifpack2::Preconditioner<SC,LO,GO,NO>> newPrec;

        Ifpack2::Details::OneLevelFactory<TRowMatrix> ifpack2Factory;
        
        try {
            newPrec = ifpack2Factory.create(precType, rowMatConst);
        } catch (std::exception& e) {
            std::cerr << "ifpack2Factory.create() threw: " << e.what() << std::endl;
            return -1;
        }
        if (newPrec.is_null()) {
            std::cerr << "Ifpack2RACEPreconditionerTpetra::initialize: ifpack2Factory.create() returned null." << std::endl;
            return -1;
        }
#ifdef DANE_DEBUG
        auto mat = newPrec->getMatrix();
        std::cout << "Ifpack2 preconditioner now uses matrix with "
                << mat->getGlobalNumRows() << " rows and "
                << mat->getGlobalNumEntries() << " nnz" << std::endl;
#endif

        // Reapply the exact same parameters
        newPrec->setParameters(params);

        // Replace old instance
        Ifpack2RACEPreconditioner_ = newPrec;

        // Save the RACE RCP into the class
        this->race_ = race;
#ifdef DANE_DEBUG
        {
            auto map = Ifpack2RACEPreconditioner_->getMatrix()->getRowMap(); // or original map
            Tpetra::MultiVector<SC,LO,GO,NO> t1(map, 1);
            Tpetra::MultiVector<SC,LO,GO,NO> t2(map, 1);
            Tpetra::MultiVector<SC,LO,GO,NO> t3(map, 1);

            // fill t1 with distinct values so we can track them
            {
                auto v = t1.getDataNonConst(0);
                for (size_t i=0;i<v.size();++i) v[i] = static_cast<SC>(i+1);
            }

            race_->origToPerm(t2, t1); // t2 = perm(t1)
            race_->permToOrig(t3, t2); // t3 = invperm(t2)

            // check equality t1 == t3
            double err = 0.0;
            {
                auto a = t1.get2dView();
                auto b = t3.get2dView();
                for (size_t k=0;k<1;k++)
                    for (size_t i=0;i<t1.getLocalLength(); ++i)
                        err = std::max(err, std::abs(a[k][i] - b[k][i]));
            }
            std::cout << "RACE permutation round-trip max error: " << err << std::endl;
            TEUCHOS_TEST_FOR_EXCEPTION(err > 1e-12, std::runtime_error,
                "RACE permutation round-trip failed (swap perm semantics).");
        }
#endif
#endif
        
        this->IsComputed_ = true;
        Ifpack2RACEPreconditioner_->compute();
#ifdef DANE_DEBUG
        std::cout << "compute() done" << std::endl;
#endif
        return 0;
    }

    template<class SC,class LO,class GO,class NO>
    void Ifpack2RACEPreconditionerTpetra<SC,LO,GO,NO>::apply(const XMultiVector &x,
                                                         XMultiVector &y,
                                                         ETransp mode,
                                                         SC alpha,
                                                         SC beta) const
    {
#ifdef DANE_DEBUG
        std::cout << "Calling Ifpack2RACEPreconditionerTpetra<SC,LO,GO,NO>::apply..." << std::endl;

        // Print diagonal of permuted matrix
        auto Aperm = Ifpack2RACEPreconditioner_->getMatrix();
        Tpetra::Vector<SC,LO,GO,NO> diag(Aperm->getRowMap());
        Aperm->getLocalDiagCopy(diag);
        auto dv = diag.getData();
        for (int i=0;i<std::min<int>(10, dv.size()); ++i) std::cout << dv[i] << " ";
#endif
        FROSCH_TIMER_START_SOLVER(applyTime,"Ifpack2RACEPreconditionerTpetra::apply");
        FROSCH_ASSERT(this->IsComputed_,"FROSch::Ifpack2RACEPreconditionerTpetra: !this->IsComputed_.");

        const TpetraMultiVector<SC,LO,GO,NO> * xTpetraMultiVectorX = dynamic_cast<const TpetraMultiVector<SC,LO,GO,NO> *>(&x);
        FROSCH_ASSERT(xTpetraMultiVectorX,"FROSch::Ifpack2RACEPreconditionerTpetra: dynamic_cast failed.");
        TMultiVectorPtr tpetraMultiVectorX = xTpetraMultiVectorX->getTpetra_MultiVector();

        const TpetraMultiVector<SC,LO,GO,NO> * xTpetraMultiVectorY = dynamic_cast<const TpetraMultiVector<SC,LO,GO,NO> *>(&y);
        FROSCH_ASSERT(xTpetraMultiVectorY,"FROSch::Ifpack2RACEPreconditionerTpetra: dynamic_cast failed.");
        TMultiVectorPtr tpetraMultiVectorY = xTpetraMultiVectorY->getTpetra_MultiVector();
#ifdef USE_RACE

#if 0
        // allocate workspace once if not present
        if (raceXwork_.is_null() || raceXwork_->getNumVectors() != tpetraMultiVectorX->getNumVectors()) {
            raceXwork_ = Teuchos::rcp(new Tpetra::MultiVector<SC,LO,GO,NO>(tpetraMultiVectorX->getMap(), tpetraMultiVectorX->getNumVectors()));
        }
        if (raceYwork_.is_null() || raceYwork_->getNumVectors() != tpetraMultiVectorY->getNumVectors()) {
            raceYwork_ = Teuchos::rcp(new Tpetra::MultiVector<SC,LO,GO,NO>(tpetraMultiVectorY->getMap(), tpetraMultiVectorY->getNumVectors()));
        }
#elif 1
        // Alternate way to allocate working vectors
        // get preconditioner matrix (permuted) maps
        auto precMat = Ifpack2RACEPreconditioner_->getMatrix();
        auto domainMap = precMat->getDomainMap();   // input map for apply
        auto rangeMap  = precMat->getRangeMap();    // output map for apply

        if (raceXwork_.is_null() || raceXwork_->getNumVectors() != tpetraMultiVectorX->getNumVectors()
            || !raceXwork_->getMap()->isCompatible(*domainMap)) {
            raceXwork_ = Teuchos::rcp(new Tpetra::MultiVector<SC,LO,GO,NO>(domainMap, tpetraMultiVectorX->getNumVectors()));
        }
        if (raceYwork_.is_null() || raceYwork_->getNumVectors() != tpetraMultiVectorY->getNumVectors()
            || !raceYwork_->getMap()->isCompatible(*rangeMap)) {
            raceYwork_ = Teuchos::rcp(new Tpetra::MultiVector<SC,LO,GO,NO>(rangeMap, tpetraMultiVectorY->getNumVectors()));
        }
#endif

        // Permute original x -> permuted Xp
        race_->origToPerm(*raceXwork_, *tpetraMultiVectorX);

        // Apply preconditioner built on permuted matrix
        Ifpack2RACEPreconditioner_->apply(*raceXwork_, *raceYwork_, mode, alpha, beta);

        // Permute result back to original ordering
        race_->permToOrig(*tpetraMultiVectorY, *raceYwork_);
#else
#ifdef HAVE_SHYLU_DDFROSCH_ZOLTAN2
        if (this->useZoltan2 && this->useRILUK) {
            auto A = Ifpack2RACEPreconditioner_->getMatrix();
            auto filteredA = rcp_dynamic_cast<const TRowMatrixFilterType> (A);

            // make copies of X & Y
            TMultiVector ReorderedX (*tpetraMultiVectorX, Teuchos::Copy);
            TMultiVector ReorderedY (*tpetraMultiVectorY, Teuchos::Copy);

            // permute X & Y
            filteredA->permuteOriginalToReordered (*tpetraMultiVectorX, ReorderedX);
            filteredA->permuteOriginalToReordered (*tpetraMultiVectorY, ReorderedY);

            // solve
            Ifpack2RACEPreconditioner_->apply(ReorderedX,ReorderedY,mode,alpha,beta);

            // permute X back
            filteredA->permuteReorderedToOriginal (ReorderedY, *tpetraMultiVectorY);
        } else
#endif
#endif
        {
            Ifpack2RACEPreconditioner_->apply(*tpetraMultiVectorX,*tpetraMultiVectorY,mode,alpha,beta);
        }
#ifdef DANE_DEBUG
        std::cout << "apply() done" << std::endl;

        {
        auto M = raceYwork_;
        auto mv = M->get2dView();
        bool anyNaN = false;
        size_t rows = M->getLocalLength();
        size_t nvec = M->getNumVectors();
        for (size_t k=0;k<nvec && !anyNaN;++k) {
            for (size_t i=0;i<rows; ++i) {
            if (! std::isfinite(mv[k][i])) { anyNaN = true; break; }
            }
        }
        std::cout << "DEBUG: permuted result has NaN? " << (anyNaN ? "YES" : "no") << std::endl;

        // Also print first few entries for manual inspection
        std::cout << "DEBUG: first 10 entries of permuted result (v0): ";
        for (size_t i=0;i<std::min<size_t>(10, rows); ++i) std::cout << mv[0][i] << " ";
        std::cout << std::endl;
        }
#endif
    }

    template<class SC,class LO,class GO,class NO>
    int Ifpack2RACEPreconditionerTpetra<SC,LO,GO,NO>::updateMatrix(ConstXMatrixPtr k,
                                                               bool reuseInitialize)
    {
#ifdef DANE_DEBUG
        std::cout << "Calling Ifpack2RACEPreconditionerTpetra<SC,LO,GO,NO>::updateMatrix..." << std::endl;
#endif
        if (this->useRILUK) {
            const CrsMatrixWrap<SC,LO,GO,NO>& crsOp = dynamic_cast<const CrsMatrixWrap<SC,LO,GO,NO>&>(*this->K_);
            const TpetraCrsMatrix<SC,LO,GO,NO>& xTpetraMat = dynamic_cast<const TpetraCrsMatrix<SC,LO,GO,NO>&>(*crsOp.getCrsMatrix());
            ConstTCrsMatrixPtr tpetraMat = xTpetraMat.getTpetra_CrsMatrix();

            auto RILUPreconditioner = rcp_dynamic_cast<Ifpack2::RILUK<TRowMatrix>>(Ifpack2RACEPreconditioner_);
#ifdef HAVE_SHYLU_DDFROSCH_ZOLTAN2
            if (this->useZoltan2) {
                // if K is replaced, then we need to re-wrap into matrix filter
                auto rowMat = rcp_dynamic_cast<const TRowMatrix>(tpetraMat);
                auto filteredMat = rcp(new TRowMatrixFilterType(rowMat, this->perm, this->revperm));
                RILUPreconditioner->setMatrix(filteredMat);
            } else
#endif
            {
                RILUPreconditioner->setMatrix(tpetraMat);
            }
            return 0;
        }
        FROSCH_ASSERT(false,"FROSch::Ifpack2RACEPreconditionerTpetra: updateMatrix() is not implemented for the Ifpack2RACEPreconditionerTpetra yet.");
#ifdef DANE_DEBUG
        std::cout << "updateMatrix done" << std::endl;
#endif
    }

    template<class SC,class LO,class GO,class NO>
    Ifpack2RACEPreconditionerTpetra<SC,LO,GO,NO>::Ifpack2RACEPreconditionerTpetra(ConstXMatrixPtr k,
                                                                          ParameterListPtr parameterList,
                                                                          string description) :
    Solver<SC,LO,GO,NO> (k,parameterList,description)
    {
#ifdef DANE_DEBUG
        std::cout << "Calling Ifpack2RACEPreconditionerTpetra ctor" << std::endl;
#endif
        FROSCH_TIMER_START_SOLVER(Ifpack2RACEPreconditionerTpetraTime,"Ifpack2RACEPreconditionerTpetra::Ifpack2RACEPreconditionerTpetra");
        FROSCH_ASSERT(!this->K_.is_null(),"FROSch::Ifpack2RACEPreconditionerTpetra: K_ is null.");
        FROSCH_ASSERT(this->K_->getRowMap()->lib()==UseTpetra,"FROSch::Ifpack2RACEPreconditionerTpetra: Not compatible with Epetra.")

        const CrsMatrixWrap<SC,LO,GO,NO>& crsOp = dynamic_cast<const CrsMatrixWrap<SC,LO,GO,NO>&>(*this->K_);
        const TpetraCrsMatrix<SC,LO,GO,NO>& xTpetraMat = dynamic_cast<const TpetraCrsMatrix<SC,LO,GO,NO>&>(*crsOp.getCrsMatrix());
        ConstTCrsMatrixPtr tpetraMat = xTpetraMat.getTpetra_CrsMatrix();
        TEUCHOS_TEST_FOR_EXCEPT(tpetraMat.is_null());

        auto solverName = this->ParameterList_->get("Solver","RILUK");
        this->useRILUK = (solverName == "RILUK");
        this->useZoltan2 = this->ParameterList_->get("RILUK: use reordering", false);

        Ifpack2::Details::OneLevelFactory<TRowMatrix> ifpack2Factory;
#ifdef HAVE_SHYLU_DDFROSCH_ZOLTAN2
        if (this->useZoltan2 && this->useRILUK) {
            // pre-ordering matrix, before constructing Ifpack2RACEPreconditioner_ with the matrix
            {
                typedef Tpetra::RowGraph<LO, GO, NO> row_graph_type;
                typedef Zoltan2::TpetraRowGraphAdapter<row_graph_type> z2_adapter_type;
                typedef Zoltan2::OrderingProblem<z2_adapter_type> z2_ordering_problem_type;
                typedef Zoltan2::LocalOrderingSolution<LO> z2_ordering_solution_type;

                auto comm = tpetraMat->getRowMap()->getComm();
                auto constActiveGraph = Teuchos::rcp_const_cast<const row_graph_type>(tpetraMat->getGraph());
                z2_adapter_type Zoltan2Graph (constActiveGraph);

                Teuchos::ParameterList zoltan2_params = this->ParameterList_->sublist ("RILUK: reordering list");
                z2_ordering_problem_type MyOrderingProblem (&Zoltan2Graph, &zoltan2_params, comm);

                MyOrderingProblem.solve ();
                z2_ordering_solution_type sol (*MyOrderingProblem.getLocalOrderingSolution());
                this->perm = sol.getPermutationRCPConst (true);
                this->revperm = sol.getPermutationRCPConst ();
            }

            // wrap into matrix filter
            auto rowMat = rcp_dynamic_cast<const TRowMatrix>(tpetraMat);
            auto filteredMat = rcp(new TRowMatrixFilterType(rowMat, this->perm, this->revperm));
            Ifpack2RACEPreconditioner_ = ifpack2Factory.create(solverName,filteredMat);
        } else
#endif
        {
            Ifpack2RACEPreconditioner_ = ifpack2Factory.create(solverName,tpetraMat);
        }

        // Dane 17.02.2026 Save, to use for RACE during "initialize()" //
        Ifpack2Type_ = solverName;

        ParameterListPtr ifpack2ParameterList = sublist(this->ParameterList_,"Ifpack2_RACE");
        if (ifpack2ParameterList->isSublist(solverName)) {
            ifpack2ParameterList = sublist(ifpack2ParameterList, solverName);
        }
        ifpack2ParameterList->setName("Ifpack2_RACE");

        // store a copy
        Ifpack2Params_ = *ifpack2ParameterList;

        // apply to preconditioner
        Ifpack2RACEPreconditioner_->setParameters(Ifpack2Params_);
        //
#ifdef DANE_DEBUG
        std::cout << "ctor done" << std::endl;
#endif
        // Just for debugging
        Ifpack2Params_.print(std::cout);
    }
}

#endif
