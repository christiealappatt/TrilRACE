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
#include <iostream>
#include <stdexcept>

// Teuchos
#include <Teuchos_StandardCatchMacros.hpp>

// Galeri
#include <Galeri_XpetraParameters.hpp>
#include <Galeri_XpetraProblemFactory.hpp>
#include <Galeri_XpetraUtils.hpp>
#include <Galeri_XpetraMaps.hpp>

#include <Kokkos_DefaultNode.hpp> // For Epetra only runs this points to FakeKokkos in Xpetra

#include "Xpetra_ConfigDefs.hpp"
#include <Xpetra_MatrixMatrix.hpp>

#include <MueLu.hpp>
#include <MueLu_Level.hpp>
#include <MueLu_MLParameterListInterpreter.hpp>
#include <Tpetra_Operator.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <Tpetra_CrsMatrixMultiplyOp.hpp>
#include <Tpetra_Experimental_BlockCrsMatrix_Helpers.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_Utilities.hpp>
#include <Xpetra_TpetraVector.hpp>
#include <Xpetra_TpetraBlockCrsMatrix.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Ifpack2_Factory.hpp>

// Belos
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosSolverFactory.hpp"
#include <BelosTpetraAdapter.hpp>

// Define default data types
typedef double Scalar;
typedef int LocalOrdinal;
typedef int GlobalOrdinal;
typedef KokkosClassic::DefaultNode::DefaultNodeType Node;


using Teuchos::RCP;
using Teuchos::rcp;


//----------------------------------------------------------------------------------------------------------
//
// This example demonstrates how to use MueLu in a fashion that looks like ML's LevelWrap
//
// In this example, we suppose that the user provides a fine matrix A and P & R operators.  These are
// given to MueLu which then forms the next finest matrix and makes a hierarchy from there.
//
//----------------------------------------------------------------------------------------------------------

const std::string thickSeparator = "==========================================================================================================================";
const std::string thinSeparator  = "--------------------------------------------------------------------------------------------------------------------------";

const std::string prefSeparator = "=====================================";


namespace MueLuExamples {
#include <MueLu_UseShortNames.hpp>

  Teuchos::ScalarTraits<SC>::magnitudeType diff_vectors(const Vector & X, const Vector & Y) {
    RCP<Vector> diff = VectorFactory::Build(X.getMap());
    diff->update(1.0,X,-1.0,Y,0.0);
    Teuchos::Array<Teuchos::ScalarTraits<SC>::magnitudeType> mt(1);
    diff->norm2(mt);
    return mt[0];
  }

  Teuchos::ScalarTraits<SC>::magnitudeType compute_resid_norm(const Matrix & A, const Vector & X, const Vector & B) {
    RCP<Vector> temp = VectorFactory::Build(X.getMap());
    A.apply(X,*temp);
    temp->update(1.0,B,-1.0);
    Teuchos::Array<Teuchos::ScalarTraits<SC>::magnitudeType> mt(1);
    temp->norm2(mt);
    return mt[0];
  }

// --------------------------------------------------------------------------------------
  void solve_system_belos(RCP<Matrix> & A, RCP<Vector>&  X, RCP<Vector> & B, Teuchos::ParameterList & MueLuList, const std::string & belos_solver, RCP<Teuchos::ParameterList> & SList) {
    using Teuchos::RCP;
    using Teuchos::rcp;
    typedef Tpetra::Operator<SC,LO,GO> Tpetra_Operator;
    typedef Tpetra::CrsMatrix<SC,LO,GO> Tpetra_CrsMatrix;
    typedef Tpetra::Experimental::BlockCrsMatrix<SC,LO,GO,Tpetra_CrsMatrix::node_type> Tpetra_BlockCrsMatrix;
    typedef Xpetra::TpetraBlockCrsMatrix<SC,LO,GO,Tpetra_CrsMatrix::node_type> Xpetra_TpetraBlockCrsMatrix;
    typedef Tpetra::Vector<SC,LO,GO> Tpetra_Vector;
    typedef Tpetra::MultiVector<SC,LO,GO> Tpetra_MultiVector;

    RCP<Tpetra_Operator>    At = MueLu::Utilities<SC,LO,GO,Tpetra_CrsMatrix::node_type>::Op2NonConstTpetraRow(A);
    RCP<Tpetra_Operator>    Mt = MueLu::CreateTpetraPreconditioner(At,MueLuList);
    RCP<Tpetra_MultiVector> Xt = Xpetra::toTpetra(*X);
    RCP<Tpetra_MultiVector> Bt = Xpetra::toTpetra(*B);

    if(Xt.is_null() || Bt.is_null() || At.is_null() || Mt.is_null()) throw std::runtime_error("ERROR: Xpetra to Tpetra conversion failed");

    typedef Tpetra_MultiVector MV;
    typedef Tpetra_Operator OP;
    RCP<Belos::LinearProblem<SC,MV,OP> > belosProblem = rcp(new Belos::LinearProblem<SC,MV,OP>(At, Xt, Bt));
    belosProblem->setLeftPrec(Mt);
    belosProblem->setProblem(Xt,Bt);
    
    Belos::SolverFactory<SC, MV, OP> BelosFactory;
    Teuchos::RCP<Belos::SolverManager<SC, MV, OP> > BelosSolver = BelosFactory.create(belos_solver, SList);
    BelosSolver->setProblem(belosProblem);
    BelosSolver->solve();
  }

 // --------------------------------------------------------------------------------------
  void solve_system_ifpack2(RCP<Matrix> & A, RCP<Vector>&  X, RCP<Vector> & B, const std::string & ifpack2_solver, Teuchos::ParameterList & Ifpack2List) {
    using Teuchos::RCP;
    using Teuchos::rcp;
    typedef Tpetra::Operator<SC,LO,GO> Tpetra_Operator;
    typedef Ifpack2::Preconditioner<SC,LO,GO> Ifpack2_Preconditioner;
    typedef Tpetra::CrsMatrix<SC,LO,GO> Tpetra_CrsMatrix;
    typedef Tpetra::RowMatrix<SC,LO,GO> Tpetra_RowMatrix;
    typedef Tpetra::Experimental::BlockCrsMatrix<SC,LO,GO,Tpetra_CrsMatrix::node_type> Tpetra_BlockCrsMatrix;
    typedef Xpetra::TpetraBlockCrsMatrix<SC,LO,GO,Tpetra_CrsMatrix::node_type> Xpetra_TpetraBlockCrsMatrix;
    typedef Tpetra::Vector<SC,LO,GO> Tpetra_Vector;
    typedef Tpetra::MultiVector<SC,LO,GO> Tpetra_MultiVector;

    RCP<Tpetra_RowMatrix>    At = MueLu::Utilities<SC,LO,GO,Tpetra_CrsMatrix::node_type>::Op2NonConstTpetraRow(A);
    RCP<Tpetra_MultiVector> Xt = Xpetra::toTpetra(*X);
    RCP<Tpetra_MultiVector> Bt = Xpetra::toTpetra(*B);

    RCP<Ifpack2_Preconditioner> Solver = Ifpack2::Factory::create<Tpetra_RowMatrix>(ifpack2_solver,At);
    Solver->setParameters(Ifpack2List);
    Solver->initialize();
    Solver->compute();

    Solver->apply(*Bt,*Xt);
  }

  // --------------------------------------------------------------------------------------
  // This routine generate's the user's original A matrix and nullspace
  void generate_user_matrix_and_nullspace(std::string &matrixType,  Xpetra::UnderlyingLib & lib,Teuchos::ParameterList &galeriList,  RCP<const Teuchos::Comm<int> > &comm, RCP<Matrix> & A, RCP<MultiVector> & nullspace){
    using Teuchos::RCP;

    RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    Teuchos::FancyOStream& out = *fancy;

    RCP<const Map>   map;
    RCP<MultiVector> coordinates;
    if (matrixType == "Laplace1D") {
      map = Galeri::Xpetra::CreateMap<LO, GO, Node>(lib, "Cartesian1D", comm, galeriList);
      coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC,LO,GO,Map,MultiVector>("1D", map, galeriList);

    } else if (matrixType == "Laplace2D" || matrixType == "Star2D" || matrixType == "BigStar2D" || matrixType == "Elasticity2D") {
      map = Galeri::Xpetra::CreateMap<LO, GO, Node>(lib, "Cartesian2D", comm, galeriList);
      coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC,LO,GO,Map,MultiVector>("2D", map, galeriList);

    } else if (matrixType == "Laplace3D" || matrixType == "Brick3D" || matrixType == "Elasticity3D") {
      map = Galeri::Xpetra::CreateMap<LO, GO, Node>(lib, "Cartesian3D", comm, galeriList);
      coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC,LO,GO,Map,MultiVector>("3D", map, galeriList);
    }

    // Expand map to do multiple DOF per node for block problems
    if (matrixType == "Elasticity2D" || matrixType == "Elasticity3D")
      map = Xpetra::MapFactory<LO,GO,Node>::Build(map, (matrixType == "Elasticity2D" ? 2 : 3));

    out << "Processor subdomains in x direction: " << galeriList.get<int>("mx") << std::endl
        << "Processor subdomains in y direction: " << galeriList.get<int>("my") << std::endl
        << "Processor subdomains in z direction: " << galeriList.get<int>("mz") << std::endl
        << "========================================================" << std::endl;

    RCP<Galeri::Xpetra::Problem<Map,CrsMatrixWrap,MultiVector> > Pr =
        Galeri::Xpetra::BuildProblem<SC,LO,GO,Map,CrsMatrixWrap,MultiVector>(matrixType, map, galeriList);

    A = Pr->BuildMatrix();

    if (matrixType == "Elasticity2D" || matrixType == "Elasticity3D") {
      nullspace = Pr->BuildNullspace();
      A->SetFixedBlockSize((matrixType == "Elasticity2D") ? 2 : 3);
    }

  }

}


// --------------------------------------------------------------------------------------
int main(int argc, char *argv[]) {
#include <MueLu_UseShortNames.hpp>
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::TimeMonitor;

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);

  bool success = true;
  bool verbose = true;
  try {
    RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();

    RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    Teuchos::FancyOStream& out = *fancy;

    typedef Teuchos::ScalarTraits<SC> STS;

    // =========================================================================
    // Parameters initialization
    // =========================================================================
    Teuchos::CommandLineProcessor clp(false);

    GO nx = 100, ny = 100, nz = 100;
    Galeri::Xpetra::Parameters<GO> galeriParameters(clp, nx, ny, nz, "Laplace2D"); // manage parameters of the test case
    Xpetra::Parameters             xpetraParameters(clp);                          // manage parameters of Xpetra
    std::string matFileName       = "";  clp.setOption("matrix",&matFileName,"read matrix from a file");
    LO blocksize = 1;                    clp.setOption("blocksize",&blocksize,"block size");

    switch (clp.parse(argc, argv)) {
      case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:        return EXIT_SUCCESS; break;
      case Teuchos::CommandLineProcessor::PARSE_ERROR:
      case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION: return EXIT_FAILURE; break;
      case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:                               break;
    }

    Xpetra::UnderlyingLib lib = xpetraParameters.GetLib();
    ParameterList galeriList = galeriParameters.GetParameterList();

    if(lib!=Xpetra::UseTpetra) 
      throw std::runtime_error("This test only works with Tpetra linear algebra");

    // =========================================================================
    // Problem construction
    // =========================================================================
    RCP<const Map>   map;
    RCP<Matrix> A;
    RCP<MultiVector> nullspace;

    typedef Tpetra::CrsMatrix<SC,LO,GO> Tpetra_CrsMatrix;
    typedef Tpetra::Operator<SC,LO,GO> Tpetra_Operator;
    typedef Tpetra::Experimental::BlockCrsMatrix<SC,LO,GO,Tpetra_CrsMatrix::node_type> Tpetra_BlockCrsMatrix;
    typedef Xpetra::TpetraBlockCrsMatrix<SC,LO,GO,Tpetra_CrsMatrix::node_type> Xpetra_TpetraBlockCrsMatrix;
    typedef Xpetra::CrsMatrix<SC,LO,GO,Tpetra_CrsMatrix::node_type> Xpetra_CrsMatrix;
    typedef Xpetra::CrsMatrixWrap<SC,LO,GO,Tpetra_CrsMatrix::node_type> Xpetra_CrsMatrixWrap;
    typedef Teuchos::ScalarTraits<SC>::magnitudeType SCN;

    RCP<Tpetra_CrsMatrix> Acrs;
    RCP<Tpetra_BlockCrsMatrix> Ablock;

    if(matFileName.length() > 0) {
      // Read matrix from disk
      out << thickSeparator << std::endl << "Reading matrix from disk" <<std::endl;
      typedef Tpetra::MatrixMarket::Reader<Tpetra_CrsMatrix> reader_type;
      Acrs = reader_type::readSparseFile(matFileName,comm);
    }
    else{
      // Use Galeri
      out << thickSeparator << std::endl << xpetraParameters << galeriParameters;
      std::string matrixType = galeriParameters.GetMatrixType();
      RCP<Matrix> Axp;
      MueLuExamples::generate_user_matrix_and_nullspace(matrixType,lib,galeriList,comm,Axp,nullspace);
      Acrs = Xpetra::Helpers<SC,LO,GO>::Op2NonConstTpetraCrs(Axp);
    }
    // Block this bad boy
    Ablock = Tpetra::Experimental::convertToBlockCrsMatrix(*Acrs,blocksize);

    // Now wrap BlockCrs to Xpetra::Matrix
    RCP<Xpetra_CrsMatrix> Axt = rcp(new Xpetra_TpetraBlockCrsMatrix(Ablock));
    A = rcp(new Xpetra_CrsMatrixWrap(Axt));

    // =========================================================================
    // Setups and solves
    // =========================================================================
    map=Xpetra::toXpetra(Acrs->getRowMap());

    RCP<Vector> X1 = VectorFactory::Build(map);
    RCP<Vector> X2 = VectorFactory::Build(map);
    RCP<Vector> B = VectorFactory::Build(map);
    B->setSeed(846930886);
    B->randomize();
    RCP<TimeMonitor> tm;

    // Belos Options
    RCP<Teuchos::ParameterList> SList = rcp(new Teuchos::ParameterList );
    SList->set("Verbosity",Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);
    SList->set("Output Frequency",10);
    SList->set("Output Style",Belos::Brief);
    SList->set("Maximum Iterations",10);
    SList->set("Convergence Tolerance",5e-2);


    // =========================================================================
    // Solve #1 (fixed point + Jacobi)
    // =========================================================================
    out << thickSeparator << std::endl;
    out << prefSeparator << " Solve 1: Fixed Point + Jacobi"<< prefSeparator <<std::endl;
    {
      Teuchos::ParameterList MueList;
      MueList.set("max levels",1);
      MueList.set("coarse: type", "RELAXATION");

      std::string belos_solver("Fixed Point");   
      MueLuExamples::solve_system_belos(A,X1,B,MueList,belos_solver,SList);

      SCN result = MueLuExamples::compute_resid_norm(*A,*X1,*B);
      out<<"Solve #1: Residual Norm = "<<result<<std::endl;
    }

    // =========================================================================
    // Solve #2 (striaght up Jacobi)
    // =========================================================================
    out << thickSeparator << std::endl;
    out << prefSeparator << " Solve 2: Fixed Jacobi"<< prefSeparator <<std::endl;
    {
      Teuchos::ParameterList IList;
      IList.set("relaxation: type","Jacobi");
      IList.set("relaxation: damping factor",1.0);
      IList.set("relaxation: sweeps",10);
      std::string ifpack2_precond("RELAXATION");
      
      MueLuExamples::solve_system_ifpack2(A,X2,B,ifpack2_precond,IList);

      SCN result = MueLuExamples::compute_resid_norm(*A,*X2,*B);
      out<<"Solve #2: Residual Norm = "<<result<<std::endl;
    }

    // Compare 1 & 2
    SCN norm = MueLuExamples::diff_vectors(*X1,*X2);
    if(norm > 1e-10) {
      out<<"ERROR: Norm of Solve #1 and Solve #2 differs by "<<norm<<std::endl;
      success=false;
    }
      
  }//end try
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}
