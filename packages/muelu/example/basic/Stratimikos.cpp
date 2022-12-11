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

/*
   Call MueLu via the Stratimikos interface.

Usage:
./MueLu_Stratimikos.exe : use xml configuration file stratimikos_ParameterList.xml

Note:
The source code is not MueLu specific and can be used with any Stratimikos strategy.
*/

// Teuchos includes
#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

// Thyra includes
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_SolveSupportTypes.hpp>

// Stratimikos includes
#include <Stratimikos_LinearSolverBuilder.hpp>
#include <Stratimikos_MueLuHelpers.hpp>

// Xpetra include
#include <Xpetra_Parameters.hpp>

// MueLu includes
#include <Thyra_MueLuPreconditionerFactory.hpp>
#include <MatrixLoad.hpp>

// Galeri includes
#include <Galeri_XpetraParameters.hpp>

// Ifpack2 includes
#ifdef HAVE_MUELU_IFPACK2
#include <Thyra_Ifpack2PreconditionerFactory.hpp>
#endif

#include "RACE_frontend.hpp"


// See example here:
//
// http://en.cppreference.com/w/cpp/string/byte/toupper
std::string stringToUpper (std::string s)
{
  std::transform (s.begin (), s.end (), s.begin (),
                  [] (unsigned char c) { return std::toupper (c); });
  return s;
}


template<typename Scalar,class LocalOrdinal,class GlobalOrdinal,class Node>
int main_(Teuchos::CommandLineProcessor &clp, Xpetra::UnderlyingLib lib, int argc, char *argv[]) {
  #include <MueLu_UseShortNames.hpp>
  typedef Teuchos::ScalarTraits<Scalar> STS;
  typedef typename STS::coordinateType real_type;
  typedef Xpetra::MultiVector<real_type,LocalOrdinal,GlobalOrdinal,Node> RealValuedMultiVector;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using Teuchos::TimeMonitor;

  bool success = false;
  bool verbose = true;
  try {

    //
    // MPI initialization
    //
    RCP< const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();

    //
    // Parameters
    //
    // manage parameters of the test case
    Galeri::Xpetra::Parameters<GlobalOrdinal> matrixParameters(clp, 160, 160, 160, "Laplace3D");
    // manage parameters of Xpetra
    Xpetra::Parameters                        xpetraParameters(clp);

    // command line parameters
    std::string xmlFileName       = "stratimikos_ParameterList.xml"; clp.setOption("xml",      &xmlFileName,       "read parameters from an xml file");
    std::string yamlFileName      = "";                 clp.setOption("yaml",                  &yamlFileName,      "read parameters from a yaml file");
    bool        printTimings      = false;              clp.setOption("timings", "notimings",  &printTimings,      "print timings to screen");
    bool        use_stacked_timer = false;              clp.setOption("stacked-timer", "no-stacked-timer", &use_stacked_timer, "Run with or without stacked timer output");
    std::string timingsFormat     = "table-fixed";      clp.setOption("time-format",           &timingsFormat,     "timings format (table-fixed | table-scientific | yaml)");
    bool        binaryFormat      = false;              clp.setOption("binary", "ascii",       &binaryFormat,      "read matrices in binary format");
    std::string rowMapFile;                             clp.setOption("rowmap",                &rowMapFile,        "map data file");
    std::string colMapFile;                             clp.setOption("colmap",                &colMapFile,        "colmap data file");
    std::string domainMapFile;                          clp.setOption("domainmap",             &domainMapFile,     "domainmap data file");
    std::string rangeMapFile;                           clp.setOption("rangemap",              &rangeMapFile,      "rangemap data file");
    std::string matrixFile;                             clp.setOption("matrix",                &matrixFile,        "matrix data file");
    std::string rhsFile;                                clp.setOption("rhs",                   &rhsFile,           "rhs data file");
    std::string coordFile;                              clp.setOption("coords",                &coordFile,         "coordinates data file");
    std::string coordMapFile;                           clp.setOption("coordsmap",             &coordMapFile,      "coordinates map data file");
    std::string nullFile;                               clp.setOption("nullspace",             &nullFile,          "nullspace data file");
    std::string materialFile;                           clp.setOption("material",              &materialFile,      "material data file");
    int         numVectors        = 1;                  clp.setOption("multivector",           &numVectors,        "number of rhs to solve simultaneously");
    int         numSolves         = 1;                  clp.setOption("numSolves",             &numSolves,         "number of times the system should be solved");
    bool useRACEreordering = false; clp.setOption("useRACEreordering", "no-useRACEreordering", &useRACEreordering, "Use RACE to accelerate the smoothers");
    double RACE_cacheSize = -1; clp.setOption("RACE_cacheSize", &RACE_cacheSize, "Cache size used for RACE reordering");
    int  RACE_highestPower = 1; clp.setOption("RACE_highestPower", &RACE_highestPower, "Highest power used for RACE reordering");


    switch (clp.parse(argc,argv)) {
      case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:        return EXIT_SUCCESS;
      case Teuchos::CommandLineProcessor::PARSE_ERROR:
      case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION: return EXIT_FAILURE;
      case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:          break;
    }

    RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    Teuchos::FancyOStream& out = *fancy;
    out.setOutputToRootOnly(0);

    // Set up timers
    Teuchos::RCP<Teuchos::StackedTimer> stacked_timer;
    if (use_stacked_timer)
      stacked_timer = rcp(new Teuchos::StackedTimer("Main"));
    TimeMonitor::setStackedTimer(stacked_timer);

    // Read in parameter list
    TEUCHOS_TEST_FOR_EXCEPTION(xmlFileName == "" && yamlFileName == "", std::runtime_error,
                               "Need to provide xml or yaml input file");
    RCP<ParameterList> paramList = rcp(new ParameterList("params"));
    if (yamlFileName != "")
      Teuchos::updateParametersFromYamlFileAndBroadcast(yamlFileName, paramList.ptr(), *comm);
    else
      Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, paramList.ptr(), *comm);

    paramList->print();
    //
    // Construct the problem
    //

    RCP<Matrix>                A;
    RCP<const Map>             map;
    RCP<RealValuedMultiVector> coordinates;
    RCP<MultiVector>           nullspace;
    RCP<MultiVector>           material;
    RCP<MultiVector>           X, B_orig, B, X_soln;

    std::ostringstream galeriStream;
    MatrixLoad<SC,LocalOrdinal,GlobalOrdinal,Node>(comm,lib,binaryFormat,matrixFile,rhsFile,rowMapFile,colMapFile,domainMapFile,rangeMapFile,coordFile,coordMapFile,nullFile,materialFile,map,A,coordinates,nullspace,material,X_soln,B_orig,numVectors,matrixParameters,xpetraParameters,galeriStream);
    out << galeriStream.str();
    X = MultiVectorFactory::Build(map, numVectors);
    X->putScalar(0);
    //
    // Build Thyra linear algebra objects
    //

    RCP<Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tpCrs_A = MueLu::Utilities<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Op2NonConstTpetraCrs(A);

    RCP<Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tpRACE_A;
    using RACE_type = RACE::frontend<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    std::string smootherType = "NONE";
    RCP<ParameterList> smootherParamPreList = rcp(new ParameterList("params"));
    RCP<ParameterList> smootherParamPostList = rcp(new ParameterList("params"));
    if (paramList->isSublist("Preconditioner Types") && paramList->sublist("Preconditioner Types").isSublist("MueLu") )
    {
        smootherType = paramList->sublist("Preconditioner Types").sublist("MueLu").get("smoother: type", "NONE");
        if(smootherType == "NONE")
        {
            std::string smootherType_pre = paramList->sublist("Preconditioner Types").sublist("MueLu").get("smoother: pre type", "NONE");
            std::string smootherType_post = paramList->sublist("Preconditioner Types").sublist("MueLu").get("smoother: post type", "NONE");

            if(smootherType_pre != smootherType_post)
            {
                printf("Not employing RACE. Currently we employ RACE only if pre and post smoothers are same\n");
            }
            else
            {
                smootherType = smootherType_pre;
            }

        }
        printf("smootherType = %s\n", smootherType.c_str());
        if(paramList->sublist("Preconditioner Types").sublist("MueLu").isSublist("smoother: params"))
        {
            (*smootherParamPreList) = paramList->sublist("Preconditioner Types").sublist("MueLu").sublist("smoother: params");
            (*smootherParamPostList) = paramList->sublist("Preconditioner Types").sublist("MueLu").sublist("smoother: params");
        }
        else if(paramList->sublist("Preconditioner Types").sublist("MueLu").isSublist("smoother: pre params"))
        {
            (*smootherParamPreList) = paramList->sublist("Preconditioner Types").sublist("MueLu").sublist("smoother: pre params");
            if(paramList->sublist("Preconditioner Types").sublist("MueLu").isSublist("smoother: post params"))
            {
                (*smootherParamPostList) = paramList->sublist("Preconditioner Types").sublist("MueLu").sublist("smoother: post params");
            }

        }
    }

    Teuchos::RCP<RACE_type> race;
    void* raceVoidHandle = NULL;
    int raceTunedPow = 1;

    //bool RACEswitchOff = true;;
    bool RACEswitchOff = true;
    std::string RACE_precon_type = "NONE";

    if(useRACEreordering)
    {
        ParameterList RACE_params("RACE");
        //currently we assume pre and post smoother are the same, only direction
        //can change independently
        std::string smootherSubType = smootherParamPreList->get("relaxation: type", "NONE");
        if(stringToUpper(smootherType) != "NONE")
        {
            if(stringToUpper(smootherType) == "RELAXATION")
            {

                if(stringToUpper(smootherSubType) == "TWO-STAGE GAUSS-SEIDEL")
                {
                    RACE_precon_type = "TWO-STEP-GAUSS-SEIDEL";
                    int smootherInnerSweep_default = 1;
                    int smootherInnerSweep = smootherParamPreList->get("relaxation: inner sweeps", smootherInnerSweep_default);
                    RACE_params.set("Inner iteration", smootherInnerSweep);
                    double gamma_default = 1;
                    double gamma = smootherParamPreList->get("relaxation: inner damping factor", gamma_default);
                    RACE_params.set("Inner damping", gamma);
                    bool smoother_pre_dir = smootherParamPreList->get("relaxation: backward mode", false);
                    bool smoother_post_dir = smootherParamPostList->get("relaxation: backward mode", false);
                    RACE_params.set("Pre-smoother direction", smoother_pre_dir);
                    RACE_params.set("Post-smoother direction", smoother_post_dir);
                    int smootherOuterSweep_default = 1;
                    int smootherOuterSweep = smootherParamPreList->get("relaxation: sweeps", smootherOuterSweep_default);
                    RACE_params.set("Outer iteration", smootherOuterSweep);
                    RACEswitchOff = false;
                }
            }
            else if(stringToUpper(smootherType) == "CHEBYSHEV")
            {
                RACE_precon_type = "CHEBYSHEV";
                /*Ifpack2 ignores min and accepts only max and ratio, so we
                 * mimic that
                 * double lambdaMin = std::nan("");
                lambdaMin = smootherParamPreList->get("chebyshev: min eigenvalue", lambdaMin);*/
                double lambdaMax = std::nan("");
                if(smootherParamPreList->isParameter("chebyshev: max eigenvalue"))
                {
                    lambdaMax = smootherParamPreList->get("chebyshev: max eigenvalue", lambdaMax);
                    RACE_params.set("max eigenvalue", lambdaMax);
                }
                double eigRatio = 20; //default in Ifpack2
                if(smootherParamPreList->isParameter("chebyshev: ratio eigenvalue"))
                {
                    eigRatio = smootherParamPreList->get("chebyshev: ratio eigenvalue", eigRatio);
                }
                RACE_params.set("ratio eigenvalue", eigRatio);
                RACE_params.set("min eigenvalue", lambdaMax/eigRatio);
                int smootherOuterSweep_default = 1;
                int smootherOuterSweep = smootherParamPreList->get("chebyshev: degree", smootherOuterSweep_default);
                RACE_params.set("Outer iteration", smootherOuterSweep);
                RACEswitchOff = false;
            }
        }

        if(RACEswitchOff)
        {
            printf("RACE does not support %s:%s smoother. Switching off RACE\n", smootherType.c_str(), smootherSubType.c_str());
            useRACEreordering=false;
        }

        //cache size in MB
        RACE_params.set("Cache size", RACE_cacheSize);
        RACE_params.set("Highest power", RACE_highestPower);
        RACE_params.set("Preconditioner", RACE_precon_type);
     // paramList->setParameters(RACE_params); //push RACE parameters to the pramList
//#ifdef BELOS_TEUCHOS_TIME_MONITOR
        Teuchos::RCP<Teuchos::Time> RACEPreTime;
        RACEPreTime = Teuchos::TimeMonitor::getNewCounter("Total RACE pre-procesing time");
//#endif
        {
//#ifdef BELOS_TEUCHOS_TIME_MONITOR
            Teuchos::TimeMonitor updateTimer( *RACEPreTime);
//#endif


            race = Teuchos::RCP<RACE_type>(new RACE_type(tpCrs_A, RACE_params));
            tpRACE_A = race->getPermutedMatrix();
            raceVoidHandle = (void*)(race.getRawPtr());

        }

        using Tpetra_MV = Tpetra::MultiVector<>;
        //#ifdef BELOS_TEUCHOS_TIME_MONITOR
        Teuchos::RCP<Teuchos::Time> RACETuningTime;
        RACETuningTime = Teuchos::TimeMonitor::getNewCounter("Total RACE tuning time");
//#endif
        {
//#ifdef BELOS_TEUCHOS_TIME_MONITOR
            Teuchos::TimeMonitor updateTimer( *RACETuningTime);
//#endif
            //get tuned power size for GmresSstep
            RCP<Tpetra_MV> test_x, test_b, test_r;
            test_x = Teuchos::rcp (new Tpetra_MV(tpCrs_A->getRangeMap(), 1));
            test_b = Teuchos::rcp (new Tpetra_MV(tpCrs_A->getRangeMap(), 1));
            test_r = Teuchos::rcp (new Tpetra_MV(tpCrs_A->getRangeMap(), 1));
            std::vector<std::complex<double>> theta(RACE_highestPower, -1);
            raceTunedPow = race->apply_Smoother(RACE_highestPower, *test_x, *test_b, *test_r, false, true, -1);
            //raceTunedPow = race->apply(highestPower, *test, 1, 0, -1);
            printf("tuned pow = %d\n", raceTunedPow);
        }

        RACE_params.set("RACE void handle", raceVoidHandle);
        RACE_params.set("Use RACE", useRACEreordering);
        //RACE_params.set("Use RACE", false);
        RACE_params.set("RACE tuned power", raceTunedPow);
        paramList->sublist("Preconditioner Types").sublist("MueLu").set("RACE: params", RACE_params);

        //for getting eigenvalues
        paramList->sublist("Preconditioner Types").sublist("MueLu").sublist("smoother: params").set("RACE: params", RACE_params);

        B = MultiVectorFactory::Build(B_orig->getMap(), B_orig->getNumVectors());
        Tpetra_MV B_orig_tpetra = Xpetra::toTpetra(*B_orig);
        Tpetra_MV B_tpetra = Xpetra::toTpetra(*B);
        race->origToPerm(B_tpetra, B_orig_tpetra);
    }
    else
    {
        tpRACE_A = tpCrs_A;
        B = B_orig;
    }

    RCP<Xpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Axt = rcp(new Xpetra::TpetraCrsMatrix<SC,LO,GO,NO>(tpRACE_A));
    RCP<const Xpetra::CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node> > xpCrsA = rcp(new Xpetra::CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node> (Axt));
    RCP<const Thyra::LinearOpBase<Scalar> >    thyraA = Xpetra::ThyraUtils<Scalar,LocalOrdinal,GlobalOrdinal,Node>::toThyra(xpCrsA->getCrsMatrix());
    RCP<      Thyra::MultiVectorBase<Scalar> > thyraX = Teuchos::rcp_const_cast<Thyra::MultiVectorBase<Scalar> >(Xpetra::ThyraUtils<Scalar,LocalOrdinal,GlobalOrdinal,Node>::toThyraMultiVector(X));
    RCP<const Thyra::MultiVectorBase<Scalar> > thyraB = Xpetra::ThyraUtils<Scalar,LocalOrdinal,GlobalOrdinal,Node>::toThyraMultiVector(B);


    //
    // Build Stratimikos solver
    //

    // This is the Stratimikos main class (= factory of solver factory).
    Stratimikos::LinearSolverBuilder<Scalar> linearSolverBuilder;
    // Register MueLu as a Stratimikos preconditioner strategy.
    Stratimikos::enableMueLu<Scalar,LocalOrdinal,GlobalOrdinal,Node>(linearSolverBuilder);
#ifdef HAVE_MUELU_IFPACK2
    // Register Ifpack2 as a Stratimikos preconditioner strategy.
    typedef Thyra::PreconditionerFactoryBase<Scalar> Base;
    typedef Thyra::Ifpack2PreconditionerFactory<Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Impl;
    linearSolverBuilder.setPreconditioningStrategyFactory(Teuchos::abstractFactoryStd<Base, Impl>(), "Ifpack2");
#endif

    // add coordinates and nullspace to parameter list
    if (paramList->isSublist("Preconditioner Types") &&
        paramList->sublist("Preconditioner Types").isSublist("MueLu")) {
        ParameterList& userParamList = paramList->sublist("Preconditioner Types").sublist("MueLu").sublist("user data");
        //userParamList.set<RCP<RealValuedMultiVector> >("Coordinates", coordinates);//Cheking what happens with pure AMG
        //userParamList.set<RCP<MultiVector> >("Nullspace", nullspace);
      }

    // Setup solver parameters using a Stratimikos parameter list.
    linearSolverBuilder.setParameterList(paramList);

    // Build a new "solver factory" according to the previously specified parameter list.
    RCP<Thyra::LinearOpWithSolveFactoryBase<Scalar> > solverFactory = Thyra::createLinearSolveStrategy(linearSolverBuilder);
    auto precFactory = solverFactory->getPreconditionerFactory();
    RCP<Thyra::PreconditionerBase<Scalar> > prec;
    Teuchos::RCP<Thyra::LinearOpWithSolveBase<Scalar> > thyraInverseA;
    if (!precFactory.is_null()) {
      prec = precFactory->createPrec();

      // Build a Thyra operator corresponding to A^{-1} computed using the Stratimikos solver.
      Thyra::initializePrec<Scalar>(*precFactory, thyraA, prec.ptr());
      thyraInverseA = solverFactory->createOp();
      Thyra::initializePreconditionedOp<Scalar>(*solverFactory, thyraA, prec, thyraInverseA.ptr());
    } else {
      thyraInverseA = Thyra::linearOpWithSolve(*solverFactory, thyraA);
    }

    Teuchos::RCP<Teuchos::Time> solveTime;
    solveTime = Teuchos::TimeMonitor::getNewCounter("Pure solve time");
    Thyra::SolveStatus<Scalar> status;
    {
        Teuchos::TimeMonitor updateTimer( *solveTime);
        // Solve Ax = b.
        status = Thyra::solve<Scalar>(*thyraInverseA, Thyra::NOTRANS, *thyraB, thyraX.ptr());
        success = (status.solveStatus == Thyra::SOLVE_STATUS_CONVERGED);

        for (int solveno = 1; solveno < numSolves; solveno++) {
            if (!precFactory.is_null())
                Thyra::initializePrec<Scalar>(*precFactory, thyraA, prec.ptr());
            thyraX->assign(0.);

            status = Thyra::solve<Scalar>(*thyraInverseA, Thyra::NOTRANS, *thyraB, thyraX.ptr());

            success = success && (status.solveStatus == Thyra::SOLVE_STATUS_CONVERGED);
        }
    }

    //find convergence and others
    Teuchos::Array<typename STS::magnitudeType> norm_vec(numVectors);
    STS::magnitudeType err_norm=0, res_norm=0, b_norm=0, x_norm;
    RCP<MultiVector> res = Utilities::Residual(*A, *X, *B);
    res->norm2(norm_vec);
    for(int j=0; j<numVectors; ++j)
    {
        res_norm = norm_vec[j] > res_norm ? norm_vec[j]:res_norm;
    }
    X_soln->update(1.0, *X, -1.0);
    X_soln->norm2(norm_vec);
    for(int j=0; j<numVectors; ++j)
    {
        err_norm = norm_vec[j] > err_norm ? norm_vec[j]:err_norm;
    }
    B->norm2(norm_vec);
    for(int j=0; j<numVectors; ++j)
    {
        b_norm = norm_vec[j] > b_norm ? norm_vec[j]:b_norm;
    }
    X->norm2(norm_vec);
    for(int j=0; j<numVectors; ++j)
    {
        x_norm = norm_vec[j] > x_norm ? norm_vec[j]:x_norm;
    }

    using std::cout;
    using std::endl;
    std::string str = status.message;
    std::string startDelim = "in";
    std::string endDelim = "iterations with total";
    unsigned first = str.find(startDelim);
    //now get the substring after start
    std::string secStr = str.substr(first+startDelim.size());
    unsigned last = secStr.find(endDelim);
    std::string iterStr = secStr.substr (0,last);
    cout << "Results:" << endl
        << "  Converged: " << (success ? "true" : "false") << endl
        << "  Number of iterations: " << atoi(iterStr.c_str()) << endl
        << "  Achieved tolerance: " << status.achievedTol << endl
     //   << "  Loss of accuracy: " << status.lossOfAccuracy << endl
        << "  ||B-A*X||_2: " << res_norm << endl
        << "  ||X-X_soln||_2: " << err_norm << endl
        << "  ||B||_2: " << b_norm << endl
        << "  ||X||_2: " << x_norm << endl;
    if (b_norm != Kokkos::ArithTraits<STS::magnitudeType>::zero ()) {
        cout << "  ||B-A*X||_2 / ||B||_2: " << (res_norm / b_norm)
            << endl;
    }
    cout << endl;

    // print timings
    if (printTimings) {
      if (use_stacked_timer) {
        stacked_timer->stop("Main");
        Teuchos::StackedTimer::OutputOptions options;
        options.output_fraction = options.output_histogram = options.output_minmax = true;
        stacked_timer->report(out, comm, options);
      } else {
        RCP<ParameterList> reportParams = rcp(new ParameterList);
        if (timingsFormat == "yaml") {
          reportParams->set("Report format",             "YAML");            // "Table" or "YAML"
          reportParams->set("YAML style",                "compact");         // "spacious" or "compact"
        }
        reportParams->set("How to merge timer sets",   "Union");
        reportParams->set("alwaysWriteLocal",          false);
        reportParams->set("writeGlobalStats",          true);
        reportParams->set("writeZeroTimers",           false);

        const std::string filter = "";

        std::ios_base::fmtflags ff(out.flags());
        if (timingsFormat == "table-fixed") out << std::fixed;
        else                                out << std::scientific;
        TimeMonitor::report(comm.ptr(), out, filter, reportParams);
        out << std::setiosflags(ff);
      }
    }

    TimeMonitor::clearCounters();
    out << std::endl;

  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}


//- -- --------------------------------------------------------
#define MUELU_AUTOMATIC_TEST_ETI_NAME main_
#include "MueLu_Test_ETI.hpp"

int main(int argc, char *argv[]) {
  return Automatic_Test_ETI(argc,argv);
}
