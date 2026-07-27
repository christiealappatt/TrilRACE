#ifndef _RACE_FRONTEND_H_
#define _RACE_FRONTEND_H_

// clang-format off
#include "RACE_pre_process.hpp"
#include "RACE_kernels.hpp"
#include "RACE_packtype.hpp"
#include "Tpetra_CrsMatrix.hpp"
#ifdef LIKWID_MG_SMOOTHER
#include <likwid.h>
#include <omp.h>
#endif
#include <sched.h>
// clang-format on
namespace RACE {

// TODO: template on CRS and MV types, so CRS and MV can have different types
// template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class frontend {
  using packtype = RACE_packtype<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using CrsMatrixType = typename packtype::CRS_type;
  using preProcess_type = preProcess<packtype>;
  using kernels_type = kernels<packtype>;

  // Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  preProcess_type pre;
  kernels_type exec;

public:
  // constructor
  frontend(Teuchos::RCP<CrsMatrixType> origA_,
           Teuchos::ParameterList &paramList,
           Teuchos::RCP<CrsMatrixType> M = Teuchos::null)
      : pre(origA_, paramList), exec(&pre) {
    // Teuchos::RCP<CrsMatrixType> permA = pre.getPermutedMatrix();
    // exec.init(&pre);
  }

  Teuchos::RCP<CrsMatrixType> getPermutedMatrix() {
    return pre.getPermutedMatrix();
  }

  int *getPerm() { return pre.getPerm(); }

  int *getInvPerm() { return pre.getInvPerm(); }

  using MV = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  // DL 2026-03-30 TODO: Slow!
  void permToOrig(MV &dest_vec, const MV &src_vec) {
    if (src_vec.getLocalLength() != pre.getNrows()) {
      ERROR_PRINT("Error in dimension");
    }

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<const Scalar>> src_ptr =
        src_vec.get2dView();
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Scalar>> dest_ptr =
        dest_vec.get2dViewNonConst();

    const size_t nrows = src_vec.getLocalLength();
    const size_t nvecs = src_vec.getNumVectors();
#if 1
    const int *invperm = pre.getInvPerm();
    if (invperm) {
      for (size_t k = 0; k < nvecs; k++) {
        const Scalar *__restrict__ src = src_ptr[k].getRawPtr();
        Scalar *__restrict__ dst = dest_ptr[k].getRawPtr();

        const int *__restrict__ ip = invperm;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nrows; i++)
          dst[i] = src[ip[i]];
      }
    }
#elif 0 // Swap gather loads for scatter reads
    const int *perm = pre.getPerm();
    if (perm) {
      for (size_t k = 0; k < nvecs; k++) {
        const Scalar *__restrict__ src = src_ptr[k].getRawPtr();
        Scalar *__restrict__ dst = dest_ptr[k].getRawPtr();
        const int *__restrict__ p = perm;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nrows; i++)
          dst[p[i]] = src[i];
      }
    }
#endif
    else {
      for (size_t k = 0; k < nvecs; k++) {
        const Scalar *__restrict__ src = src_ptr[k].getRawPtr();
        Scalar *__restrict__ dst = dest_ptr[k].getRawPtr();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nrows; i++)
          dst[i] = src[i];
      }
    }
  }

  // DL NOTE: Do I really need two methods for this?
  // DL 2026-03-30 TODO: Slow!
  void origToPerm(MV &dest_vec, const MV &src_vec) {
    if (src_vec.getLocalLength() != pre.getNrows()) {
      ERROR_PRINT("Error in dimension");
    }

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<const Scalar>> src_ptr =
        src_vec.get2dView();
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Scalar>> dest_ptr =
        dest_vec.get2dViewNonConst();

    const size_t nrows = src_vec.getLocalLength();
    const size_t nvecs = src_vec.getNumVectors();

#if 1
    const int *perm = pre.getPerm();
    if (perm) {
      for (size_t k = 0; k < nvecs; k++) {
        const Scalar *__restrict__ src = src_ptr[k].getRawPtr();
        Scalar *__restrict__ dst = dest_ptr[k].getRawPtr();

        const int *__restrict__ p = perm;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nrows; i++)
          dst[i] = src[p[i]];
      }
    }
#elif 0 // Swap gather loads for scatter reads
    const int *invperm = pre.getInvPerm();
    if (invperm) {
      for (size_t k = 0; k < nvecs; k++) {
        const Scalar *__restrict__ src = src_ptr[k].getRawPtr();
        Scalar *__restrict__ dst = dest_ptr[k].getRawPtr();
        const int *__restrict__ ip = invperm;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nrows; i++)
          dst[ip[i]] = src[i];
      }
    }
#endif
    else {
      for (size_t k = 0; k < nvecs; k++) {
        const Scalar *__restrict__ src = src_ptr[k].getRawPtr();
        Scalar *__restrict__ dst = dest_ptr[k].getRawPtr();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nrows; i++)
          dst[i] = src[i];
      }
    }
  }

  using vec_type =
      Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  void updateParamList(Teuchos::ParameterList newParams) {
    pre.updateParamList(newParams);
    exec.paramUptodate = false;
  }

  void setupKernels() { exec.setupParams(); }

  int apply(int power, vec_type &x,
            Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
            Scalar beta = Teuchos::ScalarTraits<Scalar>::zero(),
            int tunedPow = 1) {
    std::string precType = exec.getPrecType();
    if ((precType == "NONE") || (precType == "JACOBI")) {
      return exec.MPK(power, x, alpha, beta, tunedPow);
    } else {
      ERROR_PRINT("MPK with %s preconditioner not implemented yet in RACE",
                  precType.c_str());
      return -2;
    }
  }

  using complex_type = typename packtype::complex_type;

  int apply_Precon(int power, const vec_type &b, vec_type &x,
                   bool fwdDir = true) {
    // timer
    Teuchos::RCP<Teuchos::Time> timer =
        Teuchos::TimeMonitor::getNewCounter("RACE::Prec-apply");
    Teuchos::TimeMonitor LocalTimer(*timer);

    std::string precType = exec.getPrecType();
    if ((precType == "NONE" || precType == "JACOBI") ||
        (precType == "GAUSS-SEIDEL" || precType == "JACOBI-GAUSS-SEIDEL") ||
        (precType == "TWO-STEP-GAUSS-SEIDEL")) {
      return exec.PreconKernel(power, b, x, fwdDir);
    } else {
      ERROR_PRINT("%s preconditioner kernel not implemented yet in RACE",
                  precType.c_str());
      return -2;
    }
  }

  int apply_GmresSstep(int power, int iter, vec_type &x,
                       std::vector<complex_type> theta, int tunedPow = 1) {
    // timer
    Teuchos::RCP<Teuchos::Time> timer =
        Teuchos::TimeMonitor::getNewCounter("RACE::GmresSstep kernel");
    Teuchos::TimeMonitor LocalTimer(*timer);

    std::string precType = exec.getPrecType();
    if ((precType == "NONE" || precType == "JACOBI") ||
        (precType == "GAUSS-SEIDEL" || precType == "JACOBI-GAUSS-SEIDEL") ||
        (precType == "TWO-STEP-GAUSS-SEIDEL")) {
      return exec.MPK_GmresSstepKernel(power, iter, x, theta, tunedPow);
    } else {
      ERROR_PRINT(
          "GMRES-s-step with %s preconditioner not implemented yet in RACE",
          precType.c_str());
      return -2;
    }
  }

  int apply_GmresPolyPrecon(int power, vec_type &prod, vec_type &y,
                            std::vector<complex_type> theta, int tunedPow = 1) {
    // timer
    Teuchos::RCP<Teuchos::Time> timer =
        Teuchos::TimeMonitor::getNewCounter("RACE::GmresPoly kernel");
    Teuchos::TimeMonitor LocalTimer(*timer);

    // step size and use it
    std::string precType = exec.getPrecType();
    if ((precType == "NONE" || precType == "JACOBI") ||
        (precType == "GAUSS-SEIDEL" || precType == "JACOBI-GAUSS-SEIDEL") ||
        (precType == "TWO-STEP-GAUSS-SEIDEL")) {
      return exec.MPK_GmresPolyPreconKernel(power, prod, y, theta, tunedPow);
    } else {
      ERROR_PRINT("GMRES polynomial preconditioner with %s preconditioner not "
                  "implemented yet in RACE",
                  precType.c_str());
      return -2;
    }
  }

  int apply_Smoother(int sweeps, vec_type &x, vec_type &b,
                     bool zeroGuess = false, bool fwdDir = true,
                     int tunedPow = 1) {
    Teuchos::RCP<Teuchos::Time> timer =
        Teuchos::TimeMonitor::getNewCounter("RACE::MGSmoother kernel");
    Teuchos::TimeMonitor LocalTimer(*timer);
#ifdef LIKWID_MG_SMOOTHER
#pragma omp parallel
    {
      LIKWID_MARKER_START("MG_SMOOTH");
    }
#endif
#ifdef DANE_RACE_DEBUG
    printf("[RACE] apply_Smoother: sweeps=%d, precType=%s\n", sweeps,
           exec.getPrecType().c_str());
#endif
    std::string precType = exec.getPrecType();
    int ret;
    if ((precType == "TWO-STEP-GAUSS-SEIDEL") || (precType == "CHEBYSHEV")) {

      ret =
          exec.MPK_MGSmootherKernel(sweeps, x, b, zeroGuess, fwdDir, tunedPow);
    } else {
      ERROR_PRINT(
          "MG Smoother with %s preconditioner not implemented yet in RACE",
          precType.c_str());
      ret = -2;
    }

#ifdef LIKWID_MG_SMOOTHER
#pragma omp parallel
    {
      LIKWID_MARKER_STOP("MG_SMOOTH");
    }
#endif
    return ret;
  }

  // fused with Residual computation
  int apply_Smoother(int sweeps, vec_type &x, vec_type &b, vec_type &res,
                     bool zeroGuess = false, bool fwdDir = true,
                     int tunedPow = 1) {
    // timer
    Teuchos::RCP<Teuchos::Time> timer =
        Teuchos::TimeMonitor::getNewCounter("RACE::MGSmoother+residual kernel");
    Teuchos::TimeMonitor LocalTimer(*timer);

    // step size and use it
    std::string precType = exec.getPrecType();
    if ((precType == "TWO-STEP-GAUSS-SEIDEL") || (precType == "CHEBYSHEV")) {
      return exec.MPK_MGSmootherKernel(sweeps, x, b, res, zeroGuess, fwdDir,
                                       tunedPow);
    } else {
      ERROR_PRINT(
          "MG Smoother with %s preconditioner not implemented yet in RACE",
          precType.c_str());
      return -2;
    }
  }

}; // class frontend
} // namespace RACE

#endif
