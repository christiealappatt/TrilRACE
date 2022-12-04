#include "RACE_CRS_raw.hpp"
#include "RACE_frontend.hpp"
#include "RACE_GmresPolyPreconKernel.hpp"
#include "RACE_GmresSstepKernel.hpp"
#include "RACE_kernels.hpp"
#include "RACE_MGSmootherKernel.hpp"
#include "RACE_packtype.hpp"
#include "RACE_Precon.hpp"
#include "RACE_pre_process.hpp"
#include "RACE_SpMV.hpp"

#include <iostream>

namespace TrilinosRACE {

void dummy()
{
  std::cout << "Hello World!" << std::endl;
}

}
