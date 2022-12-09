INCLUDE(TrilinosCreateClientTemplateHeaders)

# Function to generate ETI (explicit template instantiation) files
# from a template and list of class names
#FUNCTION(TrillinosRACE_PROCESS_ETI_TEMPLATE ETI_CLASSES TEMPLATE_FILE PROCESSED_FILE SOURCES_LIST)
#  SET(SRCS "")
#  FOREACH(CLASS ${ETI_CLASSES})
#    string(REPLACE "::" "_" CLASS_FILE_NAME "${CLASS}")
#    string(TOUPPER "${CLASS_FILE_NAME}" UPPER_CASE_CLASS)
#    string(REPLACE "CLASS_FILE_NAME" "${CLASS_FILE_NAME}" FINAL_FILE_NAME "${PROCESSED_FILE}")
#    CONFIGURE_FILE(${TEMPLATE_FILE} ${FINAL_FILE_NAME})
#    SET(SRCS ${SRCS} ${FINAL_FILE_NAME})
#  ENDFOREACH()
#  SET(${SOURCES_LIST} ${SRCS} PARENT_SCOPE)
#ENDFUNCTION(TrillinosRACE_PROCESS_ETI_TEMPLATE)
#
## Function that follows the Tpetra convention for mangling C++ types
## so that they can be used as C preprocessor macro arguments.
#FUNCTION(TrillinosRACE_MANGLE_TEMPLATE_PARAMETER TYPE_MANGLED_OUT TYPE_IN)
#  STRING(REPLACE "<" "0" TMP0 "${TYPE_IN}")
#  STRING(REPLACE ">" "0" TMP1 "${TMP0}")
#  STRING(REPLACE "::" "_" TMP2 "${TMP1}")
#  # Spaces (as in "long long") get squished out.
#  STRING(REPLACE " " "" TMP3 "${TMP2}")
#  SET(${TYPE_MANGLED_OUT} ${TMP3} PARENT_SCOPE)
#ENDFUNCTION(TrillinosRACE_MANGLE_TEMPLATE_PARAMETER)
#
## Function that turns a valid Scalar template parameter into a macro
## name (all caps, with no white space and no punctuation other than
## underscore).
#FUNCTION(TrillinosRACE_SCALAR_MACRO_NAME NAME_OUT NAME_IN)
#  STRING(COMPARE EQUAL "${NAME_IN}" "__float128" IS_FLOAT128)
#  IF(IS_FLOAT128)
#    # __float128 is a special case; we remove the __ from the macro name.
#    SET(${NAME_OUT} "FLOAT128" PARENT_SCOPE)
#  ELSE()
#    STRING(COMPARE EQUAL "${NAME_IN}" "std::complex<float>" IS_COMPLEX_FLOAT)
#    IF(IS_COMPLEX_FLOAT)
#      SET(${NAME_OUT} "COMPLEX_FLOAT" PARENT_SCOPE)
#    ELSE()
#      STRING(COMPARE EQUAL "${NAME_IN}" "std::complex<double>" IS_COMPLEX_DOUBLE)
#      IF(IS_COMPLEX_DOUBLE)
#        SET(${NAME_OUT} "COMPLEX_DOUBLE" PARENT_SCOPE)
#      ELSE()
#>---#long double is a special name; add _ to macro name
#        STRING(COMPARE EQUAL "${NAME_IN}" "long double" IS_LONG_DOUBLE)
#>---IF(IS_LONG_DOUBLE)~
#>---  SET(${NAME_OUT} "LONG_DOUBLE" PARENT_SCOPE)
#        ELSE()
#          # Convert to upper case, convert double colons to underscores,
#          # and hope for the best.
#          #
#          # It would be nice if CMake were consistent about where output
#          # arguments go.  Alas, this is not to be.  TOUPPER puts the
#          # output argument last; REPLACE puts it after the search and
#          # substitute strings, before the input string.
#          STRING(TOUPPER "${NAME_IN}" TMP0)
#          STRING(REPLACE "::" "_" TMP1 "${TMP0}")
#          STRING(REPLACE " " "_" TMP2 "${TMP1}")
#          SET(${NAME_OUT} ${TMP2} PARENT_SCOPE)
#        ENDIF()
#      ENDIF()
#    ENDIF()
#  ENDIF()
#ENDFUNCTION(TrillinosRACE_SCALAR_MACRO_NAME)
#
## Function that turns a valid LocalOrdinal or GlobalOrdinal template
## parameter (which must be a C++ built-in integer type) into a macro
## name (all caps, with no white space and no punctuation other than
## underscore).
#FUNCTION(TrillinosRACE_ORDINAL_MACRO_NAME NAME_OUT NAME_IN)
#  # It would be nice if CMake were consistent about where output
#  # arguments go.  Alas, this is not to be.  TOUPPER puts the
#  # output argument last; REPLACE puts it after the search and
#  # substitute strings, before the input string.
#  #
#  # "long long" -> "LONG LONG"
#  STRING(TOUPPER "${NAME_IN}" TMP0)
#  # "LONG LONG" -> "LONG_LONG"
#  STRING(REPLACE " " "_" TMP1 "${TMP0}")
#  SET(${NAME_OUT} ${TMP1} PARENT_SCOPE)
#ENDFUNCTION(TrillinosRACE_ORDINAL_MACRO_NAME)
#
## Function that turns a valid Node template parameter into a macro
## name (all caps, with no white space and no punctuation other than
## underscore).
#FUNCTION(TrillinosRACE_NODE_MACRO_NAME NAME_OUT NAME_IN)
#  STRING(REGEX MATCH "Kokkos::Compat::Kokkos(.*)WrapperNode" TMP0 "${NAME_IN}")
#  STRING(COMPARE EQUAL "${TMP0}" "" DOES_NOT_MATCH)
#  IF(DOES_NOT_MATCH)
#    MESSAGE(FATAL_ERROR "Ifpack2: Node $NAME_IN is not a supported Node type.")
#  ELSE()
#    # Extract the Kokkos execution space (KOKKOS_EXEC_SPACE) from the Node name.
#    STRING(REGEX REPLACE "Kokkos::Compat::Kokkos(.*)WrapperNode" "\\1" KOKKOS_EXEC_SPACE "${NAME_IN}")
#
#    # Special case: Threads.  The macro name unfortunately differs
#    # from the execution space name in a way that doesn't fit the
#    # pattern of the other execution spaces.
#    STRING(COMPARE EQUAL "${KOKKOS_EXEC_SPACE}" "Threads" IS_THREADS)
#    IF(IS_THREADS)
#      SET(${NAME_OUT} "PTHREAD" PARENT_SCOPE)
#    ELSE()
#      # The other cases (Cuda, HIP, Serial, OpenMP) are easy.
#      STRING(TOUPPER "${KOKKOS_EXEC_SPACE}" NAME_OUT_TMP)
#      SET(${NAME_OUT} ${NAME_OUT_TMP} PARENT_SCOPE)
#    ENDIF()
#  ENDIF()
#ENDFUNCTION(TrillinosRACE_NODE_MACRO_NAME)
#
## Function to generate one .cpp file for the given (Scalar,
## LocalOrdinal, GlobalOrdinal, Node) template parameter combination,
## for run-time registration of Ifpack2's LinearSolverFactory over
## those template parameters.  This is meant to be called by
## TrillinosRACE_PROCESS_ALL_FACTORY_TEMPLATES.  This function takes the
## names already mangled, to avoid unnecessary string processing
## overhead.
#FUNCTION(TrillinosRACE_PROCESS_ONE_FACTORY_TEMPLATE OUTPUT_FILE TEMPLATE_FILE SC LO GO NT SC_MANGLED LO_MANGLED GO_MANGLED NT_MANGLED SC_MACRO_NAME LO_MACRO_NAME GO_MACRO_NAME NT_MACRO_NAME)
#  STRING(REPLACE ".tmpl" "_${SC_MACRO_NAME}_${LO_MACRO_NAME}_${GO_MACRO_NAME}_${NT_MACRO_NAME}.cpp" OUT_FILE "${TEMPLATE_FILE}")
#  CONFIGURE_FILE("${TEMPLATE_FILE}" "${OUT_FILE}")
#
#  SET(${OUTPUT_FILE} ${OUT_FILE} PARENT_SCOPE)
#ENDFUNCTION(TrillinosRACE_PROCESS_ONE_FACTORY_TEMPLATE)
#
## Function to generate .cpp files for run-time registration of
## Ifpack2's LinearSolverFactory, from a template.  We generate one
## .cpp file for each (Scalar, LocalOrdinal, GlobalOrdinal, Node) type
## combination over which Ifpack2 does tests and/or ETI.
#FUNCTION(TrillinosRACE_PROCESS_ALL_FACTORY_TEMPLATES OUTPUT_FILES TEMPLATE_FILE SCALAR_TYPES LOCALORDINAL_TYPES GLOBALORDINAL_TYPES NODE_TYPES)
#  SET(OUT_FILES "")
#  FOREACH(NT ${NODE_TYPES})
#    TrillinosRACE_MANGLE_TEMPLATE_PARAMETER(NT_MANGLED "${NT}")
#    TrillinosRACE_NODE_MACRO_NAME(NT_MACRO_NAME "${NT}")
#    FOREACH(GO ${GLOBALORDINAL_TYPES})
#      TrillinosRACE_MANGLE_TEMPLATE_PARAMETER(GO_MANGLED "${GO}")
#      TrillinosRACE_ORDINAL_MACRO_NAME(GO_MACRO_NAME "${GO}")
#      FOREACH(LO ${LOCALORDINAL_TYPES})
#        TrillinosRACE_MANGLE_TEMPLATE_PARAMETER(LO_MANGLED "${LO}")
#        TrillinosRACE_ORDINAL_MACRO_NAME(LO_MACRO_NAME "${LO}")
#        FOREACH(SC ${SCALAR_TYPES})
#          TrillinosRACE_MANGLE_TEMPLATE_PARAMETER(SC_MANGLED "${SC}")
#          TrillinosRACE_SCALAR_MACRO_NAME(SC_MACRO_NAME "${SC}")
#          TrillinosRACE_PROCESS_ONE_FACTORY_TEMPLATE(OUT_FILE "${TEMPLATE_FILE}" "${SC}" "${LO}" "${GO}" "${NT}" "${SC_MANGLED}" "${LO_MANGLED}" "${GO_MANGLED}" "${NT_MANGLED}" "${SC_MACRO_NAME}" "${LO_MACRO_NAME}" "${GO_MACRO_NAME}"           "${NT_MACRO_NAME}")
#          LIST(APPEND OUT_FILES ${OUT_FILE})
#        ENDFOREACH() # SC
#      ENDFOREACH() # LO
#    ENDFOREACH() # GO
#  ENDFOREACH() # NT
#
#  # This is the standard CMake idiom for setting an output variable so
#  # that the caller can see the result.
#  SET(${OUTPUT_FILES} ${OUT_FILES} PARENT_SCOPE)
#ENDFUNCTION(TrillinosRACE_PROCESS_ALL_FACTORY_TEMPLATES)
#

#
# A) Package-specific configuration options
#

include(${PROJECT_SOURCE_DIR}/packages/${PACKAGE_NAME}/cmake/OptimizeForArchitecture.cmake)

AutodetectHostArchitecture()
OptimizeForArchitecture()

#Set appropriate vector length
if(USE_AVX512F)
    set(VECTOR_LENGTH 8)
elseif(USE_AVX OR USE_AVX2)
    set(VECTOR_LENGTH 4)
elseif(USE_SSE2 OR USE_SSE3 OR SSE4_1 OR SSE4_2 OR SSE_4a)
    set(VECTOR_LENGTH 2)
else()
    set(VECTOR_LENGTH 1)
endif()

TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.h)

SET(HEADERS "")
SET(SOURCES "dummy.cpp")

SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR})
APPEND_GLOB(HEADERS ${DIR}/*.hpp)
APPEND_GLOB(SOURCES ${DIR}/*.cpp)
TRILINOS_CREATE_CLIENT_TEMPLATE_HEADERS(${DIR})


#find installed RACE library; required for building this package
find_package(RACE REQUIRED CONFIG)
message(STATUS "Found RACE")

#link_libraries(${RACE_LIBRARIES})
SET(TrilinosRACE_LIBRARY_DIRS "${TrilinosRACE_LIBRARY_DIRS}; ${RACE_LIBRARIES}")
SET_AND_INC_DIRS(DIR "${RACE_INCLUDE_DIRS}")
link_libraries(${RACE_LIBRARIES})
# Must glob the binary dir last to get all of the auto-generated headers
SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_BINARY_DIR})
APPEND_GLOB(HEADERS ${DIR}/*.hpp)
APPEND_SET(HEADERS ${DIR}/${PACKAGE_NAME}_config.h )
#APPEND_SET(HEADERS ${DIR}/${PACKAGE_NAME}_ETIHelperMacros.h )


TRIBITS_ADD_LIBRARY(
  TrillinosRACE
  HEADERS ${HEADERS}
  SOURCES ${SOURCES} #${TrillinosRACE_CPP_SOURCES}
  )
