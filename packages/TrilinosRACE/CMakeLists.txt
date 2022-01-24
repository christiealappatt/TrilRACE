#
# Define the package
#
TRIBITS_PACKAGE( TrilinosRACE )

# If deprecated warnings are on, and the compiler supports them, then
# define IFPACK2_DEPRECATED to emit deprecated warnings.  Otherwise,
# give it an empty definition.
TRIBITS_ADD_SHOW_DEPRECATED_WARNINGS_OPTION()

# Make sure that Tpetra actually defined these variables, even if they
# are empty.
#ASSERT_DEFINED(TpetraCore_ETI_SCALARS)
#ASSERT_DEFINED(TpetraCore_ETI_LORDS)
#ASSERT_DEFINED(TpetraCore_ETI_GORDS)
#ASSERT_DEFINED(TpetraCore_ETI_NODES)
#
#SET(TrilinosRACE_ETI_SCALARS ${TpetraCore_ETI_SCALARS})
#SET(TrilinosRACE_ETI_LORDS   ${TpetraCore_ETI_LORDS})
#SET(TrilinosRACE_ETI_GORDS   ${TpetraCore_ETI_GORDS})
#SET(TrilinosRACE_ETI_NODES   ${TpetraCore_ETI_NODES})
#
TRIBITS_ADD_OPTION_AND_DEFINE(
  ${PACKAGE_NAME}_Trilinos
  HAVE_${PACKAGE_NAME}_Trilinos
  "Building as a Trilinos package"
  ON
  )

TRIBITS_ADD_OPTION_AND_DEFINE(
  ${PACKAGE_NAME}_ENABLE_DEBUG
  HAVE_${PACKAGE_NAME}_DEBUG
  "Enable runtime debug checking."
  ${${PROJECT_NAME}_ENABLE_DEBUG}
  )

TRIBITS_ADD_EXPLICIT_INSTANTIATION_OPTION()

# we use this for testing as well as library eti, so do it regardless
#TRIBITS_ADD_ETI_SUPPORT()

ADD_SUBDIRECTORY(src)

# Promote these variables to be visible outside of the "scope" of this
# directory -- e.g., to other packages.
#GLOBAL_SET(TrilinosRACE_ETI_SCALARS ${TrilinosRACE_ETI_SCALARS})
#GLOBAL_SET(TrilinosRACE_ETI_LORDS   ${TrilinosRACE_ETI_LORDS})
#GLOBAL_SET(TrilinosRACE_ETI_GORDS   ${TrilinosRACE_ETI_GORDS})
#GLOBAL_SET(TrilinosRACE_ETI_NODES   ${TrilinosRACE_ETI_NODES})
#

TRIBITS_PACKAGE_POSTPROCESS()
