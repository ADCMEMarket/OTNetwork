add_library(OTNetwork SHARED src/EMD_wrapper.cpp OTNetwork.cpp)
include_directories(src)

set_property(TARGET OTNetwork PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(OTNetwork ${TF_LIB_FILE} "${TORCH_LIBRARIES}")
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set_target_properties(OTNetwork PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
