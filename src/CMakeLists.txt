# set target
set(ORT_DEPLOY_LIB ${PROJECT_NAME})
add_library(${ORT_DEPLOY_LIB} SHARED)
# add_dependencies(${ORT_DEPLOY_LIB} format)

# find all source files
file(GLOB_RECURSE HCMAKE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
message(STATUS "${CMAKE_CURRENT_SOURCE_DIR} SRCS: ${SRCS}")

# assign target include dirs and sources
target_include_directories(${ORT_DEPLOY_LIB} PUBLIC ${INC_DIR} ${ORT_INC_DIR})
target_sources(${ORT_DEPLOY_LIB} PRIVATE ${HCMAKE_SRCS})

# assign target compile options
target_compile_features(${ORT_DEPLOY_LIB} PRIVATE cxx_std_14)
target_compile_options(${ORT_DEPLOY_LIB} PRIVATE -Wall -Werror -Wextra)

target_link_libraries(${ORT_DEPLOY_LIB} PUBLIC ${ORT_LIB})
