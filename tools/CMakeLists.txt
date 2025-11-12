# Add a C++ executable.
add_executable(analysis analysis.cpp)

# Link your executable to MLIR libraries.
target_link_libraries(analysis PUBLIC
  MLIRIR
  MLIROptLib
  ${dialect_libs}
  ${conversion_libs}
)

# Set up include directories for MLIR headers.
target_include_directories(analysis PUBLIC
  ${MLIR_INCLUDE_DIRS}
)