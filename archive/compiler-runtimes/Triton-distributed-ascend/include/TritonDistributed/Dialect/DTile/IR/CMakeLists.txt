set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS DTileOps.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=dtile)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=dtile)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(DTileTableGen)
