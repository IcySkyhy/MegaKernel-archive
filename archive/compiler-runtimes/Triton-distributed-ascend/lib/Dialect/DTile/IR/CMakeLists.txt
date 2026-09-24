add_triton_library(DTileIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  DTileTableGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
