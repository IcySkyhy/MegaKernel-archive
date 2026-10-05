//===- SplitLoopTransformOps.h - Split-loop schedule as transform ops -----===//
//
// See SplitLoopTransformOps.td.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEXLA_SPLIT_LOOP_TRANSFORM_OPS_H
#define ENZYMEXLA_SPLIT_LOOP_TRANSFORM_OPS_H

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/TransformOps/SplitLoopTransformOps.h.inc"

namespace mlir {
namespace enzyme {
// Registers the split-loop transform ops, and the upstream transform.tune extension that the
// split-loop schedules use for their knobs.
void registerSplitLoopTransformExtension(mlir::DialectRegistry &registry);
} // namespace enzyme
} // namespace mlir

#endif // ENZYMEXLA_SPLIT_LOOP_TRANSFORM_OPS_H
