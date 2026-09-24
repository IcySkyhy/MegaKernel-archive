/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "TritonDistributed/Conversion/TritonDistributedToHIVM/TritonDistributedToHIVMPass.h"
#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"
#include "TritonDistributed/Dialect/DTile/IR/Dialect.h"
#include "ascend/include/Utils/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include <string>
#include <type_traits>

using namespace mlir;
using namespace mlir::triton;

namespace {

/// Generic template pattern to convert distributed ops to hivm.custom
/// The callee name is derived from the distributed op name with a prefix
template <typename DistOp>
class DistributedOpToHIVM : public OpRewritePattern<DistOp> {
public:
  using OpAdaptor = typename DistOp::Adaptor;

  DistributedOpToHIVM(MLIRContext *context, bool existDot = false,
                      StringRef customPrefix = "",
                      PatternBenefit benefit = PatternBenefit(1))
      : OpRewritePattern<DistOp>(context, benefit), existDotFlag(existDot),
        customPrefix(customPrefix) {}

  void inferCoreType(DistOp op) const {
    if constexpr (std::is_same_v<DistOp, distributed::ExternCallOp>) {
      auto externCallOp = llvm::cast<distributed::ExternCallOp>(op);
      if (llvm::is_contained(ASCEND::vecTypeList, externCallOp.getSymbol())) {
        op->setAttr(hivm::TCoreTypeAttr::name,
                    hivm::TCoreTypeAttr::get(op->getContext(),
                                             hivm::TCoreType::VECTOR));
        return;
      }
      if (llvm::is_contained(ASCEND::mixTypeList, externCallOp.getSymbol())) {
        op->setAttr(hivm::TCoreTypeAttr::name,
                    hivm::TCoreTypeAttr::get(op->getContext(),
                                             hivm::TCoreType::CUBE_AND_VECTOR));
        return;
      }
    }
    auto coreType = existDotFlag ? hivm::TCoreType::CUBE_AND_VECTOR
                                 : hivm::TCoreType::VECTOR;
    op->setAttr(hivm::TCoreTypeAttr::name,
                hivm::TCoreTypeAttr::get(op->getContext(), coreType));
  }

  LogicalResult getTypeName(Type type, std::string &typeName) const {
    if (auto fpTy = llvm::dyn_cast<FloatType>(type)) {
      if (fpTy.isBF16()) {
        typeName = "bfloat16";
        return success();
      } else if (fpTy.isF16()) {
        typeName = "half";
        return success();
      } else if (fpTy.isF32()) {
        typeName = "float";
        return success();
      }
    }
    if (auto intTy = llvm::dyn_cast<IntegerType>(type)) {
      switch (intTy.getWidth()) {
      case 8:
        typeName = "int8";
        return success();
      case 16:
        typeName = "int16";
        return success();
      case 32:
        typeName = "int32";
        return success();
      case 64:
        typeName = "int64";
        return success();
      }
    }
    if (auto pointerTy = llvm::dyn_cast<PointerType>(type)) {
      std::string pointeeTypeName;
      if (failed(getTypeName(pointerTy.getPointeeType(), pointeeTypeName))) {
        return failure();
      }
      typeName = pointeeTypeName + "_ptr_1d";
      return success();
    }
    if (auto tensorTy = llvm::dyn_cast<RankedTensorType>(type)) {
      auto pointerTy = llvm::dyn_cast<PointerType>(tensorTy.getElementType());
      if (!pointerTy) {
        return failure();
      }
      std::string pointeeTypeName;
      if (failed(getTypeName(pointerTy.getPointeeType(), pointeeTypeName))) {
        return failure();
      }
      typeName =
          pointeeTypeName + "_ptr_" + std::to_string(tensorTy.getRank()) + "d";
      return success();
    }
    return failure();
  }

  LogicalResult matchAndRewrite(DistOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Validate before mutating anything so a failed match leaves the IR intact.
    if constexpr (std::is_same_v<DistOp, distributed::NotifyOp>) {
      // aclshmemx_signal_op only supports an `int32_t *` signal ptr.
      auto sigPtrTy =
          llvm::cast<triton::PointerType>(op.getSigAddr().getType());
      auto sigElemTy = llvm::dyn_cast<IntegerType>(sigPtrTy.getPointeeType());
      if (!sigElemTy || sigElemTy.getWidth() != 32) {
        return rewriter.notifyMatchFailure(
            op, "aclshmemx_signal_op requires a 32-bit integer signal pointer");
      }
    }

    // DcciOp lowers directly to hivm::DCCIOp
    if (auto dcciOp = llvm::dyn_cast<dtile::DcciOp>(&op)) {
      auto extractConstInt = [](Value v) -> int32_t {
        if (auto constOp = v.getDefiningOp<mlir::arith::ConstantOp>())
          return static_cast<int32_t>(
              mlir::cast<IntegerAttr>(constOp.getValue()).getInt());
        return 0;  // fallback if not a constant
      };
      int32_t modeVal = extractConstInt(dcciOp->getMode());
      int32_t dataCacheKindVal = extractConstInt(dcciOp->getDataCacheKind());
      hivm::DCCIMode hivmMode =
          (modeVal == 1) ? hivm::DCCIMode::ALL_CACHE_LINES
                         : hivm::DCCIMode::SINGLE_CACHE_LINE;
      hivm::DataCacheKind hivmDCK;
      switch (dataCacheKindVal) {
      case 0: hivmDCK = hivm::DataCacheKind::ALL; break;
      case 1: hivmDCK = hivm::DataCacheKind::UB; break;
      case 2: hivmDCK = hivm::DataCacheKind::OUT; break;
      case 3: hivmDCK = hivm::DataCacheKind::ATOMIC; break;
      default: hivmDCK = hivm::DataCacheKind::OUT; break;
      }
      auto modeAttr = hivm::DCCIModeAttr::get(op->getContext(), hivmMode);
      auto dckAttr = hivm::DataCacheKindAttr::get(op->getContext(), hivmDCK);
      // ptr is Optional<AnyMemRef>. Pass nullptr for now (TODO: convert triton
      // pointer to memref when needed).
      rewriter.create<hivm::DCCIOp>(loc, modeAttr, dckAttr, /*ptr=*/Value());
      rewriter.eraseOp(op);
      return success();
    }

    inferCoreType(op);
    std::string symbolName;
    if (auto sym = op->template getAttrOfType<StringAttr>("symbol")) {
      symbolName = sym.str();
    }
    bool hasSideEffect = !mlir::isMemoryEffectFree(op.getOperation());
    bool unsupportedTypeForSymbol = false;

    llvm::TypeSwitch<Operation *, void>(op)
        .Case([&](distributed::SymmAtOp) {
          auto elemTy =
              llvm::cast<triton::PointerType>(op->getOperand(0).getType())
                  .getPointeeType();
          std::string typeName;
          if (failed(getTypeName(elemTy, typeName))) {
            unsupportedTypeForSymbol = true;
            return;
          }
          symbolName = "aclshmem_ptr_" + typeName;
        })
        .Case([&](distributed::GetRankOp) { symbolName = "aclshmem_my_pe"; })
        .Case(
            [&](distributed::GetNumRanksOp) { symbolName = "aclshmem_n_pes"; })
        .Case(
            [&](distributed::NotifyOp) { symbolName = "aclshmemx_signal_op"; })
        .Case([&](distributed::ConsumeTokenOp consumeTokenOp) {
          std::string typeName;
          if (failed(
                  getTypeName(consumeTokenOp.getInput().getType(), typeName))) {
            unsupportedTypeForSymbol = true;
            return;
          }
          symbolName = "aclshmem_consume_token_" + typeName;
        })
        .Case([&](distributed::WaitOp waitOp) {
          std::string typeName;
          if (failed(getTypeName(
                  llvm::cast<PointerType>(waitOp.getBarrierPtr().getType())
                      .getPointeeType(),
                  typeName))) {
            unsupportedTypeForSymbol = true;
            return;
          }
          symbolName = "aclshmem_wait_" + typeName;
        })
        .Case([&](dtile::WaitDepsOp) {
          symbolName = "dtile_wait_deps";
        })
        .Case([&](dtile::NotifyTileDoneOp) {
          symbolName = "dtile_notify_tile_done";
        })
        .Case([&](dtile::QueryTsConditionOp) {
          symbolName = "dtile_query_ts_condition";
        })
        .Case([&](dtile::SetCondOp) {
          symbolName = "dtile_set_cond";
        })
        .Case([&](dtile::GlobalBlockIdxMixOp) {
          symbolName = "dtile_global_block_idx_mix";
        })
        .Case([&](dtile::StIOOp stioOp) {
          auto valTy = stioOp.getVal().getType();
          if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(valTy)) {
            unsigned width = intTy.getWidth();
            if (width <= 16)
              symbolName = "dtile_st_io_u16";
            else if (width <= 32)
              symbolName = "dtile_st_io_u32";
            else
              symbolName = "dtile_st_io_u64";
          } else {
            symbolName = "dtile_st_io_u64";
          }
        })
        .Case([&](dtile::LdIOOp) {
          symbolName = "dtile_ld_io";
        })
        .Case([&](dtile::SetStatusOp) {
          symbolName = "dtile_set_status";
        });

    if (unsupportedTypeForSymbol) {
      return rewriter.notifyMatchFailure(
          op, "unsupported type when deriving distributed symbol");
    }

    if (symbolName == "aclshmem_ptr") {
      auto elemTy = llvm::cast<triton::PointerType>(op->getOperand(0).getType())
                        .getPointeeType();
      std::string typeName;
      if (failed(getTypeName(elemTy, typeName))) {
        return rewriter.notifyMatchFailure(
            op, "failed to derive aclshmem_ptr type suffix");
      }
      symbolName = "aclshmem_ptr_" + typeName;
    }

    if (symbolName.empty()) {
      symbolName = op->getName().getStringRef().drop_front(
          ASCEND::distributedDialectPrefixLen);
    }
    std::string customName = customPrefix + "." + symbolName;
    ValueRange operands = op->getOperands();
    SmallVector<Value> operandStorage;
    if (symbolName == "aclshmem_my_pe" || symbolName == "aclshmem_n_pes") {
      // TODO: Support device mesh
      operands = ValueRange();
    } else if constexpr (std::is_same_v<DistOp, distributed::NotifyOp>) {
      auto notifyOp = llvm::cast<distributed::NotifyOp>(op);
      auto i32Ty = rewriter.getI32Type();
      Value signal =
          rewriter.create<arith::TruncIOp>(loc, i32Ty, notifyOp.getSignalVal());
      // ACLSHMEM_SIGNAL_SET = 0, ACLSHMEM_SIGNAL_ADD = 1
      int32_t aclSigOp =
          notifyOp.getSigOp() == distributed::SignalOp::ADD ? 1 : 0;
      Value sigOpVal = rewriter.create<arith::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(aclSigOp));
      operandStorage = {notifyOp.getSigAddr(), signal, sigOpVal,
                        notifyOp.getRank()};
      operands = ValueRange(operandStorage);
    }
    llvm::SmallVector<Value> customResults;
    for (auto res : op->getResults()) {
      if (auto tensorTy = llvm::dyn_cast<RankedTensorType>(res.getType())) {
        auto emptyOp = rewriter.create<tensor::EmptyOp>(
            op->getLoc(), tensorTy.getShape(), tensorTy.getElementType());
        customResults.emplace_back(emptyOp);
      }
    }
    auto customOp =
        rewriter.create<hivm::CustomOp>(loc, op->getResultTypes(), customName,
                                        operands, customResults, ValueRange{});
    customOp->setAttrs(op->getAttrs());
    customOp->setAttr("hivm.is_distributed", rewriter.getUnitAttr());
    auto pipePair = ASCEND::pipeMap.find(customName);
    if (pipePair != ASCEND::pipeMap.end()) {
      customOp.setPipe(pipePair->getSecond());
    } else {
      customOp.setPipe(hivm::PIPE::PIPE_S);
    }
    customOp.setVFMode(hivm::VFMode::SIMD);
    customOp->setAttr("symbol", rewriter.getStringAttr(symbolName));
    if (llvm::isa<distributed::ConsumeTokenOp>(op) &&
        llvm::isa<RankedTensorType>(customOp->getResult(0).getType())) {
      auto annotationOp = rewriter.create<annotation::MarkOp>(
          op->getLoc(), customOp->getResult(0));
      annotationOp->setAttr(ConverterUtils::continuousAttrName,
                            rewriter.getUnitAttr());
      customOp->setAttr(ConverterUtils::customSrcPtrIndexAttrName,
                        rewriter.getDenseI32ArrayAttr({0}));
    }
    if (!hasSideEffect) {
      customOp->setAttr("no_side_effect", rewriter.getUnitAttr());
    }
    llvm::SmallVector<int> gmAddrArgsIndices;
    auto funcOp = customOp->template getParentOfType<triton::FuncOp>();
    assert(funcOp && "custom op should be in tt.func op");
    for (auto &&[idx, operand] : llvm::enumerate(customOp->getOperands())) {
      if (!llvm::isa<triton::PointerType>(operand.getType())) {
        continue;
      }
      if (auto arg = llvm::dyn_cast<BlockArgument>(operand)) {
        if (arg.getOwner() == &funcOp.getFunctionBody().front()) {
          gmAddrArgsIndices.emplace_back(idx);
        }
      }
    }
    customOp->setAttr("gm_addr_args_indices",
                      rewriter.getDenseI32ArrayAttr(gmAddrArgsIndices));
    // Replace the original op with the custom op
    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, customOp);
    }

    return success();
  }

private:
  std::string customPrefix;
  bool existDotFlag;
};

/// Helper function to register distributed op to hivm.custom patterns
template <typename... Args>
void registerDistributedOpToHIVM(RewritePatternSet &patterns, bool existDot,
                                 StringRef calleePrefix = "",
                                 PatternBenefit benefit = PatternBenefit(1)) {
  patterns.add<DistributedOpToHIVM<Args>...>(patterns.getContext(), existDot,
                                             calleePrefix, benefit);
}

} // namespace

/// Populate the pattern set with distributed op to hivm.custom patterns
void mlir::triton::ASCEND::populateDistributedOpToHIVMPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit, bool existDot) {
  // Use the template function to register all distributed ops
  // All ops convert to hivm.custom with the name "dist.<op_name>"
  registerDistributedOpToHIVM<distributed::GetRankOp,
                              distributed::GetNumRanksOp, distributed::SymmAtOp,
                              distributed::WaitOp, distributed::ConsumeTokenOp,
                              distributed::NotifyOp, distributed::ExternCallOp,
                              dtile::WaitDepsOp,
                              dtile::NotifyTileDoneOp,
                              dtile::QueryTsConditionOp,
                              dtile::SetCondOp,
                              dtile::GlobalBlockIdxMixOp,
                              dtile::DcciOp,
                              dtile::StIOOp,
                              dtile::SetStatusOp>(
      patterns, existDot, "dist", benefit);
}
