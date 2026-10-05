//===- SplitLoopTransformOps.cpp - Split-loop schedule as transform ops ---===//
//
// The rewrites of stages S1-S3 (SplitLoops/ir/tail/make_s*.py) and of the fused variant
// (SplitLoops/both) as transform ops. See the .td for what each op matches, checks and produces.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/TransformOps/SplitLoopTransformOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/TuneExtension/TuneExtension.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "llvm/ADT/SetVector.h"

#include <functional>

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/TransformOps/SplitLoopTransformOps.cpp.inc"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// The single integer a transform param holds (e.g. the value a tune.knob selected).
FailureOr<int64_t> getIntParam(transform::TransformState &state, Value param) {
  ArrayRef<Attribute> attrs = state.getParams(param);
  if (attrs.size() != 1)
    return failure();
  auto ia = dyn_cast<IntegerAttr>(attrs.front());
  if (!ia)
    return failure();
  return ia.getInt();
}

std::optional<int64_t> getConstInt(Value v) {
  if (!v)
    return std::nullopt;
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  if (auto c = v.getDefiningOp<LLVM::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  return std::nullopt;
}

bool isCallTo(Operation *op, StringRef name) {
  auto call = dyn_cast_or_null<LLVM::CallOp>(op);
  if (!call)
    return false;
  std::optional<StringRef> callee = call.getCallee();
  return callee && *callee == name;
}

/// Are `a` and `b` the same code, up to exchanging `p` and `q` (values defined outside both)?
/// Alias-scope declarations may name different scopes: each launch declares its own.
struct SwapEquivalence {
  Value p, q;
  DenseMap<Value, Value> map;

  bool operandsMatch(Value x, Value y) {
    auto it = map.find(x);
    if (it != map.end())
      return it->second == y;
    if (x == p)
      return y == q;
    if (x == q)
      return y == p;
    return x == y;
  }

  bool ops(Operation *a, Operation *b) {
    if (a->getName() != b->getName() ||
        a->getNumOperands() != b->getNumOperands() ||
        a->getNumResults() != b->getNumResults() ||
        a->getNumRegions() != b->getNumRegions())
      return false;
    if (a->getName().getStringRef() !=
        "llvm.intr.experimental.noalias.scope.decl") {
      if (a->getDiscardableAttrDictionary() !=
          b->getDiscardableAttrDictionary())
        return false;
      if (a->getPropertiesAsAttribute() != b->getPropertiesAsAttribute())
        return false;
    }
    for (auto [x, y] : llvm::zip(a->getOperands(), b->getOperands()))
      if (!operandsMatch(x, y))
        return false;
    for (auto [x, y] : llvm::zip(a->getResults(), b->getResults())) {
      if (x.getType() != y.getType())
        return false;
      map[x] = y;
    }
    for (auto [ra, rb] : llvm::zip(a->getRegions(), b->getRegions()))
      if (!regions(ra, rb))
        return false;
    return true;
  }

  bool regions(Region &ra, Region &rb) {
    if (ra.getBlocks().size() != rb.getBlocks().size())
      return false;
    for (auto [ba, bb] : llvm::zip(ra, rb)) {
      if (ba.getNumArguments() != bb.getNumArguments() ||
          ba.getOperations().size() != bb.getOperations().size())
        return false;
      for (auto [x, y] : llvm::zip(ba.getArguments(), bb.getArguments())) {
        if (x.getType() != y.getType())
          return false;
        map[x] = y;
      }
      for (auto [oa, ob] : llvm::zip(ba, bb))
        if (!ops(&oa, &ob))
          return false;
    }
    return true;
  }
};

/// The buffers a launch touches: pointers defined outside it, viewed through pointer2memref
/// inside it. `ok` is false if an access goes through anything else.
struct LaunchBuffers {
  llvm::SetVector<Value> all, stored, loaded;
  bool ok = true;
};

LaunchBuffers collectBuffers(Operation *launch) {
  LaunchBuffers lb;
  Region &body = launch->getRegion(0);
  auto base = [&](Value memref) -> Value {
    auto view = memref.getDefiningOp<enzymexla::Pointer2MemrefOp>();
    if (!view || !body.isAncestor(view->getParentRegion()))
      return Value();
    Value ptr = view.getSource();
    if (body.isAncestor(ptr.getParentRegion()))
      return Value();
    return ptr;
  };
  launch->walk([&](Operation *op) {
    Value memref;
    bool isStore = false;
    if (auto ld = dyn_cast<affine::AffineLoadOp>(op)) {
      memref = ld.getMemRef();
    } else if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
      memref = st.getMemRef();
      isStore = true;
    } else {
      return;
    }
    Value ptr = base(memref);
    if (!ptr) {
      lb.ok = false;
      return;
    }
    lb.all.insert(ptr);
    (isStore ? lb.stored : lb.loaded).insert(ptr);
  });
  return lb;
}

/// expr = sum(coeffs[k] * input_k) + cst, inputs being the map's dims then symbols.
LogicalResult linearize(AffineExpr e, unsigned numDims,
                        SmallVectorImpl<int64_t> &coeffs, int64_t &cst,
                        int64_t mult = 1) {
  switch (e.getKind()) {
  case AffineExprKind::Add: {
    auto bin = cast<AffineBinaryOpExpr>(e);
    if (failed(linearize(bin.getLHS(), numDims, coeffs, cst, mult)))
      return failure();
    return linearize(bin.getRHS(), numDims, coeffs, cst, mult);
  }
  case AffineExprKind::Mul: {
    auto bin = cast<AffineBinaryOpExpr>(e);
    if (auto c = dyn_cast<AffineConstantExpr>(bin.getRHS()))
      return linearize(bin.getLHS(), numDims, coeffs, cst, mult * c.getValue());
    if (auto c = dyn_cast<AffineConstantExpr>(bin.getLHS()))
      return linearize(bin.getRHS(), numDims, coeffs, cst, mult * c.getValue());
    return failure();
  }
  case AffineExprKind::Constant:
    cst += mult * cast<AffineConstantExpr>(e).getValue();
    return success();
  case AffineExprKind::DimId:
    coeffs[cast<AffineDimExpr>(e).getPosition()] += mult;
    return success();
  case AffineExprKind::SymbolId:
    coeffs[numDims + cast<AffineSymbolExpr>(e).getPosition()] += mult;
    return success();
  default:
    return failure();
  }
}

LLVM::LLVMFuncOp declareFunc(OpBuilder &b, ModuleOp module, StringRef name,
                             Type result, ArrayRef<Type> params) {
  if (auto f = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return f;
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToEnd(module.getBody());
  return LLVM::LLVMFuncOp::create(b, module.getLoc(), name,
                                  LLVM::LLVMFunctionType::get(result, params));
}

Value call(OpBuilder &b, Location loc, LLVM::LLVMFuncOp f, ValueRange args) {
  auto c = LLVM::CallOp::create(b, loc, f, args);
  return c.getNumResults() ? c.getResult() : Value();
}

//===----------------------------------------------------------------------===//
// normalize_pingpong: matching
//===----------------------------------------------------------------------===//

struct PingPong {
  scf::WhileOp loop;
  enzymexla::GPUWrapperOp launchA;
  Value src, dst;        // launch A reads src and writes dst
  Value tripCount;       // i32, at least 1
  Block *errorReport = nullptr; // reports a launch error and exits
  Value errorValue;      // the loop result that block reports
  SmallVector<unsigned> zeroOperands; // parent terminator operands that are 0 on normal exit
};

/// Is `v` extui(cmpi eq(next, tc)) (either operand order)?
bool isDoneFlag(Value v, Value next, Value tc) {
  auto ext = v.getDefiningOp<arith::ExtUIOp>();
  if (!ext)
    return false;
  auto cmp = ext.getIn().getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::eq)
    return false;
  return (cmp.getLhs() == next && cmp.getRhs() == tc) ||
         (cmp.getLhs() == tc && cmp.getRhs() == next);
}

FailureOr<PingPong> matchPingPong(scf::WhileOp loop, std::string &why) {
  PingPong m;
  m.loop = loop;
  auto fail = [&](const Twine &msg) {
    why = msg.str();
    return failure();
  };

  // Two launches, the first directly in the loop body.
  SmallVector<enzymexla::GPUWrapperOp> launches;
  loop.getBefore().walk<WalkOrder::PreOrder>([&](enzymexla::GPUWrapperOp w) {
    launches.push_back(w);
    return WalkResult::skip();
  });
  bool launchInDo = false;
  loop.getAfter().walk([&](enzymexla::GPUWrapperOp) { launchInDo = true; });
  if (launches.size() != 2 || launchInDo)
    return fail("expected exactly two launches, both in the loop's body");
  Block &body = loop.getBefore().front();
  enzymexla::GPUWrapperOp a = launches[0], b = launches[1];
  if (a->getBlock() != &body)
    return fail("the first launch is not at the top level of the loop body");

  // Same kernel, buffers swapped.
  LaunchBuffers ba = collectBuffers(a), bb = collectBuffers(b);
  if (!ba.ok || !bb.ok)
    return fail("a launch accesses memory other than through views of pointers "
                "defined outside it");
  if (ba.all.size() != 2 || ba.stored.size() != 1)
    return fail("the first launch does not read one buffer and write another");
  m.dst = ba.stored.front();
  m.src = ba.all[0] == m.dst ? ba.all[1] : ba.all[0];
  if (bb.all.size() != 2 || bb.stored.size() != 1 || bb.stored.front() != m.src ||
      !bb.all.contains(m.dst))
    return fail("the second launch does not write the first launch's source");
  SwapEquivalence eq;
  eq.p = m.src;
  eq.q = m.dst;
  if (!eq.ops(a, b))
    return fail("the two launches are not the same kernel with the buffers "
                "swapped");

  // Besides the launches, only error checks and loop control.
  std::string other;
  loop.getBefore().walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<enzymexla::GPUWrapperOp>(op))
      return WalkResult::skip();
    Dialect *dialect = op->getDialect();
    if (isa<scf::IfOp, scf::YieldOp, scf::ConditionOp>(op) ||
        (dialect && (isa<arith::ArithDialect>(dialect) ||
                     dialect->getNamespace() == "ub")) ||
        isCallTo(op, "cudaGetLastError"))
      return WalkResult::advance();
    other = op->getName().getStringRef().str();
    return WalkResult::interrupt();
  });
  if (!other.empty())
    return fail("the loop body does more than launch and check errors: " +
                other);

  // Counter from 0, forwarded through the do-region, compared with a trip count >= 1.
  if (loop.getInits().size() < 1 || getConstInt(loop.getInits()[0]) != 0)
    return fail("the loop counter does not start at 0");
  Value iv = body.getArgument(0);
  arith::AddIOp next;
  for (Operation *user : iv.getUsers())
    if (auto add = dyn_cast<arith::AddIOp>(user))
      if ((add.getLhs() == iv && getConstInt(add.getRhs()) == 1) ||
          (add.getRhs() == iv && getConstInt(add.getLhs()) == 1))
        next = add;
  scf::ConditionOp cond = loop.getConditionOp();
  if (!next || cond.getArgs().empty() || cond.getArgs()[0] != next.getResult())
    return fail("the loop does not step its counter by one");
  Block &doBlock = loop.getAfter().front();
  if (doBlock.getOperations().size() != 1 ||
      loop.getYieldOp().getNumOperands() != 1 ||
      loop.getYieldOp().getOperand(0) != doBlock.getArgument(0))
    return fail("the do-region does more than pass the counter back");

  // Exit protocol: continue = no error in either launch && next != tripcount.
  auto ifOp = cond.getCondition().getDefiningOp<scf::IfOp>();
  if (!ifOp || ifOp->getBlock() != &body || !ifOp->isAncestor(b))
    return fail("the loop condition is not the error check around the second "
                "launch");
  auto condIdx = cast<OpResult>(cond.getCondition()).getResultNumber();
  Value thenCont = ifOp.thenYield().getOperand(condIdx);
  Value elseCont = ifOp.elseYield().getOperand(condIdx);
  auto andOp = thenCont.getDefiningOp<arith::AndIOp>();
  if (!andOp || getConstInt(elseCont) != 0)
    return fail("the loop continues after a launch error");
  Value tc;
  for (Value operand : andOp->getOperands())
    if (auto cmp = operand.getDefiningOp<arith::CmpIOp>())
      if (cmp.getPredicate() == arith::CmpIPredicate::ne) {
        if (cmp.getLhs() == next.getResult())
          tc = cmp.getRhs();
        else if (cmp.getRhs() == next.getResult())
          tc = cmp.getLhs();
      }
  if (!tc || loop->isAncestor(tc.getParentRegion()->getParentOp()))
    return fail("no loop-invariant trip count");
  auto maxOp = tc.getDefiningOp<arith::MaxUIOp>();
  if (!maxOp || (getConstInt(maxOp.getLhs()) != 1 &&
                 getConstInt(maxOp.getRhs()) != 1))
    return fail("cannot show the trip count is at least 1 (a do-while runs "
                "once)");
  m.tripCount = tc;
  auto errCheck = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!errCheck || !isCallTo(errCheck.getLhs().getDefiningOp(),
                             "cudaGetLastError"))
    return fail("the first launch is not followed by an error check");

  // After the loop: report an error and exit, or fall through with status 0.
  Block *parent = loop->getBlock();
  Operation *term = parent->getTerminator();
  llvm::SmallPtrSet<Operation *, 16> post;
  for (Operation *op = loop->getNextNode(); op && op != term;
       op = op->getNextNode())
    post.insert(op);
  for (Operation *user : loop->getUsers()) {
    Operation *top = parent->findAncestorOpInBlock(*user);
    if (!top || !post.contains(top))
      return fail("a result of the loop is used outside the code after it");
  }
  for (Operation *op : post) {
    for (Operation *user : op->getUsers()) {
      Operation *top = parent->findAncestorOpInBlock(*user);
      if (!top || (top != term && !post.contains(top)))
        return fail("a value computed after the loop is used elsewhere");
    }
    WalkResult w = op->walk([&](Operation *inner) {
      if (isa<LLVM::CallOp>(inner) &&
          !(isCallTo(inner, "cudaGetErrorString") ||
            isCallTo(inner, "fprintf") || isCallTo(inner, "printf") ||
            isCallTo(inner, "fputs") || isCallTo(inner, "exit")))
        return WalkResult::interrupt();
      if (isa<affine::AffineStoreOp, LLVM::StoreOp>(inner))
        return WalkResult::interrupt();
      if (isCallTo(inner, "cudaGetErrorString") &&
          isa<OpResult>(inner->getOperand(0)) &&
          inner->getOperand(0).getDefiningOp() == loop.getOperation()) {
        Block *blk = inner->getBlock();
        bool exits = llvm::any_of(
            *blk, [](Operation &o) { return isCallTo(&o, "exit"); });
        if (exits && !m.errorReport) {
          m.errorReport = blk;
          m.errorValue = inner->getOperand(0);
        }
      }
      return WalkResult::advance();
    });
    if (w.wasInterrupted())
      return fail("the code after the loop does more than report an error");
  }
  if (!m.errorReport)
    return fail("no error report after the loop");
  // Terminator operands computed from the loop must be extui(status != 1), where the status
  // the loop exits with on its last error-free iteration is extui(next == tripcount) = 1.
  for (auto [idx, operand] : llvm::enumerate(term->getOperands())) {
    Operation *def = operand.getDefiningOp();
    if (!def || !post.contains(parent->findAncestorOpInBlock(*def)))
      continue;
    auto ext = operand.getDefiningOp<arith::ExtUIOp>();
    auto cmp = ext ? ext.getIn().getDefiningOp<arith::CmpIOp>() : nullptr;
    auto status = cmp ? cmp.getLhs().getDefiningOp<arith::IndexCastUIOp>()
                      : nullptr;
    if (!status || cmp.getPredicate() != arith::CmpIPredicate::ne ||
        getConstInt(cmp.getRhs()) != 1 ||
        status.getIn().getDefiningOp() != loop.getOperation())
      return fail("unrecognized use of the loop's exit status");
    unsigned k = cast<OpResult>(status.getIn()).getResultNumber();
    auto statusDef = dyn_cast<OpResult>(cond.getArgs()[k]);
    if (!statusDef || statusDef.getOwner() != ifOp.getOperation())
      return fail("unrecognized loop exit status");
    auto sel = ifOp.thenYield()
                   .getOperand(statusDef.getResultNumber())
                   .getDefiningOp<arith::SelectOp>();
    if (!sel || !isDoneFlag(sel.getTrueValue(), next.getResult(), tc))
      return fail("the loop's exit status is not 1 on normal completion");
    m.zeroOperands.push_back(idx);
  }
  m.launchA = a;
  return m;
}

//===----------------------------------------------------------------------===//
// fission: legality
//===----------------------------------------------------------------------===//

struct FissionPlan {
  enzymexla::GPUWrapperOp launch;
  affine::AffineParallelOp par;
  int64_t gridX, rows, width;
  int64_t rowStride;       // bytes between consecutive rows
  int64_t reach;           // write reach + read reach + a row's width, in bytes
  int64_t conflictRows;    // rows further apart than this never touch the same byte
  Type gepElem;            // element type of the pointer shift
  int64_t gepElemBytes;
  llvm::SetVector<Value> buffers;
  // A launch made by fuse_pairs: cut by tile layers of layerRows rows.
  bool fused = false;
  int64_t layerRows = 0, layers = 0;
};

FailureOr<FissionPlan> planFission(enzymexla::GPUWrapperOp launch,
                                   std::string &why) {
  FissionPlan p;
  p.launch = launch;
  auto fail = [&](const Twine &msg) {
    why = msg.str();
    return failure();
  };
  auto dims = launch.getBlockDims();
  if (dims.size() != 6)
    return fail("expected grid and block dimensions on the launch");
  std::optional<int64_t> gx = getConstInt(dims[0]), gy = getConstInt(dims[1]),
                         gz = getConstInt(dims[2]), bx = getConstInt(dims[3]),
                         by = getConstInt(dims[4]), bz = getConstInt(dims[5]);
  if (!gx || !gy || !gz || !bx || !by || !bz || *gz != 1 || *by != 1 ||
      *bz != 1)
    return fail("expected a constant 2-D grid of 1-D blocks");
  Block &body = launch.getRegion().front();
  p.par = dyn_cast<affine::AffineParallelOp>(&body.front());
  if (!p.par || p.par->getNextNode() != body.getTerminator() ||
      p.par.getNumDims() != 2)
    return fail("the launch body is not one 2-D affine.parallel");
  std::optional<SmallVector<int64_t, 8>> ranges = p.par.getConstantRanges();
  for (AffineExpr lb : p.par.getLowerBoundsMap().getResults())
    if (auto c = dyn_cast<AffineConstantExpr>(lb); !c || c.getValue() != 0)
      return fail("the parallel loop does not start at 0");
  if (!ranges || llvm::any_of(p.par.getSteps(), [](int64_t s) { return s != 1; }))
    return fail("the parallel loop's bounds are not constant");
  p.gridX = *gx;
  p.rows = (*ranges)[0];
  p.width = (*ranges)[1];
  if (p.rows != *gx * *gy || p.width != *bx)
    return fail("the parallel loop does not cover (grid rows, block threads)");

  LaunchBuffers bufs = collectBuffers(launch);
  if (!bufs.ok)
    return fail("an access does not go through a view of a pointer defined "
                "outside the launch");
  for (Value s : bufs.stored)
    if (bufs.loaded.contains(s))
      return fail("the launch reads and writes the same buffer");
  p.buffers = bufs.all;

  Value rowIV = p.par.getIVs()[0], colIV = p.par.getIVs()[1];
  struct Access {
    int64_t offset;
    bool store;
  };
  SmallVector<Access> accesses;
  p.rowStride = -1;
  int64_t colStride = -1;
  std::string bad;
  p.par.walk([&](Operation *op) {
    if (op->getName().getStringRef().contains("atomic")) {
      bad = "the launch uses atomics";
      return WalkResult::interrupt();
    }
    AffineMap map;
    ValueRange operands;
    Value memref;
    bool store = false;
    if (auto ld = dyn_cast<affine::AffineLoadOp>(op)) {
      map = ld.getAffineMap();
      operands = ld.getMapOperands();
      memref = ld.getMemRef();
    } else if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
      map = st.getAffineMap();
      operands = st.getMapOperands();
      memref = st.getMemRef();
      store = true;
    } else if (isa<LLVM::LoadOp, LLVM::StoreOp>(op) ||
               op->getName().getStringRef() == "memref.load" ||
               op->getName().getStringRef() == "memref.store") {
      bad = "the launch has a non-affine access";
      return WalkResult::interrupt();
    } else {
      return WalkResult::advance();
    }
    auto type = cast<MemRefType>(memref.getType());
    if (!type.getElementType().isIntOrFloat() || map.getNumResults() != 1) {
      bad = "unsupported view";
      return WalkResult::interrupt();
    }
    int64_t elemBytes = type.getElementTypeBitWidth() / 8;
    SmallVector<int64_t> coeffs(map.getNumInputs(), 0);
    int64_t cst = 0;
    if (failed(linearize(map.getResult(0), map.getNumDims(), coeffs, cst))) {
      bad = "an index is not linear";
      return WalkResult::interrupt();
    }
    int64_t cr = 0, cc = 0;
    for (auto [k, operand] : llvm::enumerate(operands)) {
      if (!coeffs[k])
        continue;
      if (operand == rowIV)
        cr += coeffs[k];
      else if (operand == colIV)
        cc += coeffs[k];
      else {
        bad = "an index depends on more than the launch indices";
        return WalkResult::interrupt();
      }
    }
    if (p.rowStride < 0)
      p.rowStride = cr * elemBytes;
    if (colStride < 0)
      colStride = cc * elemBytes;
    if (cr * elemBytes != p.rowStride || cc * elemBytes != colStride) {
      bad = "rows or threads are not uniformly strided";
      return WalkResult::interrupt();
    }
    if (!p.gepElem && isa<FloatType>(type.getElementType())) {
      p.gepElem = type.getElementType();
      p.gepElemBytes = elemBytes;
    }
    accesses.push_back({cst * elemBytes, store});
    return WalkResult::advance();
  });
  if (!bad.empty())
    return fail(bad);
  if (accesses.empty() || p.rowStride <= 0 || colStride <= 0)
    return fail("no strided accesses");
  int64_t rowBytes = (p.width - 1) * colStride;
  if (rowBytes >= p.rowStride)
    return fail("a row's threads overlap the next row");
  if (!p.gepElem) {
    p.gepElem = IntegerType::get(launch.getContext(), 8);
    p.gepElemBytes = 1;
  }
  if (p.rowStride % p.gepElemBytes)
    return fail("the row stride is not a whole number of elements");

  // Group accesses into fields: offsets closer than half the grid belong together.
  llvm::sort(accesses, [](const Access &x, const Access &y) {
    return x.offset < y.offset;
  });
  int64_t gap = p.rows * p.rowStride / 2;
  int64_t readReach = 0, writeReach = 0;
  for (size_t begin = 0; begin < accesses.size();) {
    size_t end = begin + 1;
    while (end < accesses.size() &&
           accesses[end].offset - accesses[end - 1].offset <= gap)
      ++end;
    std::optional<int64_t> base;
    int stores = 0;
    for (size_t k = begin; k < end; ++k)
      if (!accesses[k].store && !base)
        base = accesses[k].offset;
    for (size_t k = begin; k < end; ++k)
      if (accesses[k].store) {
        ++stores;
        if (!base)
          base = accesses[k].offset;
      }
    if (stores > 1)
      return fail("a field is written twice, so the writes cannot be shown "
                  "injective");
    for (size_t k = begin; k < end; ++k) {
      int64_t d = std::abs(accesses[k].offset - *base);
      (accesses[k].store ? writeReach : readReach) =
          std::max(accesses[k].store ? writeReach : readReach, d);
    }
    begin = end;
  }
  p.reach = writeReach + readReach + rowBytes;
  p.conflictRows = p.reach / p.rowStride;
  return p;
}

/// A launch made by fuse_pairs: its tile layers and reach are the ones fuse_pairs recorded, since
/// the fused kernel's accesses are computed, not affine in the launch indices.
FailureOr<FissionPlan> planFusedFission(enzymexla::GPUWrapperOp launch,
                                        std::string &why) {
  FissionPlan p;
  p.launch = launch;
  p.fused = true;
  auto get = [&](StringRef name) -> int64_t {
    auto a = launch->getAttrOfType<IntegerAttr>(name);
    return a ? a.getInt() : -1;
  };
  p.layerRows = get("split_loop.layer_rows");
  p.layers = get("split_loop.layers");
  p.rowStride = get("split_loop.row_stride_bytes");
  p.reach = get("split_loop.reach_bytes");
  if (p.layerRows <= 0 || p.layers <= 0 || p.rowStride <= 0 || p.reach < 0) {
    why = "the fused launch lacks the tile layers and reach that "
          "split_loop.fuse_pairs records";
    return failure();
  }
  Block &body = launch.getRegion().front();
  p.par = dyn_cast<affine::AffineParallelOp>(&body.front());
  std::optional<SmallVector<int64_t, 8>> ranges =
      p.par ? p.par.getConstantRanges() : std::nullopt;
  if (!p.par || p.par.getNumDims() != 3 || !ranges ||
      (*ranges)[2] != p.layers ||
      !llvm::any_of(*p.par.getBody(), [](Operation &op) {
        return isa<affine::AffineParallelOp>(op);
      })) {
    why = "the fused launch is not the grid of tiles split_loop.fuse_pairs "
          "builds";
    return failure();
  }
  p.gridX = (*ranges)[0];
  p.rows = p.layers * p.layerRows;
  p.conflictRows = p.reach / p.rowStride;
  return p;
}

//===----------------------------------------------------------------------===//
// fuse_pairs: legality
//===----------------------------------------------------------------------===//

/// An affine access of a launch body: index = cr * row + cc * col + cst, in elements of its view.
struct Access {
  Operation *op = nullptr;
  bool store = false;
  int64_t cr = 0, cc = 0, cst = 0, elemBytes = 0;
  int64_t offsetBytes() const { return cst * elemBytes; }
};

Value memrefOf(Operation *op) {
  if (auto ld = dyn_cast<affine::AffineLoadOp>(op))
    return ld.getMemRef();
  return cast<affine::AffineStoreOp>(op).getMemRef();
}

LogicalResult analyzeAccess(Operation *op, Value rowIV, Value colIV, Access &a,
                            std::string &why) {
  AffineMap map;
  ValueRange operands;
  if (auto ld = dyn_cast<affine::AffineLoadOp>(op)) {
    map = ld.getAffineMap();
    operands = ld.getMapOperands();
  } else {
    auto st = cast<affine::AffineStoreOp>(op);
    map = st.getAffineMap();
    operands = st.getMapOperands();
    a.store = true;
  }
  a.op = op;
  auto type = cast<MemRefType>(memrefOf(op).getType());
  if (type.getRank() != 1 || map.getNumResults() != 1 ||
      !type.getElementType().isIntOrFloat() ||
      type.getElementTypeBitWidth() % 8) {
    why = "an access goes through an unsupported view";
    return failure();
  }
  a.elemBytes = type.getElementTypeBitWidth() / 8;
  SmallVector<int64_t> coeffs(map.getNumInputs(), 0);
  if (failed(linearize(map.getResult(0), map.getNumDims(), coeffs, a.cst))) {
    why = "an index is not linear";
    return failure();
  }
  for (auto [k, operand] : llvm::enumerate(operands)) {
    if (!coeffs[k])
      continue;
    if (operand == rowIV) {
      a.cr += coeffs[k];
    } else if (operand == colIV) {
      a.cc += coeffs[k];
    } else {
      why = "an index depends on more than the launch indices";
      return failure();
    }
  }
  return success();
}

/// The shared memory a block can have on the launch's target: static shared memory is limited to
/// 48 KiB, and beyond that the GPU lowering places block-scope arrays in dynamic shared memory,
/// up to the device's opt-in maximum.
int64_t maxSharedBytesPerBlock(Operation *launch, std::string &arch) {
  auto target = launch->getAttrOfType<StringAttr>("target_cpu");
  arch = target ? target.getValue().str() : "unknown target";
  StringRef rest = arch;
  unsigned sm = 0;
  if (rest.consume_front("sm_") && !rest.consumeInteger(10, sm)) {
    if (sm == 70 || sm == 72)
      return 96 * 1024;
    if (sm == 75)
      return 64 * 1024;
    if (sm == 80 || sm == 87)
      return 163 * 1024;
    if (sm == 86 || sm == 89 || sm == 120 || sm == 121)
      return 99 * 1024;
    if (sm == 90 || sm == 100 || sm == 103)
      return 227 * 1024;
  }
  return 48 * 1024;
}

/// v mod n in (-n/2, n/2].
int64_t centeredMod(int64_t v, int64_t n) {
  int64_t m = ((v % n) + n) % n;
  return 2 * m > n ? m - n : m;
}

struct FusePlan {
  scf::ForOp stepLoop;
  enzymexla::GPUWrapperOp launch;
  affine::AffineParallelOp par;
  Value src, dst;   // what the launch reads and writes, chosen by the step's parity
  Value bufA, bufB; // src and dst of the even steps
  // Cells: column = thread (width of them), row = grid.x * plane + row in plane.
  int64_t planeRows = 0, planes = 0, width = 0, rows = 0;
  int64_t rowStride = 0, colStride = 0; // bytes
  int64_t pitch = 0;                    // columns from one row to the next
  struct Field {
    Access load;                    // a load of the field at the cell itself
    int64_t dx = 0, dy = 0, dz = 0; // where the cell stores the field, relative to itself
  };
  SmallVector<Field> fields;          // the written fields, in address order
  DenseMap<Operation *, Access> accesses;
  // For a load or store of a written field, the field; -1 for a load of a field nothing writes.
  DenseMap<Operation *, int> fieldOf;
  Type elemType;
  int64_t elemBytes = 0;
  int64_t storeReach = 0; // bytes from a cell to the farthest slot it writes
};

FailureOr<FusePlan> planFusePairs(scf::ForOp stepLoop, std::string &why) {
  FusePlan p;
  p.stepLoop = stepLoop;
  auto fail = [&](const Twine &msg) {
    why = msg.str();
    return failure();
  };
  if (!stepLoop->hasAttr("split_loop.steps"))
    return fail("expected a step loop from split_loop.normalize_pingpong");
  if (getConstInt(stepLoop.getLowerBound()) != 0 ||
      getConstInt(stepLoop.getStep()) != 1 || stepLoop.getNumRegionIterArgs())
    return fail("expected a unit-stride step loop from 0 without loop-carried "
                "values");
  Block *body = stepLoop.getBody();
  p.launch = dyn_cast_or_null<enzymexla::GPUWrapperOp>(
      body->getTerminator()->getPrevNode());
  if (!p.launch)
    return fail("the step loop's body does not end with the launch "
                "(fuse_pairs goes before fission)");
  for (Operation &op : body->without_terminator())
    if (&op != p.launch.getOperation() &&
        (op.getNumRegions() || !isMemoryEffectFree(&op)))
      return fail("the step loop does more than choose the step's buffers and "
                  "launch: " +
                  op.getName().getStringRef());

  // A constant 2-D grid of 1-D blocks, raised to one parallel loop over (rows, columns).
  auto dims = p.launch.getBlockDims();
  if (dims.size() != 6)
    return fail("expected grid and block dimensions on the launch");
  std::optional<int64_t> gx = getConstInt(dims[0]), gy = getConstInt(dims[1]),
                         gz = getConstInt(dims[2]), bx = getConstInt(dims[3]),
                         by = getConstInt(dims[4]), bz = getConstInt(dims[5]);
  if (!gx || !gy || !gz || !bx || !by || !bz || *gz != 1 || *by != 1 ||
      *bz != 1)
    return fail("expected a constant 2-D grid of 1-D blocks");
  Block &launchBody = p.launch.getRegion().front();
  p.par = dyn_cast<affine::AffineParallelOp>(&launchBody.front());
  if (!p.par || p.par->getNextNode() != launchBody.getTerminator() ||
      p.par.getNumDims() != 2)
    return fail("the launch body is not one 2-D affine.parallel");
  std::optional<SmallVector<int64_t, 8>> ranges = p.par.getConstantRanges();
  for (AffineExpr lb : p.par.getLowerBoundsMap().getResults())
    if (auto c = dyn_cast<AffineConstantExpr>(lb); !c || c.getValue() != 0)
      return fail("the parallel loop does not start at 0");
  if (!ranges ||
      llvm::any_of(p.par.getSteps(), [](int64_t s) { return s != 1; }))
    return fail("the parallel loop's bounds are not constant");
  p.planeRows = *gx;
  p.planes = *gy;
  p.width = *bx;
  p.rows = *gx * *gy;
  if ((*ranges)[0] != p.rows || (*ranges)[1] != p.width)
    return fail("the parallel loop does not cover (grid rows, block threads)");

  // It reads one buffer and writes the other; the step's parity picks which.
  LaunchBuffers bufs = collectBuffers(p.launch);
  if (!bufs.ok || bufs.all.size() != 2 || bufs.loaded.size() != 1 ||
      bufs.stored.size() != 1 || bufs.loaded.front() == bufs.stored.front())
    return fail("the launch does not read one buffer and write another");
  p.src = bufs.loaded.front();
  p.dst = bufs.stored.front();
  auto srcSel = p.src.getDefiningOp<LLVM::SelectOp>();
  auto dstSel = p.dst.getDefiningOp<LLVM::SelectOp>();
  auto even = srcSel ? srcSel.getCondition().getDefiningOp<arith::CmpIOp>()
                     : arith::CmpIOp();
  auto parity =
      even ? even.getLhs().getDefiningOp<arith::RemUIOp>() : arith::RemUIOp();
  if (!dstSel || !parity || dstSel.getCondition() != srcSel.getCondition() ||
      srcSel.getTrueValue() != dstSel.getFalseValue() ||
      srcSel.getFalseValue() != dstSel.getTrueValue() ||
      even.getPredicate() != arith::CmpIPredicate::eq ||
      getConstInt(even.getRhs()) != 0 ||
      parity.getLhs() != stepLoop.getInductionVar() ||
      getConstInt(parity.getRhs()) != 2)
    return fail("the step's buffers are not (a, b) on even steps and (b, a) on "
                "odd ones");
  p.bufA = srcSel.getTrueValue();
  p.bufB = srcSel.getFalseValue();
  if (!stepLoop.isDefinedOutsideOfLoop(p.bufA) ||
      !stepLoop.isDefinedOutsideOfLoop(p.bufB))
    return fail("the buffers change inside the loop");

  // The body: arithmetic, conditionals, and affine accesses through views of the two buffers.
  Value rowIV = p.par.getIVs()[0], colIV = p.par.getIVs()[1];
  SmallVector<Access> all;
  std::string bad;
  p.par.getBody()->walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    StringRef ns = op->getDialect() ? op->getDialect()->getNamespace() : "";
    if (isa<affine::AffineLoadOp, affine::AffineStoreOp>(op)) {
      if (isa<affine::AffineStoreOp>(op) && op->getBlock() != p.par.getBody()) {
        bad = "a store is conditional, so which slots a step writes could "
              "depend on data";
        return WalkResult::interrupt();
      }
      Access a;
      if (failed(analyzeAccess(op, rowIV, colIV, a, bad)))
        return WalkResult::interrupt();
      all.push_back(a);
      p.accesses[op] = a;
      return WalkResult::advance();
    }
    if (name.contains("atomic")) {
      bad = "the kernel uses atomics";
      return WalkResult::interrupt();
    }
    if (isa<scf::IfOp, scf::YieldOp, affine::AffineYieldOp,
            enzymexla::Pointer2MemrefOp>(op) ||
        name == "llvm.intr.experimental.noalias.scope.decl" || ns == "arith" ||
        ns == "math" || ns == "ub")
      return WalkResult::advance();
    bad = ("the kernel contains " + name +
           ", which the fusion does not know how to run twice")
              .str();
    return WalkResult::interrupt();
  });
  if (!bad.empty())
    return fail(bad);
  if (all.empty())
    return fail("the kernel accesses no memory");
  p.rowStride = all.front().cr * all.front().elemBytes;
  p.colStride = all.front().cc * all.front().elemBytes;
  for (const Access &a : all)
    if (a.cr * a.elemBytes != p.rowStride || a.cc * a.elemBytes != p.colStride)
      return fail("rows or threads are not uniformly strided");
  if (p.rowStride <= 0 || p.colStride <= 0 || p.rowStride % p.colStride)
    return fail("rows are not a whole number of columns apart");
  p.pitch = p.rowStride / p.colStride;
  if (p.pitch < p.width)
    return fail("a row's threads overlap the next row");
  if (p.pitch < 3 || p.planeRows < 3)
    return fail("rows or planes are too small to tell neighbours apart");

  // Fields: accesses closer than half the grid belong together. A field is read at the cell and
  // written to one neighbour (scatter form), or only read.
  SmallVector<Access> sorted(all);
  llvm::stable_sort(sorted, [](const Access &x, const Access &y) {
    return x.offsetBytes() < y.offsetBytes();
  });
  int64_t gap = p.rows * p.rowStride / 2;
  for (size_t begin = 0; begin < sorted.size();) {
    size_t end = begin + 1;
    while (end < sorted.size() &&
           sorted[end].offsetBytes() - sorted[end - 1].offsetBytes() <= gap)
      ++end;
    SmallVector<const Access *> loads, stores;
    for (size_t k = begin; k < end; ++k)
      (sorted[k].store ? stores : loads).push_back(&sorted[k]);
    begin = end;
    if (stores.size() > 1)
      return fail("a field is written twice, so the writes cannot be shown "
                  "injective");
    if (loads.empty())
      return fail("a field is written but not read at the cell, so the kernel "
                  "is not in scatter form");
    int64_t base = loads.front()->offsetBytes();
    for (const Access *l : loads)
      if (l->offsetBytes() != base)
        return fail("a load reads a neighbouring cell, so the kernel is not in "
                    "scatter form (each cell reads only itself)");
    int field = stores.empty() ? -1 : (int)p.fields.size();
    for (const Access *l : loads)
      p.fieldOf[l->op] = field;
    if (stores.empty())
      continue;
    const Access *st = stores.front();
    p.fieldOf[st->op] = field;
    Type type = cast<MemRefType>(memrefOf(st->op).getType()).getElementType();
    for (const Access *l : loads)
      if (cast<MemRefType>(memrefOf(l->op).getType()).getElementType() != type)
        return fail("a field is read and written with different types");
    if (!p.elemType) {
      p.elemType = type;
      p.elemBytes = st->elemBytes;
    } else if (p.elemType != type) {
      return fail("the written fields have different types");
    }
    int64_t off = st->offsetBytes() - base;
    if (off % p.colStride)
      return fail("a field is stored off the column grid");
    FusePlan::Field f;
    f.load = *loads.front();
    int64_t v = off / p.colStride;
    f.dx = centeredMod(v, p.pitch);
    int64_t dr = (v - f.dx) / p.pitch;
    f.dy = centeredMod(dr, p.planeRows);
    f.dz = (dr - f.dy) / p.planeRows;
    if (std::abs(f.dx) > 1 || std::abs(f.dy) > 1 || std::abs(f.dz) > 1)
      return fail("field " + Twine(p.fields.size()) + " is stored (" +
                  Twine(f.dx) + ", " + Twine(f.dy) + ", " + Twine(f.dz) +
                  ") cells away; the fusion handles neighbours at most one "
                  "cell away in each dimension");
    p.storeReach = std::max(p.storeReach, std::abs(off));
    p.fields.push_back(f);
  }
  if (p.fields.empty())
    return fail("the kernel writes nothing");
  return p;
}

//===----------------------------------------------------------------------===//
// fuse_pairs: emission
//===----------------------------------------------------------------------===//

Value linearIndex(OpBuilder &b, Location loc, const Access &a, Value row,
                  Value col) {
  Value v = arith::ConstantIndexOp::create(b, loc, a.cst);
  if (a.cr)
    v = arith::AddIOp::create(
        b, loc, v,
        arith::MulIOp::create(b, loc, row,
                              arith::ConstantIndexOp::create(b, loc, a.cr)));
  if (a.cc)
    v = arith::AddIOp::create(
        b, loc, v,
        arith::MulIOp::create(b, loc, col,
                              arith::ConstantIndexOp::create(b, loc, a.cc)));
  return v;
}

/// Clones the launch body for the cell (row, col). Affine accesses become memref accesses at
/// the index their linear form gives there; `onLoad` and `onStore` decide what each one does.
struct BodyCloner {
  RewriterBase &b;
  FusePlan &plan;
  IRMapping map;
  Value row, col;
  std::function<Value(affine::AffineLoadOp)> onLoad;
  std::function<void(affine::AffineStoreOp)> onStore;

  BodyCloner(RewriterBase &b, FusePlan &plan) : b(b), plan(plan) {}

  Value index(Operation *op) {
    return linearIndex(b, op->getLoc(), plan.accesses.find(op)->second, row,
                       col);
  }
  void cloneBlock(Block &block) {
    for (Operation &op : block.without_terminator())
      cloneOp(op);
  }
  void cloneOp(Operation &op) {
    if (op.getName().getStringRef() ==
        "llvm.intr.experimental.noalias.scope.decl")
      return;
    if (auto ld = dyn_cast<affine::AffineLoadOp>(op)) {
      map.map(ld.getResult(), onLoad(ld));
      return;
    }
    if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
      onStore(st);
      return;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool withElse = !ifOp.getElseRegion().empty();
      auto clone = scf::IfOp::create(b, ifOp.getLoc(), ifOp.getResultTypes(),
                                     map.lookupOrDefault(ifOp.getCondition()),
                                     withElse);
      auto fill = [&](Block &from, Block *to) {
        if (!to->empty())
          b.eraseOp(to->getTerminator());
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToEnd(to);
        cloneBlock(from);
        b.clone(*from.getTerminator(), map);
      };
      fill(ifOp.getThenRegion().front(), clone.thenBlock());
      if (withElse)
        fill(ifOp.getElseRegion().front(), clone.elseBlock());
      for (auto [from, to] : llvm::zip(ifOp.getResults(), clone.getResults()))
        map.map(from, to);
      return;
    }
    b.clone(op, map);
  }
};

struct FuseTile {
  int64_t x, y, z, threads;
};

/// The kernel that runs steps 2p and 2p+1 of `p`'s launch from `src` into `dst`: blocks own
/// tiles; a block runs the body on its tile and a one-cell halo, keeps the values the first step
/// stores into the tile in shared memory, and runs the body on the tile again from there.
enzymexla::GPUWrapperOp emitFusedLaunch(RewriterBase &b, FusePlan &p,
                                        Value src, Value dst, FuseTile t,
                                        bool unwrittenFromSrc) {
  Location loc = p.launch.getLoc();
  auto ceilDiv = [](int64_t n, int64_t d) { return (n + d - 1) / d; };
  int64_t ntx = ceilDiv(p.width, t.x), nty = ceilDiv(p.planeRows, t.y),
          ntz = ceilDiv(p.planes, t.z);
  int64_t hx = t.x + 2, hy = t.y + 2, hz = t.z + 2;
  int64_t core = t.x * t.y * t.z, halo = hx * hy * hz;
  int64_t nf = p.fields.size();
  Type i32 = b.getI32Type(), index = b.getIndexType();
  auto cidx = [&](int64_t v) -> Value {
    return arith::ConstantIndexOp::create(b, loc, v);
  };
  auto c32 = [&](int64_t v) -> Value {
    return arith::ConstantIntOp::create(b, loc, v, 32);
  };
  auto add = [&](Value x, Value y) -> Value {
    return arith::AddIOp::create(b, loc, x, y);
  };
  auto mul = [&](Value x, int64_t c) -> Value {
    return arith::MulIOp::create(b, loc, x, c32(c));
  };
  auto divu = [&](Value x, int64_t c) -> Value {
    return arith::DivUIOp::create(b, loc, x, c32(c));
  };
  auto remu = [&](Value x, int64_t c) -> Value {
    return arith::RemUIOp::create(b, loc, x, c32(c));
  };
  auto lt = [&](Value x, int64_t c) -> Value {
    return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, x, c32(c));
  };
  auto ge = [&](Value x, int64_t c) -> Value {
    return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, x, c32(c));
  };
  auto both = [&](ArrayRef<Value> conds) -> Value {
    Value r = conds.front();
    for (Value c : conds.drop_front())
      r = arith::AndIOp::create(b, loc, r, c);
    return r;
  };
  auto toI32 = [&](Value v) -> Value {
    return arith::IndexCastOp::create(b, loc, i32, v);
  };
  auto toIndex = [&](Value v) -> Value {
    return arith::IndexCastOp::create(b, loc, index, v);
  };

  auto launch = enzymexla::GPUWrapperOp::create(
      b, loc,
      ValueRange{cidx(ntx), cidx(nty), cidx(ntz), cidx(t.threads), cidx(1),
                 cidx(1)});
  // The original launch's attributes, except that the fused kernel is registered under no host
  // stub of its own (it is not the original kernel), synchronizes, and has a known block size.
  launch->setDiscardableAttrs(p.launch->getDiscardableAttrDictionary());
  SmallVector<Attribute> passthrough;
  if (auto old = p.launch->getAttrOfType<ArrayAttr>("passthrough"))
    for (Attribute a : old) {
      if (auto s = dyn_cast<StringAttr>(a);
          s && (s.getValue() == "nosync" || s.getValue() == "nofree"))
        continue;
      if (auto kv = dyn_cast<ArrayAttr>(a); kv && kv.size() == 2)
        if (auto key = dyn_cast<StringAttr>(kv[0]);
            key && (key.getValue() == "polygeist.host_symbol" ||
                    key.getValue() == "nvvm.maxntid"))
          continue;
      passthrough.push_back(a);
    }
  passthrough.push_back(
      b.getArrayAttr({b.getStringAttr("nvvm.maxntid"),
                      b.getStringAttr(std::to_string(t.threads))}));
  launch->setAttr("passthrough", b.getArrayAttr(passthrough));

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(launch.getRegion().front().getTerminator());
  auto grid = affine::AffineParallelOp::create(
      b, loc, TypeRange(), ArrayRef<arith::AtomicRMWKind>(),
      ArrayRef<int64_t>{ntx, nty, ntz});
  b.setInsertionPoint(grid.getBody()->getTerminator());
  auto alloca = memref::AllocaOp::create(
      b, loc, MemRefType::get({nf * core}, p.elemType));
  alloca.setAlignment(p.elemBytes);
  Value mid = alloca.getResult();
  auto block = affine::AffineParallelOp::create(
      b, loc, TypeRange(), ArrayRef<arith::AtomicRMWKind>(),
      ArrayRef<int64_t>{t.threads});
  b.setInsertionPoint(block.getBody()->getTerminator());
  Value tid = block.getIVs()[0];
  // Tile origin. Cells use 32-bit arithmetic (the caller checked the grid fits).
  Value ox = mul(toI32(grid.getIVs()[0]), t.x);
  Value oy = mul(toI32(grid.getIVs()[1]), t.y);
  Value oz = mul(toI32(grid.getIVs()[2]), t.z);
  // A slot of the middle state that no cell writes holds the middle buffer's own value: dst's,
  // or src's when both buffers are known to agree there.
  Value unwritten = unwrittenFromSrc ? src : dst;

  // Phase 1: every cell of the tile and its halo runs the first step; what it stores into the
  // tile goes to shared memory.
  auto halo1 = scf::ForOp::create(b, loc, tid, cidx(halo), cidx(t.threads));
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(halo1.getBody()->getTerminator());
    Value h = toI32(halo1.getInductionVar());
    Value hxv = remu(h, hx), hyv = remu(divu(h, hx), hy),
          hzv = divu(h, hx * hy);
    Value x = add(ox, add(hxv, c32(-1)));
    Value y = add(oy, add(hyv, c32(-1)));
    Value z = add(oz, add(hzv, c32(-1)));
    // The cell's row. y = -1 and y = planeRows are rows of the neighbouring planes, as in the
    // original indexing; a column outside [0, width) is padding, since rows are further apart
    // than the width.
    Value r = add(y, mul(z, p.planeRows));
    Value valid = both({ge(x, 0), lt(x, p.width), ge(r, 0), lt(r, p.rows)});
    SmallVector<Type> types(nf, p.elemType);
    auto first = scf::IfOp::create(b, loc, types, valid, /*withElse=*/true);
    SmallVector<Value> stored(nf);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToEnd(first.thenBlock());
      BodyCloner bc(b, p);
      bc.map.map(p.src, src);
      bc.map.map(p.dst, dst);
      bc.row = toIndex(r);
      bc.col = toIndex(x);
      bc.onLoad = [&](affine::AffineLoadOp ld) -> Value {
        return memref::LoadOp::create(b, ld.getLoc(),
                                      bc.map.lookupOrDefault(ld.getMemRef()),
                                      ValueRange{bc.index(ld)});
      };
      bc.onStore = [&](affine::AffineStoreOp st) {
        stored[p.fieldOf.lookup(st.getOperation())] =
            bc.map.lookupOrDefault(st.getValueToStore());
      };
      bc.cloneBlock(*p.par.getBody());
      scf::YieldOp::create(b, loc, stored);
      b.setInsertionPointToEnd(first.elseBlock());
      SmallVector<Value> zeros;
      for (Type type : types)
        zeros.push_back(arith::ConstantOp::create(
            b, loc, cast<TypedAttr>(b.getZeroAttr(type))));
      scf::YieldOp::create(b, loc, zeros);
    }
    for (auto [f, field] : llvm::enumerate(p.fields)) {
      Value rx = add(hxv, c32(field.dx - 1));
      Value ry = add(hyv, c32(field.dy - 1));
      Value rz = add(hzv, c32(field.dz - 1));
      Value cx = add(ox, rx), cy = add(oy, ry), cz = add(oz, rz);
      Value inTile = both({ge(rx, 0), lt(rx, t.x), ge(ry, 0), lt(ry, t.y),
                           ge(rz, 0), lt(rz, t.z), lt(cx, p.width),
                           lt(cy, p.planeRows), lt(cz, p.planes)});
      auto route = scf::IfOp::create(b, loc, TypeRange(), inTile,
                                     /*withElse=*/false);
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPoint(route.thenBlock()->getTerminator());
      auto value = scf::IfOp::create(b, loc, TypeRange{p.elemType}, valid,
                                     /*withElse=*/true);
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToEnd(value.thenBlock());
        scf::YieldOp::create(b, loc, ValueRange{first.getResult(f)});
        // No cell writes this slot: it keeps the middle buffer's value.
        b.setInsertionPointToEnd(value.elseBlock());
        auto viewType =
            cast<MemRefType>(memrefOf(field.load.op).getType());
        Value view =
            enzymexla::Pointer2MemrefOp::create(b, loc, viewType, unwritten);
        Value row = toIndex(add(cy, mul(cz, p.planeRows)));
        Value v = memref::LoadOp::create(
            b, loc, view,
            ValueRange{linearIndex(b, loc, field.load, row, toIndex(cx))});
        scf::YieldOp::create(b, loc, ValueRange{v});
      }
      Value slot = toIndex(
          add(c32(f * core), add(rx, mul(add(ry, mul(rz, t.y)), t.x))));
      memref::StoreOp::create(b, loc, value.getResult(0), mid,
                              ValueRange{slot});
    }
  }
  enzymexla::BarrierOp::create(b, loc, ValueRange{tid, cidx(0), cidx(0)});

  // Phase 2: the tile's cells run the second step, reading the first step's values from shared
  // memory and storing to dst as the original does.
  auto core2 = scf::ForOp::create(b, loc, tid, cidx(core), cidx(t.threads));
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(core2.getBody()->getTerminator());
    Value i = core2.getInductionVar();
    Value iv = toI32(i);
    Value x = add(ox, remu(iv, t.x));
    Value y = add(oy, remu(divu(iv, t.x), t.y));
    Value z = add(oz, divu(iv, t.x * t.y));
    auto inside = scf::IfOp::create(
        b, loc, TypeRange(),
        both({lt(x, p.width), lt(y, p.planeRows), lt(z, p.planes)}),
        /*withElse=*/false);
    b.setInsertionPoint(inside.thenBlock()->getTerminator());
    BodyCloner bc(b, p);
    bc.map.map(p.src, src);
    bc.map.map(p.dst, dst);
    bc.row = toIndex(add(y, mul(z, p.planeRows)));
    bc.col = toIndex(x);
    bc.onLoad = [&](affine::AffineLoadOp ld) -> Value {
      int f = p.fieldOf.lookup(ld.getOperation());
      if (f >= 0)
        return memref::LoadOp::create(
            b, ld.getLoc(), mid,
            ValueRange{
                arith::AddIOp::create(b, ld.getLoc(), cidx(f * core), i)});
      Value view = enzymexla::Pointer2MemrefOp::create(
          b, ld.getLoc(), ld.getMemRefType(), unwritten);
      return memref::LoadOp::create(b, ld.getLoc(), view,
                                    ValueRange{bc.index(ld)});
    };
    bc.onStore = [&](affine::AffineStoreOp st) {
      memref::StoreOp::create(b, st.getLoc(),
                              bc.map.lookupOrDefault(st.getValueToStore()),
                              bc.map.lookupOrDefault(st.getMemRef()),
                              ValueRange{bc.index(st)});
    };
    bc.cloneBlock(*p.par.getBody());
  }

  // What fission needs: tile layers, and how far a tile's accesses reach beyond it (the halo it
  // reads plus the neighbours it writes).
  int64_t rowBytes = (p.width - 1) * p.colStride;
  launch->setAttr("split_loop.fused", b.getUnitAttr());
  launch->setAttr("split_loop.tile",
                  b.getDenseI64ArrayAttr({t.x, t.y, t.z, t.threads}));
  launch->setAttr("split_loop.shared_bytes",
                  b.getI64IntegerAttr(nf * core * p.elemBytes));
  launch->setAttr("split_loop.layer_rows",
                  b.getI64IntegerAttr(t.z * p.planeRows));
  launch->setAttr("split_loop.layers", b.getI64IntegerAttr(ntz));
  launch->setAttr("split_loop.row_stride_bytes",
                  b.getI64IntegerAttr(p.rowStride));
  launch->setAttr("split_loop.reach_bytes",
                  b.getI64IntegerAttr(2 * p.storeReach + rowBytes));
  return launch;
}

//===----------------------------------------------------------------------===//
// capture: emission
//===----------------------------------------------------------------------===//

struct Runtime {
  LLVM::LLVMFuncOp create, devSync, begin, end, instantiate, launch, execDestroy,
      graphDestroy, sync, destroy, slrtBegin, slrtPre, slrtPost, slrtEnd,
      slrtDestroy;
};

Runtime declareRuntime(OpBuilder &b, ModuleOp module, bool dag) {
  MLIRContext *ctx = module.getContext();
  Type ptr = LLVM::LLVMPointerType::get(ctx);
  Type i32 = IntegerType::get(ctx, 32), i64 = IntegerType::get(ctx, 64);
  Type none = LLVM::LLVMVoidType::get(ctx);
  Runtime r;
  r.create = declareFunc(b, module, "cudaStreamCreateWithFlags", i32, {ptr, i32});
  r.devSync = declareFunc(b, module, "cudaDeviceSynchronize", i32, {});
  r.launch = declareFunc(b, module, "cudaGraphLaunch", i32, {ptr, ptr});
  r.sync = declareFunc(b, module, "cudaStreamSynchronize", i32, {ptr});
  r.destroy = declareFunc(b, module, "cudaStreamDestroy", i32, {ptr});
  if (dag) {
    r.slrtBegin = declareFunc(b, module, "slrt_begin", ptr, {ptr, i64, i64, i64});
    r.slrtPre = declareFunc(b, module, "slrt_pre", none, {ptr, i64, i64});
    r.slrtPost = declareFunc(b, module, "slrt_post", none, {ptr, i64, i64});
    r.slrtEnd = declareFunc(b, module, "slrt_end", ptr, {ptr});
    r.slrtDestroy = declareFunc(b, module, "slrt_destroy", none, {ptr});
  } else {
    r.begin = declareFunc(b, module, "cudaStreamBeginCapture", i32, {ptr, i32});
    r.end = declareFunc(b, module, "cudaStreamEndCapture", i32, {ptr, ptr});
    r.instantiate =
        declareFunc(b, module, "cudaGraphInstantiate", i32, {ptr, ptr, i64});
    r.execDestroy = declareFunc(b, module, "cudaGraphExecDestroy", i32, {ptr});
    r.graphDestroy = declareFunc(b, module, "cudaGraphDestroy", i32, {ptr});
  }
  return r;
}

/// Emits the band loop's body for the band starting at `bandStart`, with every launch sent to
/// `stream` (and, if `slrtCtx`, bracketed by slrt_pre / slrt_post).
LogicalResult emitBand(OpBuilder &b, scf::ForOp band, Value bandStart,
                       Value stream, Value slrtCtx, const Runtime &rt) {
  IRMapping map;
  map.map(band.getInductionVar(), bandStart);
  SmallVector<Operation *> clones;
  for (Operation &op : band.getBody()->without_terminator())
    clones.push_back(b.clone(op, map));
  SmallVector<enzymexla::GPUWrapperOp> launches;
  for (Operation *c : clones)
    c->walk([&](enzymexla::GPUWrapperOp w) { launches.push_back(w); });
  MLIRContext *ctx = band.getContext();
  Type i64 = IntegerType::get(ctx, 64);
  for (enzymexla::GPUWrapperOp w : launches) {
    OpBuilder lb(w);
    Location loc = w.getLoc();
    Value j64, i64v;
    if (slrtCtx) {
      auto stepLoop = w->getParentOfType<scf::ForOp>();
      auto waveLoop = stepLoop ? stepLoop->getParentOfType<scf::ForOp>()
                               : scf::ForOp();
      if (!stepLoop || !waveLoop || !stepLoop->hasAttr("split_loop.step") ||
          !waveLoop->hasAttr("split_loop.wave"))
        return failure();
      Value j = stepLoop.getInductionVar();
      Value i = arith::SubIOp::create(lb, loc, waveLoop.getInductionVar(), j);
      j64 = arith::IndexCastOp::create(lb, loc, i64, j);
      i64v = arith::IndexCastOp::create(lb, loc, i64, i);
      call(lb, loc, rt.slrtPre, {slrtCtx, j64, i64v});
    }
    Value token = enzymexla::StreamToTokenOp::create(
        lb, loc, async::TokenType::get(ctx), stream);
    auto exec = async::ExecuteOp::create(lb, loc, TypeRange(),
                                         ValueRange{token}, ValueRange());
    Block *execBody = exec.getBody();
    if (execBody->empty() ||
        !execBody->back().hasTrait<OpTrait::IsTerminator>()) {
      OpBuilder yb = OpBuilder::atBlockEnd(execBody);
      async::YieldOp::create(yb, loc, ValueRange());
    }
    w->moveBefore(execBody->getTerminator());
    if (slrtCtx) {
      lb.setInsertionPointAfter(exec);
      call(lb, loc, rt.slrtPost, {slrtCtx, j64, i64v});
    }
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// split_loop.normalize_pingpong
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::SplitLoopNormalizePingPongOp::apply(
    transform::TransformRewriter &rewriter, transform::TransformResults &results,
    transform::TransformState &state) {
  SmallVector<PingPong> matches;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto loop = dyn_cast<scf::WhileOp>(target);
    if (!loop)
      return emitSilenceableError() << "expected an scf.while";
    std::string why;
    FailureOr<PingPong> m = matchPingPong(loop, why);
    if (failed(m))
      return emitSilenceableError() << "not a ping-pong loop: " << why;
    matches.push_back(*m);
  }

  SmallVector<Operation *> stepLoops, launches;
  for (PingPong &m : matches) {
    Location loc = m.loop.getLoc();
    MLIRContext *ctx = m.loop.getContext();
    // The old exit handling, collected before anything is inserted after the loop.
    Operation *term = m.loop->getBlock()->getTerminator();
    SmallVector<Operation *> post;
    for (Operation *op = m.loop->getNextNode(); op != term;
         op = op->getNextNode())
      post.push_back(op);
    rewriter.setInsertionPoint(m.loop);

    // for (s = 0; s < 2 * tripcount; s++) launch A with the buffers chosen by parity.
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value c2 = arith::ConstantIndexOp::create(rewriter, loc, 2);
    Value tc = arith::IndexCastUIOp::create(rewriter, loc,
                                            rewriter.getIndexType(), m.tripCount);
    Value steps = arith::MulIOp::create(rewriter, loc, tc, c2);
    auto stepLoop = scf::ForOp::create(rewriter, loc, c0, steps, c1);
    stepLoop->setAttr("split_loop.steps", rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(stepLoop.getBody()->getTerminator());
      Value s = stepLoop.getInductionVar();
      Value parity = arith::RemUIOp::create(rewriter, loc, s, c2);
      Value even = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                         parity, c0);
      Value src = LLVM::SelectOp::create(rewriter, loc, even, m.src, m.dst);
      Value dst = LLVM::SelectOp::create(rewriter, loc, even, m.dst, m.src);
      IRMapping map;
      map.map(m.src, src);
      map.map(m.dst, dst);
      Operation *launch = rewriter.clone(*m.launchA, map);
      launches.push_back(launch);
    }
    stepLoops.push_back(stepLoop);

    // One error check after the loop, reporting the way the loop did.
    auto getLastError = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
        m.loop, StringAttr::get(ctx, "cudaGetLastError"));
    if (!getLastError)
      return emitDefiniteFailure() << "cudaGetLastError is not declared";
    Value err = LLVM::CallOp::create(rewriter, loc, getLastError, ValueRange())
                    .getResult();
    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value bad = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                      err, zero);
    auto report = scf::IfOp::create(rewriter, loc, bad, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(report.thenBlock()->getTerminator());
      IRMapping map;
      map.map(m.errorValue, err);
      for (Operation &op : m.errorReport->without_terminator())
        rewriter.clone(op, map);
    }

    // The status the enclosing region yields is 0 on the error-free path.
    for (unsigned idx : m.zeroOperands) {
      rewriter.setInsertionPoint(term);
      Value z = arith::ConstantIntOp::create(
          rewriter, loc, term->getOperand(idx).getType(), 0);
      rewriter.modifyOpInPlace(term, [&] { term->setOperand(idx, z); });
    }
    // Drop the old exit handling, then the loop.
    for (Operation *op : llvm::reverse(post)) {
      if (!op->use_empty())
        return emitDefiniteFailure() << "a result of the old exit handling is "
                                        "still used";
      rewriter.eraseOp(op);
    }
    rewriter.eraseOp(m.loop);
  }
  results.set(cast<OpResult>(getStepLoop()), stepLoops);
  results.set(cast<OpResult>(getLaunch()), launches);
  return DiagnosedSilenceableFailure::success();
}

void transform::SplitLoopNormalizePingPongOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// split_loop.fuse_pairs
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::SplitLoopFusePairsOp::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
                                       transform::TransformState &state) {
  FailureOr<int64_t> tx = getIntParam(state, getTileX()),
                     ty = getIntParam(state, getTileY()),
                     tz = getIntParam(state, getTileZ()),
                     nt = getIntParam(state, getThreads());
  if (failed(tx) || failed(ty) || failed(tz) || failed(nt) || *tx < 1 ||
      *ty < 1 || *tz < 1 || *nt < 1)
    return emitSilenceableError()
           << "the tile sizes and threads must each be one positive integer";
  if (*nt > 1024)
    return emitSilenceableError()
           << "threads = " << *nt << " is more than a block can have (1024)";
  bool unwrittenEqual = false, dstDead = false;
  if (std::optional<ArrayAttr> facts = getAssume())
    for (Attribute a : *facts) {
      StringRef fact = cast<StringAttr>(a).getValue();
      if (fact == "unwritten_slots_equal")
        unwrittenEqual = true;
      else if (fact == "dst_dead_after_loop")
        dstDead = true;
      else
        return emitSilenceableError()
               << "unknown host fact \"" << fact
               << "\" (known: unwritten_slots_equal, dst_dead_after_loop)";
    }
  FuseTile tile{*tx, *ty, *tz, *nt};

  SmallVector<FusePlan, 2> plans;
  for (Operation *target : state.getPayloadOps(getStepLoop())) {
    auto loop = dyn_cast<scf::ForOp>(target);
    if (!loop)
      return emitSilenceableError() << "expected an scf.for over steps";
    std::string why;
    FailureOr<FusePlan> p = planFusePairs(loop, why);
    if (failed(p))
      return emitSilenceableError() << "cannot fuse the steps: " << why;
    int64_t shared =
        (int64_t)p->fields.size() * tile.x * tile.y * tile.z * p->elemBytes;
    std::string arch;
    int64_t limit = maxSharedBytesPerBlock(p->launch, arch);
    if (shared > limit)
      return emitSilenceableError()
             << "a " << tile.x << "x" << tile.y << "x" << tile.z
             << " tile keeps " << shared << " bytes of the middle state ("
             << p->fields.size()
             << " written fields) in shared memory; a block can have at most "
             << limit << " on " << arch;
    if ((p->rows + 2 * p->planeRows + 4) * p->pitch >= (int64_t(1) << 31))
      return emitSilenceableError()
             << "the grid is too large for the 32-bit cell arithmetic of the "
                "fused kernel";
    plans.push_back(std::move(*p));
  }

  SmallVector<Operation *> pairLoops, fusedLaunches;
  for (FusePlan &p : plans) {
    Location loc = p.stepLoop.getLoc();
    rewriter.setInsertionPoint(p.stepLoop);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value c2 = arith::ConstantIndexOp::create(rewriter, loc, 2);
    // P pairs, P even so that a holds the result after them. Without the fact that b is dead
    // afterwards, at least one pair is left to the stock launches, which leave b as the source
    // loop does.
    Value total = p.stepLoop.getUpperBound(); // 2 * tripcount
    Value tc = arith::DivUIOp::create(rewriter, loc, total, c2);
    Value limit = dstDead ? tc : arith::SubIOp::create(rewriter, loc, tc, c1);
    Value pairs = arith::MulIOp::create(
        rewriter, loc, arith::DivUIOp::create(rewriter, loc, limit, c2), c2);
    auto pairLoop = scf::ForOp::create(rewriter, loc, c0, pairs, c1);
    pairLoop->setAttr("split_loop.steps", rewriter.getUnitAttr());
    pairLoop->setAttr("split_loop.pairs", rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(pairLoop.getBody()->getTerminator());
      Value parity = arith::RemUIOp::create(rewriter, loc,
                                            pairLoop.getInductionVar(), c2);
      Value even = arith::CmpIOp::create(rewriter, loc,
                                         arith::CmpIPredicate::eq, parity, c0);
      Value src = LLVM::SelectOp::create(rewriter, loc, even, p.bufA, p.bufB);
      Value dst = LLVM::SelectOp::create(rewriter, loc, even, p.bufB, p.bufA);
      auto fused =
          emitFusedLaunch(rewriter, p, src, dst, tile, unwrittenEqual);
      if (std::optional<ArrayAttr> facts = getAssume())
        fused->setAttr("split_loop.assume", *facts);
      fusedLaunches.push_back(fused);
    }
    // The stock steps left start where the pairs end.
    Value tailStart = arith::MulIOp::create(rewriter, loc, pairs, c2);
    rewriter.modifyOpInPlace(p.stepLoop, [&] {
      p.stepLoop.getLowerBoundMutable().assign(tailStart);
      p.stepLoop->removeAttr("split_loop.steps");
      p.stepLoop->setAttr("split_loop.tail", rewriter.getUnitAttr());
    });
    pairLoops.push_back(pairLoop);
  }
  results.set(cast<OpResult>(getPairLoop()), pairLoops);
  results.set(cast<OpResult>(getFusedLaunch()), fusedLaunches);
  return DiagnosedSilenceableFailure::success();
}

void transform::SplitLoopFusePairsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getStepLoopMutable(), effects);
  onlyReadsHandle(getTileXMutable(), effects);
  onlyReadsHandle(getTileYMutable(), effects);
  onlyReadsHandle(getTileZMutable(), effects);
  onlyReadsHandle(getThreadsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// split_loop.fission
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::SplitLoopFissionOp::apply(transform::TransformRewriter &rewriter,
                                     transform::TransformResults &results,
                                     transform::TransformState &state) {
  FailureOr<int64_t> slabRows = getIntParam(state, getSlabRows());
  if (failed(slabRows) || *slabRows <= 0)
    return emitSilenceableError() << "slab_rows must be one positive integer";
  SmallVector<FissionPlan> plans;
  for (Operation *target : state.getPayloadOps(getLaunch())) {
    auto launch = dyn_cast<enzymexla::GPUWrapperOp>(target);
    if (!launch)
      return emitSilenceableError() << "expected an enzymexla.gpu_wrapper";
    std::string why;
    FailureOr<FissionPlan> p = launch->hasAttr("split_loop.fused")
                                   ? planFusedFission(launch, why)
                                   : planFission(launch, why);
    if (failed(p))
      return emitSilenceableError() << "cannot cut the launch into slabs: "
                                    << why;
    if (p->fused) {
      if (*slabRows % p->layerRows ||
          p->layers % (*slabRows / p->layerRows))
        return emitSilenceableError()
               << "slab_rows = " << *slabRows
               << " must be a whole number of the fused kernel's tile layers ("
               << p->layerRows << " rows each) that divides its " << p->layers
               << " layers";
    } else if (*slabRows % p->gridX || p->rows % *slabRows) {
      return emitSilenceableError()
             << "slab_rows = " << *slabRows << " must be a multiple of grid.x = "
             << p->gridX << " that divides the " << p->rows << " rows";
    }
    if (*slabRows < p->conflictRows)
      return emitSilenceableError()
             << "slabs of " << *slabRows << " rows are thinner than the "
             << p->conflictRows << " rows the accesses reach ("
             << p->reach << " bytes): a slab would depend on slabs beyond its "
                            "neighbours";
    plans.push_back(*p);
  }

  SmallVector<Operation *> slabLoops, slabLaunches;
  for (FissionPlan &p : plans) {
    Location loc = p.launch.getLoc();
    MLIRContext *ctx = p.launch.getContext();
    rewriter.setInsertionPoint(p.launch);
    int64_t nslab = p.rows / *slabRows;
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value cn = arith::ConstantIndexOp::create(rewriter, loc, nslab);
    Value shift, gridY;
    if (!p.fused) {
      shift = arith::ConstantIndexOp::create(
          rewriter, loc, *slabRows * p.rowStride / p.gepElemBytes);
      gridY = arith::ConstantIndexOp::create(rewriter, loc, *slabRows / p.gridX);
    }
    auto slabLoop = scf::ForOp::create(rewriter, loc, c0, cn, c1);
    slabLoop->setAttr("split_loop.nslab", rewriter.getI64IntegerAttr(nslab));
    slabLoop->setAttr("split_loop.slab_rows",
                      rewriter.getI64IntegerAttr(*slabRows));
    slabLoop->setAttr("split_loop.row_stride_bytes",
                      rewriter.getI64IntegerAttr(p.rowStride));
    slabLoop->setAttr("split_loop.reach_bytes",
                      rewriter.getI64IntegerAttr(p.reach));
    slabLoop->setAttr("split_loop.conflict_rows",
                      rewriter.getI64IntegerAttr(p.conflictRows));
    if (p.fused) {
      // Slab i runs tile layers [i * perSlab, (i + 1) * perSlab): the grid's layer index is
      // offset inside the kernel, and the pointers stay (boundary tests use absolute cells).
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(slabLoop.getBody()->getTerminator());
      int64_t perSlab = *slabRows / p.layerRows;
      Value layersPerSlab = arith::ConstantIndexOp::create(rewriter, loc, perSlab);
      Value first = arith::MulIOp::create(
          rewriter, loc, slabLoop.getInductionVar(), layersPerSlab);
      auto launch = cast<enzymexla::GPUWrapperOp>(rewriter.clone(*p.launch));
      launch->setOperand(2, layersPerSlab);
      auto grid = cast<affine::AffineParallelOp>(
          &launch.getRegion().front().front());
      std::optional<SmallVector<int64_t, 8>> ranges = grid.getConstantRanges();
      grid.setUpperBounds(
          ValueRange(),
          AffineMap::get(0, 0,
                         {getAffineConstantExpr((*ranges)[0], ctx),
                          getAffineConstantExpr((*ranges)[1], ctx),
                          getAffineConstantExpr(perSlab, ctx)},
                         ctx));
      affine::AffineParallelOp block;
      for (Operation &op : *grid.getBody())
        if ((block = dyn_cast<affine::AffineParallelOp>(&op)))
          break;
      Value layer = grid.getIVs()[2];
      rewriter.setInsertionPointToStart(block.getBody());
      Value shifted = arith::AddIOp::create(rewriter, loc, layer, first);
      layer.replaceAllUsesExcept(shifted, shifted.getDefiningOp());
      slabLaunches.push_back(launch);
    } else {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(slabLoop.getBody()->getTerminator());
      Value off =
          arith::MulIOp::create(rewriter, loc, slabLoop.getInductionVar(), shift);
      Value off64 = arith::IndexCastOp::create(rewriter, loc,
                                               rewriter.getI64Type(), off);
      IRMapping map;
      for (Value buf : p.buffers) {
        Value shifted = LLVM::GEPOp::create(rewriter, loc, buf.getType(),
                                            p.gepElem, buf, ValueRange{off64});
        map.map(buf, shifted);
      }
      auto launch = cast<enzymexla::GPUWrapperOp>(rewriter.clone(*p.launch, map));
      launch->setOperand(1, gridY);
      auto par = cast<affine::AffineParallelOp>(
          &launch.getRegion().front().front());
      AffineMap ub = AffineMap::get(
          0, 0,
          {getAffineConstantExpr(*slabRows, ctx),
           getAffineConstantExpr(p.width, ctx)},
          ctx);
      par.setUpperBounds(ValueRange(), ub);
      slabLaunches.push_back(launch);
    }
    rewriter.eraseOp(p.launch);
    slabLoops.push_back(slabLoop);
  }
  results.set(cast<OpResult>(getSlabLoop()), slabLoops);
  results.set(cast<OpResult>(getSlabLaunch()), slabLaunches);
  return DiagnosedSilenceableFailure::success();
}

void transform::SplitLoopFissionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getLaunchMutable(), effects);
  onlyReadsHandle(getSlabRowsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// split_loop.skew
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::SplitLoopSkewOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {
  FailureOr<int64_t> band = getIntParam(state, getBand());
  if (failed(band) || *band <= 0)
    return emitSilenceableError() << "band must be one positive integer";
  auto steps = llvm::to_vector(state.getPayloadOps(getStepLoop()));
  // The slab loop ends the step loop's body.
  auto slabLoopOf = [](scf::ForOp stepLoop) -> scf::ForOp {
    Operation *last = stepLoop.getBody()->getTerminator()->getPrevNode();
    return dyn_cast_or_null<scf::ForOp>(last);
  };
  for (Operation *stepOp : steps) {
    auto stepLoop = dyn_cast<scf::ForOp>(stepOp);
    if (!stepLoop)
      return emitSilenceableError() << "expected an scf.for over steps";
    scf::ForOp slabLoop = slabLoopOf(stepLoop);
    if (!slabLoop)
      return emitSilenceableError()
             << "the step loop's body does not end with a loop over slabs";
    auto nslabAttr = slabLoop->getAttrOfType<IntegerAttr>("split_loop.nslab");
    if (!nslabAttr)
      return emitSilenceableError()
             << "the slab loop does not come from split_loop.fission, so its "
                "slabs are not known to depend only on their neighbours";
    if (getConstInt(stepLoop.getLowerBound()) != 0 ||
        getConstInt(stepLoop.getStep()) != 1 ||
        getConstInt(slabLoop.getLowerBound()) != 0 ||
        getConstInt(slabLoop.getStep()) != 1 ||
        getConstInt(slabLoop.getUpperBound()) != nslabAttr.getInt() ||
        stepLoop.getNumRegionIterArgs() || slabLoop.getNumRegionIterArgs())
      return emitSilenceableError() << "expected unit-stride loops from 0 "
                                       "without loop-carried values";
    for (Operation &op : stepLoop.getBody()->without_terminator()) {
      if (&op == slabLoop.getOperation())
        break;
      if (op.getNumRegions() || !isMemoryEffectFree(&op))
        return emitSilenceableError() << "the step loop does more than compute "
                                         "the step's buffers: "
                                      << op.getName();
    }
  }

  SmallVector<Operation *> bandLoops;
  for (Operation *stepOp : steps) {
    auto stepLoop = cast<scf::ForOp>(stepOp);
    scf::ForOp slabLoop = slabLoopOf(stepLoop);
    int64_t nslab =
        slabLoop->getAttrOfType<IntegerAttr>("split_loop.nslab").getInt();
    Location loc = stepLoop.getLoc();
    rewriter.setInsertionPoint(stepLoop);
    Value total = stepLoop.getUpperBound();
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value ck = arith::ConstantIndexOp::create(rewriter, loc, *band);
    Value cn = arith::ConstantIndexOp::create(rewriter, loc, nslab);
    Value cw = arith::ConstantIndexOp::create(rewriter, loc, nslab + *band - 1);
    auto bandLoop = scf::ForOp::create(rewriter, loc, c0, total, ck);
    bandLoop->setAttr("split_loop.band", rewriter.getI64IntegerAttr(*band));
    bandLoop->setAttr("split_loop.nslab", rewriter.getI64IntegerAttr(nslab));
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(bandLoop.getBody()->getTerminator());
    Value rem = arith::SubIOp::create(rewriter, loc, total,
                                      bandLoop.getInductionVar());
    Value kk = arith::MinUIOp::create(rewriter, loc, rem, ck);
    auto waveLoop = scf::ForOp::create(rewriter, loc, c0, cw, c1);
    waveLoop->setAttr("split_loop.wave", rewriter.getUnitAttr());
    rewriter.setInsertionPoint(waveLoop.getBody()->getTerminator());
    auto jLoop = scf::ForOp::create(rewriter, loc, c0, kk, c1);
    jLoop->setAttr("split_loop.step", rewriter.getUnitAttr());
    rewriter.setInsertionPoint(jLoop.getBody()->getTerminator());
    Value i = arith::SubIOp::create(rewriter, loc, waveLoop.getInductionVar(),
                                    jLoop.getInductionVar());
    Value ge =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge, i, c0);
    Value lt =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt, i, cn);
    Value inside = arith::AndIOp::create(rewriter, loc, ge, lt);
    auto ifOp = scf::IfOp::create(rewriter, loc, inside, /*withElse=*/false);
    rewriter.setInsertionPoint(ifOp.thenBlock()->getTerminator());
    Value s = arith::AddIOp::create(rewriter, loc, bandLoop.getInductionVar(),
                                    jLoop.getInductionVar());
    IRMapping map;
    map.map(stepLoop.getInductionVar(), s);
    map.map(slabLoop.getInductionVar(), i);
    for (Operation &op : stepLoop.getBody()->without_terminator()) {
      if (&op == slabLoop.getOperation())
        break;
      rewriter.clone(op, map);
    }
    for (Operation &op : slabLoop.getBody()->without_terminator())
      rewriter.clone(op, map);
    rewriter.eraseOp(stepLoop);
    bandLoops.push_back(bandLoop);
  }
  results.set(cast<OpResult>(getBandLoop()), bandLoops);
  return DiagnosedSilenceableFailure::success();
}

void transform::SplitLoopSkewOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getStepLoopMutable(), effects);
  onlyReadsHandle(getBandMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// split_loop.capture
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::SplitLoopCaptureOp::apply(transform::TransformRewriter &rewriter,
                                     transform::TransformResults &results,
                                     transform::TransformState &state) {
  bool dag = getEdges() == "dag";
  if (!dag && getEdges() != "chain")
    return emitSilenceableError() << "edges must be \"chain\" or \"dag\"";
  int64_t backpressure = 0;
  if (dag) {
    if (!getBackpressure())
      return emitSilenceableError() << "edges = \"dag\" needs a backpressure";
    FailureOr<int64_t> w = getIntParam(state, getBackpressure());
    if (failed(w) || *w < 2)
      return emitSilenceableError()
             << "backpressure must be at least 2 (with 1, step j would wait for "
                "a launch of step j+1 captured after it)";
    backpressure = *w;
  }
  auto bands = llvm::to_vector(state.getPayloadOps(getBandLoop()));
  for (Operation *op : bands) {
    auto bandLoop = dyn_cast<scf::ForOp>(op);
    auto k = bandLoop ? bandLoop->getAttrOfType<IntegerAttr>("split_loop.band")
                      : IntegerAttr();
    if (!k || !bandLoop->hasAttr("split_loop.nslab"))
      return emitSilenceableError() << "expected a band loop from split_loop.skew";
    if (k.getInt() % 2)
      return emitSilenceableError()
             << "the band must be even: one graph serves every band only if "
                "every band starts on the same buffer";
  }

  for (Operation *op : bands) {
    auto bandLoop = cast<scf::ForOp>(op);
    int64_t k = bandLoop->getAttrOfType<IntegerAttr>("split_loop.band").getInt();
    int64_t nslab =
        bandLoop->getAttrOfType<IntegerAttr>("split_loop.nslab").getInt();
    MLIRContext *ctx = bandLoop.getContext();
    Location loc = bandLoop.getLoc();
    auto module = bandLoop->getParentOfType<ModuleOp>();
    Runtime rt = declareRuntime(rewriter, module, dag);
    Type ptr = LLVM::LLVMPointerType::get(ctx);
    rewriter.setInsertionPoint(bandLoop);

    // A stream of our own, after the default-stream work before the loop.
    Value one64 = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                           rewriter.getI64IntegerAttr(1));
    Value streamSlot = LLVM::AllocaOp::create(rewriter, loc, ptr, ptr, one64);
    Value graphSlot, execSlot;
    if (!dag) {
      graphSlot = LLVM::AllocaOp::create(rewriter, loc, ptr, ptr, one64);
      execSlot = LLVM::AllocaOp::create(rewriter, loc, ptr, ptr, one64);
    }
    Value nonBlocking = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    call(rewriter, loc, rt.create, {streamSlot, nonBlocking});
    Value stream = LLVM::LoadOp::create(rewriter, loc, ptr, streamSlot);
    call(rewriter, loc, rt.devSync, {});

    Value total = bandLoop.getUpperBound(), ck = bandLoop.getStep();
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value full = arith::DivUIOp::create(rewriter, loc, total, ck);
    Value fullSteps = arith::MulIOp::create(rewriter, loc, full, ck);
    Value hasFull = arith::CmpIOp::create(rewriter, loc,
                                          arith::CmpIPredicate::ugt, full, c0);
    auto ifFull = scf::IfOp::create(rewriter, loc, hasFull, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(ifFull.thenBlock()->getTerminator());
      Value slrtCtx, exec, graph;
      if (dag) {
        auto i64c = [&](int64_t v) -> Value {
          return arith::ConstantIntOp::create(rewriter, loc, v, 64);
        };
        slrtCtx = call(rewriter, loc, rt.slrtBegin,
                       {stream, i64c(nslab), i64c(k), i64c(backpressure)});
      } else {
        Value relaxed = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
        call(rewriter, loc, rt.begin, {stream, relaxed});
      }
      if (failed(emitBand(rewriter, bandLoop, c0, stream, slrtCtx, rt)))
        return emitDefiniteFailure() << "the band loop's launches are not in "
                                        "the nest split_loop.skew builds";
      if (dag) {
        exec = call(rewriter, loc, rt.slrtEnd, {slrtCtx});
      } else {
        call(rewriter, loc, rt.end, {stream, graphSlot});
        graph = LLVM::LoadOp::create(rewriter, loc, ptr, graphSlot);
        Value noFlags = LLVM::ConstantOp::create(
            rewriter, loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
        call(rewriter, loc, rt.instantiate, {execSlot, graph, noFlags});
        exec = LLVM::LoadOp::create(rewriter, loc, ptr, execSlot);
      }
      auto replay = scf::ForOp::create(rewriter, loc, c0, full, c1);
      {
        OpBuilder::InsertionGuard inner(rewriter);
        rewriter.setInsertionPoint(replay.getBody()->getTerminator());
        call(rewriter, loc, rt.launch, {exec, stream});
      }
      if (dag) {
        call(rewriter, loc, rt.slrtDestroy, {slrtCtx});
      } else {
        call(rewriter, loc, rt.execDestroy, {exec});
        call(rewriter, loc, rt.graphDestroy, {graph});
      }
    }

    // A partial last band, as plain launches on the stream.
    rewriter.setInsertionPointAfter(ifFull);
    Value rem = arith::SubIOp::create(rewriter, loc, total, fullSteps);
    Value hasRem =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ugt, rem, c0);
    auto ifRem = scf::IfOp::create(rewriter, loc, hasRem, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(ifRem.thenBlock()->getTerminator());
      if (failed(emitBand(rewriter, bandLoop, fullSteps, stream, Value(), rt)))
        return emitDefiniteFailure() << "unexpected band structure";
    }
    rewriter.setInsertionPointAfter(ifRem);
    call(rewriter, loc, rt.sync, {stream});
    call(rewriter, loc, rt.destroy, {stream});
    rewriter.eraseOp(bandLoop);
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::SplitLoopCaptureOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getOperation()->getOpOperands().take_front(1), effects);
  onlyReadsHandle(getOperation()->getOpOperands().drop_front(1), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace {
class SplitLoopTransformExtension
    : public transform::TransformDialectExtension<SplitLoopTransformExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitLoopTransformExtension)
  using Base::Base;

  void init() {
    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<async::AsyncDialect>();
    declareGeneratedDialect<LLVM::LLVMDialect>();
    declareGeneratedDialect<memref::MemRefDialect>();
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<enzymexla::EnzymeXLADialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/TransformOps/SplitLoopTransformOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::enzyme::registerSplitLoopTransformExtension(
    DialectRegistry &registry) {
  registry.addExtensions<SplitLoopTransformExtension>();
  transform::registerTuneExtension(registry);
}
