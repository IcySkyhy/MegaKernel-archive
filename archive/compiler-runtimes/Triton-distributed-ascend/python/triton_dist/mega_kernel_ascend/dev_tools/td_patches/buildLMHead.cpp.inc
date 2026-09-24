// [TD] buildLMHead from lmhead_opgraph.expected.bin (merge into gb_)
// Product path: deserialize TD bin + mergeOpGraph.
// Ascend tiling derives dimExtents from op.outputs -> Tensor.desc.shape;
// bin must carry StaticOpResult<3>: y[M,H], x[M,H], rstd[M,1]
// (matmul / finalNorm use outputs[0] == "y").
// mergeOpGraph runs inferOpTilingFromSchema per op (same as ops::).
GraphTensor __CLASS__::buildLMHead(const GraphTensor& mlpOutput,
                                   const GraphTensor& residual) {
    if (gb_ == nullptr) {
        throw std::runtime_error("buildLMHead: GraphBuilder is null");
    }

    const char* kLmHeadBin = "__LMHEAD_BIN_PATH__";
    std::cout << "[TD] start: deserialize OpGraph from " << kLmHeadBin << std::endl;
    OpGraph og = load_opgraph_from_file(kLmHeadBin);
    std::cout << "[TD] end: deserialize OpGraph ok, tensors=" << og.tensors.size()
              << " ops=" << og.ops.size() << std::endl;

    if (og.ops.size() != 2) {
        throw std::runtime_error("lmhead.bin: expect exactly 2 ops, got " +
                                 std::to_string(og.ops.size()));
    }
    if (og.ops[0].type != "add_rms_norm") {
        throw std::runtime_error("lmhead.bin: op0.type want add_rms_norm, got " +
                                 og.ops[0].type);
    }
    if (og.ops[1].type != "matmul_lmhead") {
        throw std::runtime_error("lmhead.bin: op1.type want matmul_lmhead, got " +
                                 og.ops[1].type);
    }
    if (og.ops[0].outputs.size() != 3) {
        throw std::runtime_error(
            "lmhead.bin: add_rms_norm must have 3 outputs (StaticOpResult<3>), got " +
            std::to_string(og.ops[0].outputs.size()));
    }
    if (og.ops[0].outputs.empty() || og.ops[1].inputs.empty() ||
        og.ops[0].outputs[0] != og.ops[1].inputs[0]) {
        throw std::runtime_error(
            "lmhead.bin: add_rms_norm.outputs[0] (y) must equal matmul inputs[0]");
    }

    // Every op output must have desc.shape — tiling dimExtents come from shape.
    auto requireOutputShapes = [](const OpGraph& g) {
        for (const auto& op : g.ops) {
            for (int tid : op.outputs) {
                const Tensor* t = nullptr;
                for (const auto& cand : g.tensors) {
                    if (cand.id == tid) {
                        t = &cand;
                        break;
                    }
                }
                if (t == nullptr && tid >= 0 &&
                    tid < static_cast<int>(g.tensors.size())) {
                    t = &g.tensors[static_cast<size_t>(tid)];
                }
                if (t == nullptr) {
                    throw std::runtime_error(
                        "lmhead.bin: missing tensor for output id of op " + op.type);
                }
                if (t->desc.shape.empty()) {
                    throw std::runtime_error(
                        "lmhead.bin: empty desc.shape on op " + op.type +
                        " output '" + t->name +
                        "' (Ascend tiling dimExtents need non-empty shape)");
                }
            }
        }
    };
    requireOutputShapes(og);

    // Validate Ascend StaticOpResult<3> names/shapes when present in the bin.
    auto tensorById = [](const OpGraph& g, int tid) -> const Tensor* {
        for (const auto& cand : g.tensors) {
            if (cand.id == tid) {
                return &cand;
            }
        }
        if (tid >= 0 && tid < static_cast<int>(g.tensors.size())) {
            return &g.tensors[static_cast<size_t>(tid)];
        }
        return nullptr;
    };
    {
        const Tensor* t0 = tensorById(og, og.ops[0].outputs[0]);
        const Tensor* t1 = tensorById(og, og.ops[0].outputs[1]);
        const Tensor* t2 = tensorById(og, og.ops[0].outputs[2]);
        if (t0 == nullptr || t1 == nullptr || t2 == nullptr) {
            throw std::runtime_error("lmhead.bin: missing add_rms_norm output tensor");
        }
        if (t0->name != "y" || t1->name != "x" || t2->name != "rstd") {
            throw std::runtime_error(
                "lmhead.bin: add_rms_norm outs want names y,x,rstd got '" +
                t0->name + "','" + t1->name + "','" + t2->name + "'");
        }
        if (t0->desc.shape.size() != 2 || t1->desc.shape.size() != 2) {
            throw std::runtime_error(
                "lmhead.bin: y/x must be rank-2 [M,H]");
        }
        if (t2->desc.shape.size() != 2 || t2->desc.shape.back() != 1) {
            throw std::runtime_error(
                "lmhead.bin: rstd must be rank-2 [M,1]");
        }
    }

    for (auto& t : og.tensors) {
        if (t.name == "norm.weight") {
            auto it = weightCache_.find("model.norm.weight");
            if (it == weightCache_.end()) {
                throw std::runtime_error("weightCache missing model.norm.weight");
            }
            t.desc.poolHint = MemoryPool::External;
            t.desc.externalAddr = (void*)it->second;
        } else if (t.name == "lm_head.weight") {
            auto it = weightCache_.find("lm_head.weight");
            if (it == weightCache_.end()) {
                throw std::runtime_error("weightCache missing lm_head.weight");
            }
            t.desc.poolHint = MemoryPool::External;
            t.desc.externalAddr = (void*)it->second;
            if (opsArgs_.ranks > 1 && t.desc.shape.size() >= 2) {
                t.desc.shape[t.desc.shape.size() - 1] =
                    static_cast<int64_t>(vocabSize_ / opsArgs_.ranks);
            }
        } else if (t.name == "lm_head.out" && opsArgs_.ranks > 1 &&
                   t.desc.shape.size() >= 2) {
            t.desc.shape[t.desc.shape.size() - 1] =
                static_cast<int64_t>(vocabSize_ / opsArgs_.ranks);
        }
    }

    std::unordered_map<std::string, int> existing = {
        {"mlp_output", mlpOutput.id()},
        {"residual", residual.id()},
    };
    gb_->mergeOpGraph(std::move(og), existing);

    const OpGraph& g = gb_->opGraph();
    if (g.ops.size() < 2) {
        throw std::runtime_error("buildLMHead: graph missing lmhead ops after merge");
    }
    const int opNorm = g.ops[g.ops.size() - 2].id;
    const int opMatmul = g.ops[g.ops.size() - 1].id;

    // Ascend ops:: uses outputs[0] == "y" as the normed activation into lm_head.
    if (!gb_->hasTensorName("y")) {
        throw std::runtime_error("buildLMHead: missing norm output tensor name 'y'");
    }
    const int normOutId = gb_->findTensorId("y");
    const int logitsId = gb_->findTensorId("lm_head.out");

    finalNormOutput_ = gb_->tensor(normOutId, opNorm);
    GraphTensor lmHeadMatmulOutput = gb_->tensor(logitsId, opMatmul);

    if (opsArgs_.ranks > 1) {
        GraphTensor lmHead_allgather_ranks = gb_->tensor("lmHead_allgather_ranks",
            {opsArgs_.ranks}, DType::BF16).pool(MemoryPool::Recyclable);
        lmHeadMatmulOutput.pool(MemoryPool::ShmemRecyclable);
        mk::OpAttributes allGatherAttrs;
        allGatherAttrs.items["rank"] = static_cast<int64_t>(opsArgs_.rankId);
        allGatherAttrs.items["rank_size"] = static_cast<int64_t>(opsArgs_.ranks);
        allGatherAttrs.items["use_shmem"] = static_cast<int64_t>(1);
        lmHeadMatmulOutput_ = lmHeadMatmulOutput;
        allGatherOutput_ = ops::all_gather(allGatherAttrs, lmHeadMatmulOutput,
                                           lmHead_allgather_ranks);
        return allGatherOutput_;
    }

    lmHeadMatmulOutput_ = lmHeadMatmulOutput;
    return lmHeadMatmulOutput_;
}
