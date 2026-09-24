// =============================================================================
// [TD] GraphBuilder OpGraph install APIs (from sync_td_headers.py)
// Requires members: g_, tensorIdByName_, fusionGroups_
// =============================================================================

void __GB__::rebuildTensorNameIndex() {
    tensorIdByName_.clear();
    for (const auto& t : g_.tensors) {
        if (!t.name.empty()) {
            tensorIdByName_[t.name] = t.id;
        }
    }
}

int __GB__::findTensorId(const std::string& name) const {
    auto it = tensorIdByName_.find(name);
    if (it == tensorIdByName_.end()) {
        throw std::runtime_error("GraphBuilder: tensor name not found: " + name);
    }
    return it->second;
}

bool __GB__::hasTensorName(const std::string& name) const {
    return tensorIdByName_.count(name) != 0;
}

void __GB__::loadOpGraph(OpGraph og) {
    for (size_t i = 0; i < og.tensors.size(); ++i) {
        if (og.tensors[i].id < 0) {
            og.tensors[i].id = static_cast<int>(i);
        }
    }
    for (size_t i = 0; i < og.ops.size(); ++i) {
        if (og.ops[i].id < 0) {
            og.ops[i].id = static_cast<int>(i);
        }
    }
    g_ = std::move(og);
    fusionGroups_.clear();
    rebuildTensorNameIndex();
}

int __GB__::remapId(int oldId, const std::unordered_map<int, int>& m) {
    auto it = m.find(oldId);
    if (it == m.end()) {
        throw std::runtime_error(
            "GraphBuilder::mergeOpGraph: missing remap for tensor id " +
            std::to_string(oldId));
    }
    return it->second;
}

std::vector<int> __GB__::remapIds(const std::vector<int>& old,
                                  const std::unordered_map<int, int>& m) {
    std::vector<int> out;
    out.reserve(old.size());
    for (int id : old) {
        out.push_back(remapId(id, m));
    }
    return out;
}

std::unordered_map<int, int> __GB__::mergeOpGraph(
    OpGraph og,
    const std::unordered_map<std::string, int>& existingByName) {
    std::unordered_map<int, int> idMap;

    for (auto& t : og.tensors) {
        const int oldId = (t.id >= 0) ? t.id : -1;
        if (oldId < 0) {
            throw std::runtime_error("mergeOpGraph: tensor missing id");
        }

        auto ex = existingByName.find(t.name);
        if (ex != existingByName.end()) {
            idMap[oldId] = ex->second;
            if (ex->second >= 0 &&
                ex->second < static_cast<int>(g_.tensors.size()) &&
                t.desc.poolHint == MemoryPool::External) {
                g_.tensors[ex->second].desc.externalAddr = t.desc.externalAddr;
                g_.tensors[ex->second].desc.poolHint = MemoryPool::External;
            }
            continue;
        }

        if (hasTensorName(t.name)) {
            idMap[oldId] = findTensorId(t.name);
            continue;
        }

        Tensor nt = t;
        nt.id = static_cast<int>(g_.tensors.size());
        idMap[oldId] = nt.id;
        if (!nt.name.empty()) {
            tensorIdByName_[nt.name] = nt.id;
        }
        g_.tensors.push_back(std::move(nt));
    }

    const int opBase = static_cast<int>(g_.ops.size());
    std::unordered_map<int, int> opIdMap;
    for (size_t i = 0; i < og.ops.size(); ++i) {
        opIdMap[og.ops[i].id >= 0 ? og.ops[i].id : static_cast<int>(i)] =
            opBase + static_cast<int>(i);
    }

    for (size_t i = 0; i < og.ops.size(); ++i) {
        OpNode op = og.ops[i];
        const int oldOpId = (op.id >= 0) ? op.id : static_cast<int>(i);
        op.id = opIdMap.at(oldOpId);
        op.inputs = remapIds(op.inputs, idMap);
        op.outputs = remapIds(op.outputs, idMap);
        op.workspaces = remapIds(op.workspaces, idMap);

        for (int& d : op.dependOpsId) {
            auto it = opIdMap.find(d);
            if (it != opIdMap.end()) {
                d = it->second;
            }
        }
        for (int& d : op.dependedOpsId) {
            auto it = opIdMap.find(d);
            if (it != opIdMap.end()) {
                d = it->second;
            }
        }

        g_.ops.push_back(std::move(op));
        // Wire tensors/ops alone is not enough: ops::* also runs schema
        // tiling infer (tileDims / dimExtents). Do the same for merged ops.
        inferOpTilingFromSchema(g_.ops.back().id);
    }

    if (!og.tensorInitData.empty()) {
        for (auto& kv : og.tensorInitData) {
            g_.tensorInitData[remapId(kv.first, idMap)] = std::move(kv.second);
        }
    }

    return idMap;
}

void __GB__::inferOpTilingFromSchema(int opId) {
    const OpNode* op = nullptr;
    for (const auto& cand : g_.ops) {
        if (cand.id == opId) {
            op = &cand;
            break;
        }
    }
    if (op == nullptr) {
        throw std::runtime_error(
            "GraphBuilder::inferOpTilingFromSchema: op id not found: " +
            std::to_string(opId));
    }

    // ADAPT (required): call the SAME helper that ops::* /
    // addOpWithShapeInfer use after creating an op of this type.
    // Example: add_rms_norm output "x" must get schema tileDims count=1
    // (last dim reduce) — never invent tileDims from shape.ndim alone.
    //
    // Replace the next line with Ascend's real entry, e.g.:
    //   applyRegisteredOpTiling(*op);
    //   tilingInferencer_.infer(*this, *op);
    //   OpRegistry::get(op->type).inferTiling(*this, *op);
    (void)op;
    throw std::runtime_error(
        "GraphBuilder::inferOpTilingFromSchema: Ascend must wire this to the "
        "same schema tiling infer as ops:: / addOpWithShapeInfer (op.type=" +
        op->type + ", op.id=" + std::to_string(opId) + ")");
}
