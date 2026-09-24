// [TD] GraphBuilder OpGraph install APIs
    void loadOpGraph(OpGraph og);
    std::unordered_map<int, int> mergeOpGraph(
        OpGraph og,
        const std::unordered_map<std::string, int>& existingByName = {});
    int findTensorId(const std::string& name) const;
    bool hasTensorName(const std::string& name) const;
    const OpGraph& opGraph() const { return g_; }
    // Same schema tiling infer as ops::* / addOpWithShapeInfer for one op.
    // mergeOpGraph calls this for every newly appended op.
    void inferOpTilingFromSchema(int opId);
