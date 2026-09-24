// [TD] GraphBuilder OpGraph install helpers
    void rebuildTensorNameIndex();
    static int remapId(int oldId, const std::unordered_map<int, int>& m);
    static std::vector<int> remapIds(const std::vector<int>& old,
                                     const std::unordered_map<int, int>& m);
