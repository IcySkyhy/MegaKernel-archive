# MegaKernelWiki MVP

本目录把 `archive/` 中的 MegaKernel 实现归档转化为一个可查询、可追溯、可继续蒸馏的数据与 Skill 原型。

## 入口

- [调研与设计报告](KERNELWIKI_DISTILLATION_DESIGN_2026-09-01.md)
- [Kernel Design Wiki Skill](kernel-design-wiki/SKILL.md)
- [本体与边界](kernel-design-wiki/references/ontology.md)
- [蒸馏流程](kernel-design-wiki/references/distillation.md)
- [训练数据契约](kernel-design-wiki/references/dataset-contract.md)
- [107 项目录候选清单](generated-catalog-candidates.jsonl)
- [已导出的种子数据](generated-seed-data/train.jsonl)

## 快速使用

在 `kernelwiki/kernel-design-wiki` 下运行：

```bash
python scripts/query.py "B300 固定 batch-1 decode 应选静态 instruction stream 还是 ready queue" --follow-evidence
python scripts/query.py "TeraMoE StreamEP" --kind decision --follow-evidence
python scripts/validate.py
python scripts/build_index.py
python scripts/export_dataset.py --output-dir ../../generated-kernel-data --mode all
```

当前版本是一个深蒸馏 seed：优先覆盖 15 个高价值实现路径、真实边界负例与前沿 agent 工作流。下一阶段应按同一 schema 扩展至全部 A-OSS/B-OSS，再加入其他 source-visible 与相邻路线。

当前验证结果：107 个 catalog candidate 均有本地目录；curated corpus 含 20 个锚点仓库、27 条原子证据、10 条关系和 12 张 Wiki 卡；12 个 seed example 已按 `split_group` 导出为 train/validation/test，未发生谱系组跨 split。
