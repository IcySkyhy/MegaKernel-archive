PR 应该避免没有描述信息，描述信息的最重要的作用是描述修改意图，应该清晰明了。例如：标题中包含了 `fix` 标签，如果没有描述信息，就需要Review人员猜测PR具体修复了哪些内容，或者怀疑修改的必要性。如果PR和特定的Issue相关，可以单独包含一行 `Related Issue:` 放到描述信息最后。

下面是一个典型的PR描述信息:
```
The origin issue template brings too many noise, it's not suitable for git message. eg: comments
are not removed, labels are redundant as the PR tile already contains the tag.

The guide gives a clear suggestion on how to write the PR body.

Related Issue: #1234,#5678
```

更多描述信息见：[Commit Message 规范](../docs/zh/contribute/pull-request.md#commit-message)
