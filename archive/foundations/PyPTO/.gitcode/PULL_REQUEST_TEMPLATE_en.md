
PRs with no body should be avoided. The body should be written to convey the intetion of the changes, it should be clear and concise. For example, commits with a tag like `fix` if body is empty, it'll leave the reviewer wondering what exactly is fixed or why the change is necessary.
If the PR related to specific issue, `Related Issue:` could be included as the last part of the body.

Below is an example of a PR body:
```
The origin issue template brings too many noise, it's not suitable for git message. eg: comments
are not removed, labels are redundant as the PR tile already contains the tag.

The guide gives a clear suggestion on how to write the PR body.

Related Issue: #1234,#5678
```

For more details see: [Commit Message](../docs/zh/contribute/pull-request.md#commit-message)
