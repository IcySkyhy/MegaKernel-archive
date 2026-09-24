# Submit a Pull Request

## Guidelines

* Send well scoped pull request that are easy to review and revert, merge multiple unrelated changes should be avoided.
* Rebase your branch to most recent version of `master` branch, you can do it by:

  ```bash
  git fetch upstream master
  git rebase FETCH_HEAD
  ```

* Add test cases for feat or fix done in the pull request
* Document the code in the pull request
* Request code reviews from committer or other contributors by @-ing them in the pull request comment

## Commit Message

PR/commit title:

* Guarantee a tile exists for both PR and commit message, and the title should be in format of `tag(scope): [Short summary]`
  * `tag` is one of the following: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `perf`
  * `scope` is the module or component affected by the change, if multiple modules are affected, separate them with `|` or just put the most relevant module, eg `frontend|backend`, if too many modules are affected, just put `all`
  * `summary` is a short description of the change, should be written in English with uppercase begin and no period at the end, imperative mood is more preferred.
  * The title should be in English, and the summary should be in imperative mood.

PR body:

* Describe the change in detail, include motivation and context, and link to related issues if any.
* Remove unrelated template content from the PR body
* If multiple commits exist in the pull request, put a short summary for each commit in PR body

Below is an example of a PR body:

```
feat(interface): Optimize the pypto.cond with concrete value

The origin implementation of pypto.cond generate both if/else branch, even if the condition is always true or false, in this PR, we optimize the implementation to generate only one branch.

Changes:
- Optimize the pypto.cond with concrete value, which can reduce the number of branches in the program
- Update the test cases to cover the new features

Related Issues: #1234,#5678
```

Commit:

* Each commit should done only one thing, and the commit message should be in format of `tag(scope): [Short summary]`, PyPTO use squash merge, so verbose commit message body is not required.
* If multiple commits exist in the pull request, following order is recommended: `fixup` -> `refactor` -> `feat` -> `test`

## CI

* Send `compile` in pull request comment to trigger CI compilation
* The pull request can be merged after CI compilation and `approve` label from committer and `lgtm` label from other contributor
* `code-check` warnings should be fixed before merging, or shielded by the rules defined in [Code Check Rule](./code-check-rule.yaml), New rules can be added in the rule file, and the PR of rules changes should be reviewed and approved by maintainer.

## FAQ

### pre-hooks declined

* The title of commit-msg and PR should follow the rule defined in [Commit Message](#commit-message)

### CLA Signing

* For CLA signing, please refer to the [CLA Usage Guide](https://gitcode.com/cann/infrastructure/blob/main/docs/cla/cla_user_guide.en.md)

### Robot Command Usage

* For commands such as `compile`, please refer to the [Robot Usage Guide](https://gitcode.com/cann/infrastructure/blob/main/docs/robot/robot使用指南.md)
