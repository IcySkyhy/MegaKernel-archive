# evidence/hist/ —— 历史 rev 的 regbench 原始 log（**逐字节副本**）

为什么放这里：README §3.6 的「跨批次」表引用了更早两批与本轮 A 组的读数。这些数只存在于
历史 rev 的 evidence 里；为让它们在**当前 tip** 上也能被 grep/复核，把对应 log 原样复制过来
（不做任何改动），并记录来源 rev 与 sha256。

| 文件 | 来源 | sha256 |
|---|---|---|
| `103540f_regbench.log` | `git show 103540f:probe_host_dma/evidence/logs/regbench.log` | `de181d79ed86184a5b4caa5d3fd268cd066cd90071b71f3f7e3ab8cf50028f8e` |
| `5bf1b9b_regbench.log` | `git show 5bf1b9b:probe_host_dma/evidence/logs/regbench.log` | `831fd8b8020f2f633a7b09a7c6cacc8a760771fb1983265f45921f37d9b8be6d` |
| `c126bb6_regbench_run1.log` | `git show c126bb6:probe_host_dma/evidence/logs/regbench_run1.log` | `a6b007a0dc821726afc96790cdfb2def6f949601d9fbeb3e5c968d12f43950cb` |
| `c126bb6_regbench_run2.log` | `git show c126bb6:probe_host_dma/evidence/logs/regbench_run2.log` | `0edb2feca70b5422cd339be71ecce81defa2ab36a9188c3317149a0500010b24` |
| `c126bb6_regbench_run3.log` | `git show c126bb6:probe_host_dma/evidence/logs/regbench_run3.log` | `1355f1197feaa3a86431b5e42ef36ab18b67ba8289754ee71777c56fea0de754` |
| `c126bb6_regbench_run4.log` | `git show c126bb6:probe_host_dma/evidence/logs/regbench_run4.log` | `93607032b9883058d1942003bc458731c389f1a56536f6b4473fe6f9dc7e8f26` |
| `c126bb6_regbench_run5.log` | `git show c126bb6:probe_host_dma/evidence/logs/regbench_run5.log` | `aeb0d127612d935dc4ef3bb320f8ef8601d2136f5027fbf0d7f8cc093dc40d26` |

复核：上表每个 sha256 应与 `git show <rev>:probe_host_dma/evidence/logs/…` 的 sha256 一致。
