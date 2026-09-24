# PyPTO安装

> **说明**
>
> 1. 若您使用**CANN9.1.0之前**的版本，请下载与CANN版本对应的PyPTO源码，并参考对应分支下的`build_and_install.md`完成PyPTO的编译安装。针对CANN的`${cann_version}`版本，请参考下表对应关系：
>
>    | CANN版本 | PyPTO版本 |
>    | :--- | :--- |
>    | 8.5.0 | 0.1.2 |
>    | 9.0.0 | 0.2.0 |
>
> 2. 若您使用**CANN9.1.0及之后**的版本，CANN包内已集成PyPTO，CANN安装完成后即可使用PyPTO，可跳过本节。
> 3. 若您需要体验PyPTO最新master版本能力，或基于源码进行PyPTO框架开发，请参考本文档完成源码编译安装。

## 源码下载

请根据CANN软件版本下载对应分支源码，\$\{tag\_version\}表示分支标签名。

```bash
# 下载项目对应分支源码
git clone -b ${tag_version} https://gitcode.com/cann/pypto.git
```

对于WebIDE环境，**已默认提供最新商发版本的项目源码**，如需获取其他版本源码，也需通过上述命令下载源码。

> [!NOTE] 注意
>
> - gitcode平台在使用HTTPS协议的时候要配置并使用个人访问令牌代替登录密码进行克隆、推送等操作。
> - 若您的编译环境无法访问网络，无法通过git指令下载代码，请先在联网环境中下载源码，再手动上传。

## 通过源码编译安装

### 编译run包

进入PyPTO源码根目录，执行如下命令编译run包：

```bash
python3 build_ci.py --clean --no_isolation
```

**参数说明**：

| 参数 | 说明 |
| :--- | :--- |
| `--clean` | 编译前清理构建目录和安装输出目录。 |
| `--no_isolation` | 关闭whl隔离构建模式，构建依赖需提前在当前环境中安装完成。 |
| `--just_build_whl` | 只构建whl包，不打包进入run安装包。 |
| `--py_abi` | 指定whl包的Python ABI tag数字部分，例如`37`对应`cp37`（`--just_build_whl`时有效）。 |
| `--plat_name` | 指定whl包的平台标签，例如`manylinux2014`（`--just_build_whl`时有效）。 |

### 安装

执行上述编译命令后，会在PyPTO源码根目录的`build_out`目录下生成run包，文件名类似`cann-pypto_9.1.0_linux-aarch64.run`。进入`build_out`目录后，执行如下命令完成安装：

```bash
cd build_out
bash ./cann-pypto_${pypto_version}_${os_arch}.run --full -q --pylocal
```

变量含义说明：

- \$\{pypto_version}：表示PyPTO包版本号，例如`9.1.0`。
- \$\{os_arch}：表示操作系统和CPU架构信息，例如`linux-aarch64`。
- `--full`：表示执行完整安装。
- `-q`：表示静默安装。
- `--pylocal`：安装软件包时，是否将python相关信息安装到CANN软件包的安装路径。

## 安装验证

完成以上步骤后，参考[样例运行](../invocation/examples_invocation.md)执行相关用例，验证PyPTO是否成功安装。
