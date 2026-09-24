# 环境准备

- 进行PyPTO开发或运行之前，需要安装**驱动固件**和**CANN软件包**，请参考[《CANN快速安装》](https://www.hiascend.com/cann/download)完成环境准备。

    > [!NOTE] 说明
    > 安装CANN软件包后，使用CANN运行用户进行编译、运行时，需要以CANN运行用户登录环境，执行`source ${INSTALL_DIR}/set_env.sh`命令设置环境变量。`${INSTALL_DIR}`请替换为CANN软件安装后的文件存储路径。以root用户安装为例，安装后文件默认存储路径为：`/usr/local/Ascend/cann`。

- 安装PyPTO的Python依赖。

    - Python：版本 >= 3.9。
        - 需要安装Python的Development组件（常称为`python3-dev`）。

    - 安装Python依赖包。

        依赖的pip包及对应版本在`python/requirements.txt`中描述，可以使用如下命令完成安装：

        ```bash
        # 进入PyPTO项目源码根目录
        cd pypto

        # 安装相关pip包依赖
        python3 -m pip install -r python/requirements.txt
        ```

    - 安装PyTorch及TorchNPU。

        请务必先完成CANN toolkit包安装后，再安装`TorchNPU`。请根据实际环境的Python版本单独安装，详细指导请参考《[TorchNPU软件安装](https://www.hiascend.com/document/detail/zh/Pytorch/latest/installguide/swinstall/docs/zh/installation_guide/installation_description.md)》。需确保`PyTorch`、`TorchNPU`与`PyPTO`三者的Python版本一致。

- 安装CMake。PyPTO要求安装3.16.3及以上版本的CMake，如果版本不符合要求，可以参考如下示例安装满足要求的版本。

    示例：安装3.16.3版本的CMake（x86_64架构）。

    ```bash
    mkdir -p cmake-3.16 && wget -qO- "https://cmake.org/files/v3.16/cmake-3.16.3-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C cmake-3.16
    export PATH=`pwd`/cmake-3.16/bin:$PATH
    ```

- 安装其它依赖。

    - make
    - g++ >= 7.3.1
    - gcc >= 7.3.1
    - pybind11 >= 2.13.6（pip包，可通过`python3 -m pip install pybind11`安装）

- 安装PyPTO Toolkit。

具体使用文档参考[PyPTO Toolkit文档](https://pypto-tools.gitcode.com/index.html)。

> [!NOTE] 说明
> 对于PyPTO开发，并非必须安装驱动固件。在非昇腾设备上，可以利用CPU仿真环境先行进行PyPTO开发和测试，并在准备就绪后，利用昇腾设备进行验证和加速计算。仿真功能当前为试验特性，后续版本可能存在变更，暂不支持应用于生产环境。非昇腾设备的安装请参考《[CANN软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)》中“附录B：常用操作 > 在非昇腾设备上安装CANN”章节。
