# 简介

本文档为PyPTO框架错误码说明及定位指导。

- F0XXXX为外部写法问题，请直接参考打屏报错提示。若报错提示不明确，请访问社区提交 [Issue](https://gitcode.com/cann/pypto/issues)。
- 非F0XXXX为框架内部问题，请根据错误码范围查阅对应组件的定位指导。若没有对应错误码或仍未解决，请访问社区提交 [Issue](https://gitcode.com/cann/pypto/issues)。
- 定位前建议配置以下常用日志环境变量，便于快速获取关键日志。详细介绍请参考《[环境变量参考](https://gitcode.com/cann/docs/blob/master/docs/zh/env-vars/README.md)》。

    ```bash
    #示例
    export ASCEND_GLOBAL_LOG_LEVEL=0
    export ASCEND_HOST_LOG_FILE_NUM=1000
    export ASCEND_WORK_PATH=./wk
    ```
