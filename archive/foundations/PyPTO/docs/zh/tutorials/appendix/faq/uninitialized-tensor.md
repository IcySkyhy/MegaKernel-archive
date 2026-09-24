# 使用未初始化的Tensor

## 问题现象描述

使用`pypto.tensor`声明了一个Tensor，误以为它与`torch.empty`类似，会申请一块未初始化的内存。在再次写入前直接读取它（如使用`view`），会导致框架校验错误或精度问题。

## 问题原因

在PyPTO中，除了使用`pypto.full`、`pypto.zeros`等显式包含初始化行为的Tensor声明外，还可以使用`pypto.tensor`声明Tensor，但该接口不包含初始化行为。在PyPTO中，每个Tensor必须先写后读，即必须先有producer，然后才能有consumer，未初始化的Tensor不申请内存。框架中通常会校验直接使用no producer Tensor并报错，但有时因校验遗漏，可能会导致上板精度错误。

## 处理步骤

避免使用未经初始化的Tensor。
