# pypto.Element简介

在PyPTO框架中，使用Element类型存储标量常量，用于计算类操作中的常量表达。Python内置的int类型通常会映射为DT\_INT64类型，float类型则映射为DT\_FP32类型。在操作数类型同时存在Tensor和Scalar的场景中，int和float通常会被转换成对应Tensor的数据类型。
