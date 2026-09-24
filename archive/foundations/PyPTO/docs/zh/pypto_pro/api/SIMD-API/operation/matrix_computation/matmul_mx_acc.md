# pypto_pro.language.matmul_mx_acc

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

在已有FP32累加器上继续累加一次MX矩阵乘积：

```text
dst = acc + dequant(lhs, scale_a) @ dequant(rhs, scale_b)
```

主要用于K维分块累加。首个K块使用[`matmul_mx`](matmul_mx.md)，其余K块使用本接口，且每个K块必须加载与该块对应的A/B E8M0 scale。

## 函数原型

```python
pypto_pro.language.matmul_mx_acc(
    dst_tile, acc_tile, lhs_tile, rhs_tile, scale_a, scale_b, *, phase=None
)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | Acc/L0C Tile，保存累加结果 |
| `acc_tile` | 输入 | Acc/L0C Tile，已有的FP32累加值，通常与`dst_tile`为同一Tile |
| `lhs_tile` | 输入 | Left/L0A Tile，当前K块的MXFP8或MXFP4左矩阵尾数 |
| `rhs_tile` | 输入 | Right/L0B Tile，当前K块的MXFP8或MXFP4右矩阵尾数 |
| `scale_a` | 输入 | ScaleLeft Tile，当前K块的A矩阵E8M0 scale |
| `scale_b` | 输入 | ScaleRight Tile，当前K块的B矩阵E8M0 scale |
| `phase` | 输入 | 可选，K维分块时使用`pl.AccPhase.Partial`或`pl.AccPhase.Final` |

## 参数范围

数据类型、shape、内存空间、scale布局及K对齐要求与[`matmul_mx`](matmul_mx.md)一致，此外：

- `dst_tile`与`acc_tile`必须为Acc/L0C中的FP32 Tile，shape均为`[M,N]`。
- 首个K块不能使用未初始化的`acc_tile`，应调用`matmul_mx`建立初始累加值。
- 每个K块的scale Tile都必须与当前数据Tile满足`addr(scale_a) = addr(lhs_tile) >> 4`和`addr(scale_b) = addr(rhs_tile) >> 4`。多缓冲轮转时，每一组数据/scale Tile都须分别满足该关系。
- 使用`phase`时，非末块传`pl.AccPhase.Partial`，末块传`pl.AccPhase.Final`；写回GM时配合`phase=pl.STPhase.Final`。

## 调用示例

下面展示K维循环的核心部分。各Tile的创建方式见[`matmul_mx`](matmul_mx.md)示例；其中数据Tile shape为`[TILE,TILE]`，scale Tile shape分别为`[TILE,TILE/32]`和`[TILE/32,TILE]`。

```python
with pl.section_cube():
    ac = acc.current()
    for ki in pl.range(0, K_SIZE, TILE):
        al1, bl1 = a_l1.next(), b_l1.next()
        sal1, sbl1 = sa_l1.next(), sb_l1.next()
        al0, bl0 = a_l0.next(), b_l0.next()
        sal0, sbl0 = sa_l0.next(), sb_l0.next()

        pl.load(al1, a, [0, ki])
        pl.load(bl1, b, [ki, 0])
        pl.load(sal1, scale_a, [0, ki // 64, 0], order=[0, 1])
        pl.load(sbl1, scale_b, [ki // 64, 0, 0], order=[0, 1])
        pl.move(al0, al1)
        pl.move(bl0, bl1)
        pl.move(sal0, sal1)
        pl.move(sbl0, sbl1)

        if ki == 0:
            pl.matmul_mx(ac, al0, bl0, sal0, sbl0, phase=pl.AccPhase.Partial)
        elif ki + TILE < K_SIZE:
            pl.matmul_mx_acc(
                ac, ac, al0, bl0, sal0, sbl0, phase=pl.AccPhase.Partial)
        else:
            pl.matmul_mx_acc(
                ac, ac, al0, bl0, sal0, sbl0, phase=pl.AccPhase.Final)

    pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
```

如果K只有一个块，应直接调用`matmul_mx(..., phase=pl.AccPhase.Final)`，不调用`matmul_mx_acc`。
