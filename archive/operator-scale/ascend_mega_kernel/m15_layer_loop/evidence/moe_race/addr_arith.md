# M105 · 地址自算（不引用别人的算式；由资源表常量直接算出）

## 命令
```
$ g++ -std=c++17 -I <wt-105>/m15_layer_loop /tmp/m105/consts.cpp -o /tmp/m105/consts && /tmp/m105/consts
```
`addr_arith.cpp` 随本目录入库（它 #include 的是库内 `m15_moe_resources.h`，只有"
echo "`UB_IG_SCAL/UB_IG_OFF/UB_IG_CUR/IG_DIAG_GM_SLOT` 四个派生量在文件里按生成物的同一算式写出，"
echo "因为那四个定义在生成物 `m15_moe_layer.h` 里）。

## 输出
```
HIDDEN=2560 INTER=640 NUM_EXPERTS=4 TOPK_MAX=4 M_MAX=64 TOTAL_MAX=256
UB_RT_XB=144384 UB_IG_SRC=144384 UB_IG_EXP=145408 UB_IG_CNT=146432 UB_IG_INV=146688 UB_IG_WTK=147712 UB_IG_END=151808
UB_IG_SCAL=151808 UB_IG_OFF=151840 UB_IG_CUR=151872 UB_IG_SCAL_END=151904
diag_ub_byte = UB_IG_CNT+32*4 = 146560 ; minus UB_RT_XB = 2176
xrow0_span = [144384, 149504)  -> diag inside x row0 = 1
ROWCTX: RT_RB=8 RT_RB*HIDDEN*2=40960
SZ_OFFSETS=64 WS_OFFSETS=988192 IG_DIAG_GM_SLOT=8 IG_DIAG_GM_SLOT*4=32 WS_OFFSETS+SZ_OFFSETS/2=988224
WS_XNORM=0 WS_BYTES=7685472 WS_MOE=6374752 SZ_XNORM=327680
SZ_COUNTS=32 SZ_OFFSETS tail check: AlignUp((E+1)*4+4,32)=32
```

## 由输出读出的三条（第 2 项任务的①②）
```
① UB 诊断槽字地址 = UB_IG_CNT + 32*4 = 146432 + 128 = 146560
   而 UB_RT_XB = 144384  ⇒ 146560 - 144384 = **2176**（= UB_IG_CNT+128 落在 router x 窗内的字节偏移）
   x 行 0 的 UB 跨度 = [UB_RT_XB, UB_RT_XB + HIDDEN*2) = [144384, 149504)
   ⇒ 146560 ∈ [144384, 149504) —— **诊断槽那 4 B 就在 router x 行 0 缓冲里面**
② GM 诊断槽地址 = WS_OFFSETS + IG_DIAG_GM_SLOT*4 = 988192 + 8*4 = **988224**
   （IG_DIAG_GM_SLOT = AlignUp((E+1)*4+4, 32)/4 = AlignUp(24,32)/4 = 8）
③ WS_BYTES = 7685472 —— 与落盘 `L*_moe_ws.bin` 的字节数逐一相符（见 runs 证据）
```
