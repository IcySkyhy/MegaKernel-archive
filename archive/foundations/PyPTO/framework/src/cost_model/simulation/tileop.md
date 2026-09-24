# TileOp

## Vector TileOp

### Tadd

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1, unsigned src1Shape0, unsigned src1Shape1, unsigned oriShape0, unsigned oriShape1>
TILEOP void Tadd(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1)

#### Function Description


```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src0 = Tensor(dype=T, shape=(TShape0, TShape1))
src1 = Tensor(dtype=T, shape=(src1Shape0, src1Shape1))
for (int i = 0; i < TShape0; i++) {
    for (int j = 0; j < TShape1; j++) {
        dst[i][j] = src0[i][j] + src1[i][j]
    }
}
```


*note: src1 supports broadcasting*


#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src0 shape dim0
* TShape1: dst/src0 shape dim1
* src1Shape0: src1 shape dim0
* src1Shape1: src1 shape dim1
* oriShape0: ?
* oriShape1: ?

function parameters:
* dst: dst ub buffer
* src0: src0 ub buffer
* src1: src1 ub buffer


#### Constraints



#### Demonstration

TileOp::Tadd<float, 16, 512, 16, 512, 16, 512>((__ubuf__ float *)UBId13Addr, (__ubuf__ float *)UBId16Addr, (__ubuf__ float *)UBId15Addr);

### Tsub

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1, unsigned src1Shape0, unsigned src1Shape1, unsigned oriShape0, unsigned oriShape1>
TILEOP void Tsub(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src0 = Tensor(dype=T, shape=(TShape0, TShape1))
src1 = Tensor(dtype=T, shape=(src1Shape0, src1Shape1))
for (int i = 0; i < TShape0; i++) {
    for (int j = 0; j < TShape1; j++) {
        dst[i][j] = src0[i][j] - src1[i][j]
    }
}
```

*note: src1 supports broadcasting*

#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src0 shape dim0
* TShape1: dst/src0 shape dim1
* src1Shape0: src1 shape dim0
* src1Shape1: src1 shape dim1
* oriShape0: ?
* oriShape1: ?

function parameters:
* dst: dst ub buffer
* src0: src0 ub buffer
* src1: src1 ub buffer


#### Constraints



#### Demonstration

TileOp::Tsub<float, 32, 1, 32, 1, 32, 1>((__ubuf__ float *)UBId8Addr, (__ubuf__ float *)UBId1Addr, (__ubuf__ float *)UBId3Addr);

### Tmul

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1, unsigned src1Shape0, unsigned src1Shape1, unsigned oriShape0, unsigned oriShape1>
TILEOP void Tmul(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src0 = Tensor(dype=T, shape=(TShape0, TShape1))
src1 = Tensor(dtype=T, shape=(src1Shape0, src1Shape1))
for (int i = 0; i < TShape0; i++) {
    for (int j = 0; j < TShape1; j++) {
        dst[i][j] = src0[i][j] * src1[i][j]
    }
}
```

*note: src1 supports broadcasting*

#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src0 shape dim0
* TShape1: dst/src0 shape dim1
* src1Shape0: src1 shape dim0
* src1Shape1: src1 shape dim1
* oriShape0: ?
* oriShape1: ?

function parameters:
* dst: dst ub buffer
* src0: src0 ub buffer
* src1: src1 ub buffer


#### Constraints



#### Demonstration

TileOp::Tmul<float, 32, 128, 32, 8, 32, 8>((__ubuf__ float *)UBId17Addr, (__ubuf__ float *)UBId17Addr, (__ubuf__ float *)UBId8Addr);

### Tdiv

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1, unsigned src1Shape0, unsigned src1Shape1, unsigned oriShape0, unsigned oriShape1>
TILEOP void Tdiv(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src0 = Tensor(dype=T, shape=(TShape0, TShape1))
src1 = Tensor(dtype=T, shape=(src1Shape0, src1Shape1))
for (int i = 0; i < TShape0; i++) {
    for (int j = 0; j < TShape1; j++) {
        dst[i][j] = src0[i][j] / src1[i][j]
    }
}
```

*note: src1 supports broadcasting*

#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src0 shape dim0
* TShape1: dst/src0 shape dim1
* src1Shape0: src1 shape dim0
* src1Shape1: src1 shape dim1
* oriShape0: ?
* oriShape1: ?

function parameters:
* dst: dst ub buffer
* src0: src0 ub buffer
* src1: src1 ub buffer


#### Constraints



#### Demonstration

TileOp::Tdiv<float, 32, 128, 32, 8, 32, 8>((__ubuf__ float *)UBId17Addr, (__ubuf__ float *)UBId17Addr, (__ubuf__ float *)UBId8Addr);

### Texp

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1>
TILEOP void Texp(__ubuf__ T *dst, __ubuf__ T *src)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src = Tensor(dype=T, shape=(TShape0, TShape1))

for (int i = 0; i < TShape0; i++) {
    for (int j = 0; j < TShape1; j++) {
        dst[i][j] = exp(src[i][j])
    }
}
```


#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src shape dim0
* TShape1: dst/src shape dim1

function parameters:
* dst: dst ub buffer
* src: src ub buffer


#### Constraints



#### Demonstration

TileOp::Texp<float, 16, 512>((__ubuf__ float *)UBId13Addr, (__ubuf__ float *)UBId13Addr);

### TLn

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1>
TILEOP void TLn(__ubuf__ T *dst, __ubuf__ T *src)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src = Tensor(dype=T, shape=(TShape0, TShape1))

for (int i = 0; i < TShape0; i++) {
    for (int j = 0; j < TShape1; j++) {
        dst[i][j] = ln(src[i][j])
    }
}
```


#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src shape dim0
* TShape1: dst/src shape dim1

function parameters:
* dst: dst ub buffer
* src: src ub buffer


#### Constraints



#### Demonstration

TileOp::TLn<float, 16, 512>((__ubuf__ float *)UBId13Addr, (__ubuf__ float *)UBId13Addr);

### Tsqrt

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1>
TILEOP void Tsqrt(__ubuf__ T *dst, __ubuf__ T *src)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src = Tensor(dype=T, shape=(TShape0, TShape1))

for (int i = 0; i < TShape0; i++) {
    for (int j = 0; j < TShape1; j++) {
        dst[i][j] = sqrt(src[i][j])
    }
}
```


#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src shape dim0
* TShape1: dst/src shape dim1

function parameters:
* dst: dst ub buffer
* src: src ub buffer


#### Constraints



#### Demonstration

TileOp::Tsqrt<float, 64, 1>((__ubuf__ float *)UBId1Addr, (__ubuf__ float *)UBId1Addr);


### Trsqrt

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1>
TILEOP void Trsqrt(__ubuf__ T *dst, __ubuf__ T *src)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src = Tensor(dype=T, shape=(TShape0, TShape1))

for (int i = 0; i < TShape0; i++) {
    for (int j = 0; j < TShape1; j++) {
        dst[i][j] = rsqrt(src[i][j])
    }
}
```


#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src shape dim0
* TShape1: dst/src shape dim1

function parameters:
* dst: dst ub buffer
* src: src ub buffer


#### Constraints



#### Demonstration

TileOp::Trsqrt<float, 64, 1>((__ubuf__ float *)UBId1Addr, (__ubuf__ float *)UBId1Addr);


### Tceil

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1>
TILEOP void Tceil(__ubuf__ T *dst, __ubuf__ T *src)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src = Tensor(dype=T, shape=(TShape0, TShape1))

for (int i = 0; i < TShape0; i++) {
    for (int j = 0; j < TShape1; j++) {
        dst[i][j] = ceil(src[i][j])
    }
}
```


#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src shape dim0
* TShape1: dst/src shape dim1

function parameters:
* dst: dst ub buffer
* src: src ub buffer


#### Constraints



#### Demonstration

TileOp::Tceil<float, 64, 1>((__ubuf__ float *)UBId1Addr, (__ubuf__ float *)UBId1Addr);


### Tfloor

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1>
TILEOP void Tfloor(__ubuf__ T *dst, __ubuf__ T *src)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src = Tensor(dype=T, shape=(TShape0, TShape1))

for (int i = 0; i < TShape0; i++) {
    for (int j = 0; j < TShape1; j++) {
        dst[i][j] = floor(src[i][j])
    }
}
```


#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src shape dim0
* TShape1: dst/src shape dim1

function parameters:
* dst: dst ub buffer
* src: src ub buffer


#### Constraints



#### Demonstration

TileOp::Tfloor<float, 64, 1>((__ubuf__ float *)UBId1Addr, (__ubuf__ float *)UBId1Addr);


### Ttrunc

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1>
TILEOP void Ttrunc(__ubuf__ T *dst, __ubuf__ T *src)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src = Tensor(dype=T, shape=(TShape0, TShape1))

for (int i = 0; i < TShape0; i++) {
    for (int j = 0; j < TShape1; j++) {
        dst[i][j] = trunc(src[i][j])
    }
}
```


#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src shape dim0
* TShape1: dst/src shape dim1

function parameters:
* dst: dst ub buffer
* src: src ub buffer


#### Constraints



#### Demonstration

TileOp::Ttrunc<float, 64, 1>((__ubuf__ float *)UBId1Addr, (__ubuf__ float *)UBId1Addr);

### Tadds

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1>
TILEOP void Tadds(__ubuf__ T *dst, __ubuf__ T *src0, T src1)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src0 = Tensor(dype=T, shape=(TShape0, TShape1))

for (int i = 0; i < TShape0; i++) {
    for (int j = 0; j < TShape1; j++) {
        dst[i][j] = src0[i][j] + src1
    }
}
```


#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src0 shape dim0
* TShape1: dst/src0 shape dim1

function parameters:
* dst: dst ub buffer
* src0: src0 ub buffer
* src1: src1 scalar value


#### Constraints



#### Demonstration

TileOp::Tadds<float, 64, 1>((__ubuf__ float *)UBId1Addr, (__ubuf__ float *)UBId1Addr, (float)9.99999997e-07);

### Tsubs

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1>
TILEOP void Tsubs(__ubuf__ T *dst, __ubuf__ T *src0, T src1)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src0 = Tensor(dype=T, shape=(TShape0, TShape1))

for (int i = 0; i < TShape0; i++) {
    for (int j = 0; j < TShape1; j++) {
		dst[i][j] = src0[i][j] - src1
	}
}
```


#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src0 shape dim0
* TShape1: dst/src0 shape dim1

function parameters:
* dst: dst ub buffer
* src0: src0 ub buffer
* src1: src1 scalar value


#### Constraints



#### Demonstration

TileOp::Tsubs<float, 64, 1>((__ubuf__ float *)UBId1Addr, (__ubuf__ float *)UBId1Addr, (float)9.99999997e-07);

### Tmuls

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1>
TILEOP void Tmuls(__ubuf__ T *dst, __ubuf__ T *src0, T src1)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src0 = Tensor(dype=T, shape=(TShape0, TShape1))

for (int i = 0; i < TShape0; i++) {
	for (int j = 0; j < TShape1; j++) {
		dst[i][j] = src0[i][j] * src1
	}
}
```


#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src0 shape dim0
* TShape1: dst/src0 shape dim1

function parameters:
* dst: dst ub buffer
* src0: src0 ub buffer
* src1: src1 scalar value


#### Constraints



#### Demonstration

TileOp::Tmuls<float, 64, 1>((__ubuf__ float *)UBId1Addr, (__ubuf__ float *)UBId1Addr, (float)9.99999997e-07);

### Tdivs

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1>
TILEOP void Tdivs(__ubuf__ T *dst, __ubuf__ T *src0, T src1)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src0 = Tensor(dype=T, shape=(TShape0, TShape1))

for (int i = 0; i < TShape0; i++) {
	for (int j = 0; j < TShape1; j++) {
		dst[i][j] = src0[i][j] / src1
	}
}
```


#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src0 shape dim0
* TShape1: dst/src0 shape dim1

function parameters:
* dst: dst ub buffer
* src0: src0 ub buffer
* src1: src1 scalar value


#### Constraints



#### Demonstration

TileOp::Tdivs<float, 64, 1>((__ubuf__ float *)UBId1Addr, (__ubuf__ float *)UBId1Addr, (float)9.99999997e-07);

### Tmins

#### Syntax

template <typename T, unsigned TShape0, unsigned TShape1>
TILEOP void Tmins(__ubuf__ T *dst, __ubuf__ T *src0, T src1)

#### Function Description

```
dst = Tensor(dtype=T, shape=(TShape0, TShape1))
src0 = Tensor(dype=T, shape=(TShape0, TShape1))

for (int i = 0; i < TShape0; i++) {
	for (int j = 0; j < TShape1; j++) {
		dst[i][j] = min(src0[i][j], src1);
	}
}
```


#### Parameters

template parameters:
* T: dtype
* TShape0: dst/src0 shape dim0
* TShape1: dst/src0 shape dim1

function parameters:
* dst: dst ub buffer
* src0: src0 ub buffer
* src1: src1 scalar value


#### Constraints



#### Demonstration

TileOp::Tmins<float, 64, 1>((__ubuf__ float *)UBId1Addr, (__ubuf__ float *)UBId1Addr, (float)9.99999997e-07);

### Tgather

#### Syntax

template <typename T, typename T2, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned src0Shape0,
    unsigned src0Shape1, unsigned axis>
TILEOP void Tgather(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T2 *src1)

#### Function Description

```
dst = tensor(dtype=T, shape=(TShape0, TShape1, TShape2))
src0 = tensor(dtype=T, shape=(src0Shape0, src0Shape1))
src1 = tensor(dtype=T2, shape=(TShape0, TShape1))
for (int i = 0; i < TShape0; ++i) {
	for (int j = 0; j < TShape1; ++j) {
		for (int k = 0; k < TShape2; k++) {
		    dst[i][j][k] = src0[src1[i][j]*TShape2+k];
		}
	}
}
```

#### Parameters

template parameters:
* T: dst/src0 dtype
* T2: src1 dtype
* TShape0: dst/src1 shape dim0
* TShape1: dst/src1 shape dim1
* TShape2: dst shape dim2
* src0Shape0: src0 shape dim0
* src0Shape1: src0 shape dim1
* axis: reserved

function parameters:
* dst: dst ub buffer
* src0: src0 ub buffer
* src1: src1 ub buffer


#### Constraints



#### Demonstration

TileOp::Tgather<float, int, 64, 64, 64, 64, 64, 0>((__ubuf__ float *)UBId1Addr, (__ubuf__ float *)UBId2Addr, (__ubuf__ int *)UBId3Addr);
