/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tensor.h
 * \brief
 */

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <initializer_list>
#include "tilefwk/data_type.h"
#include "tilefwk/symbolic_scalar.h"

namespace npu::tile_fwk {

class LogicalTensor;
class SymbolicScalar;
using BinDataPtr = uint8_t*;
using Shape = std::vector<int64_t>;
using Offset = std::vector<int64_t>;
using Stride = std::vector<int64_t>;

class Tensor {
public:
    /**
     * \brief Constructs a new default Tensor object
     */
    Tensor();

    /**
     * \brief Destroy the Tensor object
     */
    ~Tensor();

    /**
     * \brief Constructs a new Tensor object with one input parameter
     *
     * \param s : a shared pointer to a LogicalTensor object
     */
    Tensor(std::shared_ptr<LogicalTensor> s);

    /**
     * \brief Construct a new Tensor object with 5 input parameters
     *
     * \param dataType : Data type of the tensor.
     * \param shape : A vector that stores the shape of the tensor.
     * \param name : Name of the tensor. The default value is "".
     * \param format : Format of the tensor. The default value is TileOpFormat::TILEOP_ND.
     * \attention : The parameters dataType and shape are required parameters.
     */
    Tensor(DataType dataType, const Shape& shape, std::string name = "", TileOpFormat format = TileOpFormat::TILEOP_ND);

    /**
     * \brief Construct a new Tensor object
     *
     * \param dataType : Datatype
     * \param shape : Shape of the tensor
     * \param name : Name of the tensor.
     * \param format : Format of the tensor. The default value is TileOpFormat::TILEOP_ND.
     */
    Tensor(DataType dataType, std::vector<SymbolicScalar> shape, std::string name = "",
           TileOpFormat format = TileOpFormat::TILEOP_ND);

    /**
     * \brief Construct a new Tensor object
     *
     * \param t : Datatype
     * \param shape : Shape of the tensor
     * \param name : Name of the tensor.
     * \param format : Format of the tensor. The default value is TileOpFormat::TILEOP_ND.
     * \code {.cpp}
     * Tensor t0(DT_FP32, {32, 32}) // shape with fixed type
     * Tensor t1(DT_FP32, {?, 32})  // first axis use dynamic shape
     * Tensor t2(DT_FP32, {GetInputShapeDim(t1, 0), 32}) // shape same as t1
     * \endcode
     */
    Tensor(DataType t, std::initializer_list<SymbolicScalar> shape, std::string name = "",
           TileOpFormat format = TileOpFormat::TILEOP_ND)
        : Tensor(t, std::vector<SymbolicScalar>(shape), name, format)
    {}

    /**
     * \brief Overload the assignment operator to assign the value of another Tensor object to the current Tensor
     * object.
     *
     * \param rhs : A constant reference to another Tensor object.
     * \return Tensor& : A reference to the current Tensor object.
     */
    Tensor& operator=(const Tensor& rhs);

    /**
     * \brief Move assignment operator.
     *
     * \param rhs : Rvalue reference to another Tensor object.
     * \return Tensor& : A reference to the current Tensor object.
     * \attention : The noexcept declaration indicates that the function will not throw exceptions.
     */
    Tensor& operator=(Tensor&& rhs) noexcept;

    /**
     * \brief Construct a new Tensor object by copying another Tensor object.
     *
     * \param rhs : A constant reference to another Tensor object.
     */
    Tensor(const Tensor& rhs);

    /**
     * \brief Construct a new Tensor object by moving another Tensor object.
     *
     * \param rhs : Rvalue reference to another Tensor object.
     */
    Tensor(Tensor&& rhs);

    /**
     * \brief Overload the * operator to access the LogicalTensor object.
     *
     * \return const LogicalTensor& : A reference to the LogicalTensor object.
     * \attention : The const keyword indicates that the function does not modify the object.
     */
    const LogicalTensor& operator*() const;

    /**
     * \brief Overload the * operator to access the LogicalTensor object.
     *
     * \return LogicalTensor& : A reference to the LogicalTensor object.
     */
    LogicalTensor& operator*();

    /**
     * \brief Get the const Storage object.
     *
     * \param readSlot : This parameter indicates whether slot reading is required. The default value is true.
     * \return const std::shared_ptr<LogicalTensor>& : A constant reference to the storage_ object.
     * \attention : The const keyword indicates that the function does not modify the object.
     */
    const std::shared_ptr<LogicalTensor>& GetStorage(bool readSlot = true) const;

    /**
     * \brief Get the Storage object.
     *
     * \param readSlot : This parameter indicates whether slot reading is required. The default value is true.
     * \return std::shared_ptr<LogicalTensor>& : A reference to the storage_ object.
     */
    std::shared_ptr<LogicalTensor>& GetStorage(bool readSlot = true);

    /**
     * \brief Set tensor cache policy.
     *
     * \param policy : PREFETCH mark this tensor will prefetch to cache before calculate.(Max num is 4.)
     *               NO_CACHEABLE mark this tensor will not get into cache.
     *        value : true mean enable, default is false.
     * \attention : NONE_CACHEABLE will only effect function input.
     *              NO_CACHEABLE will only effect function input and output.
     *              Two policy can not apply in one tensor.
     */
    void SetCachePolicy(CachePolicy policy, bool value);

    /**
     * \brief Get tensor cache policy.
     *
     * \param policy : CachePolicy enum.
     * \return bool : policy value.
     */
    bool GetCachePolicy(CachePolicy policy) const;

    /**
     * \brief Get the Data Type object
     *
     * \return DataType : The data type of the tensor.
     */
    DataType GetDataType() const;

    /**
     * \brief Get the shape of a tensor.
     *
     * \return const Shape & : A constant reference to the shape of the tensor.
     */
    const Shape& GetShape() const;

    /**
     * \brief Get the shape information of the specified axis of Tensor.
     *
     * \param axis : The axis of the shape to be obtained.
     * \return int : The shape of the specified axis.
     */
    int32_t GetShape(int axis) const;

    /**
     * \brief Get the valid shape of the tensor.
     *
     * \return std::vector<SymbolicScalar> : The valid shape of the tensor.
     */
    std::vector<SymbolicScalar>& GetValidShape() const;

    /**
     * \brief Get the format of the tensor.
     *
     * \return TileOpFormat : The format of the tensor.
     */
    TileOpFormat Format() const;

    /**
     * \brief Get the Id information of the Tensor.
     *
     * \return int : The Id information of the Tensor.
     */
    int Id() const { return index_; }

    /**
     * \brief Set the name of Tensor.
     *
     * \param name : The name of the tensor.
     */
    void SetName(const std::string& name) const;

    /**
     * \brief Get the name of Tensor.
     *
     * \return const std::string& : The name of the tensor.
     */
    std::string GetName() const;

    /**
     * \brief Get the Shape Dim Size of Tensor.
     */
    uint64_t Dim() const;

    /**
     * \brief Check if the tensor is empty.
     *
     * \return true : If the tensor is empty.
     * \return false : Otherwise.
     */
    bool IsEmpty() const;

private:
    void AssignStorage(const Tensor& rhs, std::shared_ptr<LogicalTensor> newStorage, bool addRef);

    std::shared_ptr<LogicalTensor> storage_;
    int index_{-1};
};

/**
 * @brief Get the size of a special dimension of input tensor
 *
 * @param t input tensor
 * @param n dimension index
 * @return SymbolicScalar : size of a special dimension of tensor
 */
SymbolicScalar GetInputShape(const Tensor& t, int n);

const std::vector<SymbolicScalar>& GetInputShape(const Tensor& tensor);

/**
 * @brief Get the Input Data of a  tensor
 *
 * @param t input tensor
 * @param offset positional shift applied to the each axis of the tensor
 * @return SymbolicScalar
 */
SymbolicScalar GetTensorData(const Tensor& t, const std::vector<SymbolicScalar>& offset);

/**
 * @brief Determines if the current iteration is the start of loop
 *
 * @param symbol current loop index
 * @param begin begin loop index
 * @return SymbolicScalar : expression to determine if currently at loop start
 */
SymbolicScalar IsLoopBegin(const SymbolicScalar& symbol, const SymbolicScalar& begin);

/**
 * @brief Determines if the current iteration is the end of loop
 *
 * @param symbol current loop index
 * @param end end loop index
 * @return SymbolicScalar : expression to determine if currently at loop start
 */
SymbolicScalar IsLoopEnd(const SymbolicScalar& symbol, const SymbolicScalar& end);

void SetTensorData(const SymbolicScalar& v, const std::vector<SymbolicScalar>& off, Tensor& dst);

void TensorMove(Tensor& src, Tensor& dst);
} // namespace npu::tile_fwk
