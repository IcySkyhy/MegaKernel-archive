/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "core/logging.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "core/error.h"
#include "bindings/ir/bindings.h"
#include "tilefwk/error.h"

namespace py = pybind11;

namespace pypto {

namespace {

void LogDebug(const std::string& message) { pypto::Logger(LogLevel::DEBUG, __LINE__) << message; }
void LogInfo(const std::string& message) { pypto::Logger(LogLevel::INFO, __LINE__) << message; }
void LogWarn(const std::string& message) { pypto::Logger(LogLevel::WARN, __LINE__) << message; }
void LogError(const std::string& message) { pypto::Logger(LogLevel::ERROR, __LINE__) << message; }
void LogFatal(const std::string& message) { pypto::Logger(LogLevel::FATAL, __LINE__) << message; }
void LogEvent(const std::string& message) { pypto::Logger(LogLevel::EVENT, __LINE__) << message; }

} // namespace

void BindError(py::module_& m)
{
    static py::exception<ir::Error> exc_error(m, "Error", PyExc_Exception);
    static py::exception<ir::ValueError> exc_value_error(m, "ValueError", PyExc_ValueError);
    static py::exception<ir::TypeError> exc_type_error(m, "TypeError", PyExc_TypeError);
    static py::exception<ir::RuntimeError> exc_runtime_error(m, "RuntimeError", PyExc_RuntimeError);
    static py::exception<ir::NotImplementedError> exc_not_implemented_error(m, "NotImplementedError",
                                                                            PyExc_NotImplementedError);
    static py::exception<ir::IndexError> exc_index_error(m, "IndexError", PyExc_IndexError);
    static py::exception<ir::AssertionError> exc_assertion_error(m, "AssertionError", PyExc_AssertionError);
    static py::exception<ir::InternalError> exc_internal_error(m, "InternalError", PyExc_RuntimeError);

    PyObject* internal_error_type = exc_internal_error.ptr();
    PyObject_SetAttrString(internal_error_type, "__module__", PyUnicode_FromString("pypto"));

    // Merged translator: use GetFullMessage() from pypto_impl for richer error messages
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const ir::ValueError& e) {
            PyErr_SetString(PyExc_ValueError, e.GetFullMessage().c_str());
        } catch (const ir::TypeError& e) {
            PyErr_SetString(PyExc_TypeError, e.GetFullMessage().c_str());
        } catch (const ir::RuntimeError& e) {
            PyErr_SetString(PyExc_RuntimeError, e.GetFullMessage().c_str());
        } catch (const ir::NotImplementedError& e) {
            PyErr_SetString(PyExc_NotImplementedError, e.GetFullMessage().c_str());
        } catch (const ir::IndexError& e) {
            PyErr_SetString(PyExc_IndexError, e.GetFullMessage().c_str());
        } catch (const ir::AssertionError& e) {
            PyErr_SetString(PyExc_AssertionError, e.GetFullMessage().c_str());
        } catch (const ir::InternalError& e) {
            PyErr_SetString(exc_internal_error.ptr(), e.GetFullMessage().c_str());
        } catch (const ir::Error& e) {
            PyErr_SetString(PyExc_Exception, e.GetFullMessage().c_str());
        } catch (const npu::tile_fwk::Error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });
}

void BindLogging(py::module_& m)
{
    py::enum_<LogLevel>(m, "LogLevel", py::arithmetic(), py::module_local(), "Enumeration of available log levels")
        .value("DEBUG", LogLevel::DEBUG, "Detailed information for debugging")
        .value("INFO", LogLevel::INFO, "General informational messages")
        .value("WARN", LogLevel::WARN, "Warning messages for potentially harmful situations")
        .value("ERROR", LogLevel::ERROR, "Error messages for failures")
        .value("FATAL", LogLevel::FATAL, "Critical errors that may cause termination")
        .value("EVENT", LogLevel::EVENT, "Special events and milestones")
        .value("NONE", LogLevel::NONE, "Disable all logging")
        .export_values();

    m.def("set_log_level", &LoggerManager::SetLevel, py::arg("level"), "Set the global log level threshold.");
    m.def("get_log_level", &LoggerManager::GetLevel, "Get the current global log level threshold.");

    // Per-level logging functions (from pypto_core)
    m.def("log_debug", &LogDebug, py::arg("message"), "Log a message at the DEBUG level");
    m.def("log_info", &LogInfo, py::arg("message"), "Log a message at the INFO level");
    m.def("log_warn", &LogWarn, py::arg("message"), "Log a message at the WARN level");
    m.def("log_error", &LogError, py::arg("message"), "Log a message at the ERROR level");
    m.def("log_fatal", &LogFatal, py::arg("message"), "Log a message at the FATAL level");
    m.def("log_event", &LogEvent, py::arg("message"), "Log a message at the EVENT level");

    // Generic log function (from pypto_impl)
    m.def(
        "log", [](LogLevel level, const std::string& message) { pypto::Logger(level, __LINE__) << message; },
        py::arg("level"), py::arg("message"), "Log a message at the specified level");

    // Test helper (from pypto_impl)
    m.def(
        "raise_error",
        [](const std::string& error_type, const std::string& message) {
            if (error_type == "ValueError") {
                throw ir::ValueError(message);
            } else if (error_type == "TypeError") {
                throw ir::TypeError(message);
            } else if (error_type == "RuntimeError") {
                throw ir::RuntimeError(message);
            } else if (error_type == "NotImplementedError") {
                throw ir::NotImplementedError(message);
            } else if (error_type == "IndexError") {
                throw ir::IndexError(message);
            } else if (error_type == "Error") {
                throw ir::Error(message);
            } else if (error_type == "AssertionError") {
                throw ir::AssertionError(message);
            } else if (error_type == "InternalError") {
                throw ir::InternalError(message);
            } else {
                throw ir::Error("Unknown error type: " + error_type);
            }
        },
        py::arg("error_type"), py::arg("message"), "Raise a C++ error for testing error handling");
}

} // namespace pypto
