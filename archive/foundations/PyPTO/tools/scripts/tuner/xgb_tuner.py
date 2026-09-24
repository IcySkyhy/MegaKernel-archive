#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

import csv
import datetime
import os
import random

import numpy as np
from scipy import optimize
import tuner
import xgboost as xgb

# performance time if test crashes
ERROR_PERFORMANCE = 1000000

random.seed(42)
np.random.seed(42)


class RandomParameterMutator:
    '''
    Takes the current combination of parameters ->
    selects one parameter ->
    get next value in candidates list or previous.
    '''

    def __init__(self, json_config):
        self.json_config = json_config
        self.tune_params = make_flat(json_config)

        # how many possible values have one parameter
        self.value_ranges = []
        for param_name in self.tune_params:
            candidates_size = len(self.tune_params[param_name])
            self.value_ranges.append(candidates_size)

        variables_count = len(self.tune_params)
        self.prev = [0] * variables_count
        self.id = 0

    def comb_vector_to_params(self, vector):
        run_values = []
        run_params = []

        param_info = flat_line_info(self.json_config)
        _generated_lines_conf = dict()

        vec_idx = 0

        for param_name, _ in param_info.items():
            candidate_idx = vector[vec_idx]
            param_value = self.tune_params[param_name][candidate_idx]
            run_value = {param_name: param_value}
            run_values.append(run_value)
            vec_idx += 1

        run_params = tuner.run_values_to_run_params(run_values, self.json_config)
        return run_params, run_values

    def generate_combinations(self):
        if self.id != 0:
            random_idx = random.randint(0, len(self.prev) - 1)
            if self.prev[random_idx] == 0:
                self.prev[random_idx] += min(1, self.value_ranges[random_idx] - 1)
            elif self.prev[random_idx] == self.value_ranges[random_idx] - 1:
                self.prev[random_idx] -= min(1, self.value_ranges[random_idx] - 1)
            else:
                self.prev[random_idx] += random.choice([-1, 1])

        self.id += 1
        run_params, run_values = self.comb_vector_to_params(self.prev)
        return run_params, run_values


def run_values_to_vec(run_values):
    flat_list = [item for d in run_values for item in list(d.values())[0]]
    return flat_list


def vec_to_run_values(vec, original_run_values):
    run_values = original_run_values.copy()
    pointer = 0

    vec = vec.copy()
    vec = list(map(int, vec))

    for run_value in run_values:
        for key in run_value.keys():
            part = vec[pointer:pointer + len(run_value[key])]
            run_value[key] = part
            pointer += len(run_value[key])

    return run_values


class MeasurePerf:
    def __init__(self, model, real_rate, original_run_values, result_folder_path, json_config):
        self.original_run_values = original_run_values
        self.json_config = json_config
        self.executor = tuner.Execution("single_run")
        self.model = model
        self.idx = 0
        self.real_rate = real_rate

        self.result_folder = result_folder_path

        self.train_x = []
        self.train_y = []

    def __call__(self, vector):
        if self.idx % self.real_rate == 0 or len(self.train_x) == 0:
            run_values = vec_to_run_values(vector, self.original_run_values)
            run_params = tuner.run_values_to_run_params(run_values, self.json_config)
            self.executor.result_folder = f"{self.result_folder}/combination_{self.idx}"
            time = self.executor.measure_perf(self.json_config, run_params)

            if time is None:
                self.idx += 1
                return ERROR_PERFORMANCE

            self.train_x.append(vector)
            self.train_y.append(time)
            self.model.fit(self.train_x, self.train_y)
        else:
            time = self.model.predict([vector])[0]
        self.idx += 1

        return time


class Storage:
    def __init__(self, save_folder, real_rate, param_mutator):
        self.iter = 0
        self.save_folder = save_folder
        self.real_rate = real_rate
        self.mutator = param_mutator

    def __call__(self, x, f, accept):
        comb_vector = self.mutator.prev
        run_params, run_values = self.mutator.comb_vector_to_params(comb_vector)

        result = {
            "combination": f"combination_{self.iter}",
            "time(us)": "Error" if f == ERROR_PERFORMANCE else float(f),
            "is_real": self.iter % self.real_rate == 0,
            "accepted": accept,
        }

        for value in run_values:
            result.update(value)

        with open(os.path.join(self.save_folder, "config_perf.csv"), "a", newline='') as res_csv:
            writer = csv.DictWriter(res_csv, fieldnames=list(result.keys()))
            if self.iter == 0:
                writer.writeheader()
            writer.writerow(result)

        self.iter += 1


def make_flat(json_config):
    """
    Flatten configuration values from a nested JSON‑style structure.

    This function return dictionary with format
    variable_name:candidates

    Example:
        {
            "my_tile1" : [tile_1, ... , tile_n],
            "my_flag" : [value_1, ... , value_n],
            ....
        }
    """
    candidates = dict()
    for file_conf in json_config["files"]:
        values = [
            candidates
            for filename, line_conf in file_conf.items()
            for line in line_conf
            for candidates in line.values()
            if isinstance(candidates, list)
        ]

        keys = [
            variable_name
            for filename, line_conf in file_conf.items()
            for line in line_conf
            for variable_name in line.keys()
            if not variable_name.startswith("string") and not variable_name.startswith("line")
        ]
        file_variables = dict(zip(keys, values))
        candidates.update(file_variables)
    return candidates


def flat_line_info(json_config):
    """
    Flatten configuration values from a nested JSON‑style structure.

    This function return dictionary with format
    variable_name: {apth_to_file, line, format_string}

    Example:
        {
            "my_tile1" : {file: "path_to_file", line: 12, string: "format_string"},
            "my_flag" : {file: "path_to_file", line: 113, string: "format_string"},
            ....
        }
    """
    line_info = dict()
    for file_conf in json_config["files"]:
        values = [
            {"file": filename, "line": line["line"], "string": line["string"]}
            for filename, line_conf in file_conf.items()
            for line in line_conf
        ]

        keys = [
            variable_name
            for filename, line_conf in file_conf.items()
            for line in line_conf
            for variable_name in line.keys()
            if not variable_name.startswith("string") and not variable_name.startswith("line")
        ]
        line_info_file = dict(zip(keys, values))
        line_info.update(line_info_file)

    return line_info


def main():
    json_config = tuner.parse_json()

    mutator = RandomParameterMutator(json_config)
    _executor = tuner.Execution("single_run")
    run_params, run_values = mutator.generate_combinations()
    original_run_values = run_values

    def dummy_minizer(f, x0, args, **kwargs):
        # we don't implement any local optimization
        dummy_result = optimize.OptimizeResult(
            x=x0,
            fun=f(x0),
            success=True,
            message="Dummy local optimization",
        )
        return dummy_result

    def take_step(vector):
        run_params, run_values = mutator.generate_combinations()
        return run_values_to_vec(run_values)

    initial_vector = run_values_to_vec(run_values)

    now = datetime.datetime.now(datetime.timezone.utc)
    today_folder_name = now.strftime("%d_%b_%H_%M_%S")
    os.makedirs(json_config["results_folder"], exist_ok=True)
    result_folder_path = json_config["results_folder"] + "/" + today_folder_name
    os.makedirs(result_folder_path, exist_ok=True)

    real_perf_limit = 100
    real_rate = 10

    model = xgb.XGBRegressor()

    mp = MeasurePerf(
        model=model,
        real_rate=real_rate,
        original_run_values=original_run_values,
        result_folder_path=result_folder_path,
        json_config=json_config,
    )

    storage = Storage(result_folder_path, real_rate, mutator)
    temperatures = [10, 5, 1, 0.25]

    best_x = None
    best_y = None

    for temp in temperatures:
        result = optimize.basinhopping(
            mp,
            initial_vector,
            take_step=take_step,
            niter=real_rate * real_perf_limit // len(temperatures),
            minimizer_kwargs={"method": dummy_minizer},
            callback=storage,
            T=temp,
        )
        if best_y is None or best_y > result.fun:
            best_y = result.fun
            best_x = result.x

    print(f"Best result: {best_y:.2f} us")
    print("Parameter values: ", vec_to_run_values(best_x, original_run_values))


if __name__ == '__main__':
    main()
