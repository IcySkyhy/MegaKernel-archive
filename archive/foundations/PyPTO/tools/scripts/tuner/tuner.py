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

import argparse
import csv
import datetime
from itertools import product
import json
import os
import re
import shutil
import subprocess

from scipy.stats import gmean


class Patcher:
    '''
    Modify files according to run_params
    '''

    def __init__(self, run_params):
        self.run_params = run_params
        self.backups = []

    def __enter__(self):
        for item in self.run_params:
            Patcher._make_backup(item["file"])
            self.backups.append(item["file"])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for filepath in self.backups:
            Patcher._restore_backup(filepath)
        return False

    @staticmethod
    def _replace_line(file_path, line_num, new_content):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            prev_line = lines[line_num - 1].rstrip()

            leading_spaces = re.match(r'^\s*', prev_line).group(0)
            if prev_line.endswith(":"):
                leading_spaces += "\t"

            new_content = leading_spaces + new_content
            lines.insert(line_num, new_content + "\n")

        with open(file_path, "w") as file:
            file.writelines(lines)

    @staticmethod
    def _make_backup(path):
        shutil.copy(path, path + ".backup")

    @staticmethod
    def _restore_backup(path):
        backup = path + ".backup"
        if os.path.exists(backup):
            os.replace(backup, path)

    def apply_changes(self):
        for item in self.run_params:
            sorted_by_lines = dict(sorted(item["lines"].items(), reverse=True))
            for line_num, text in sorted_by_lines.items():
                Patcher._replace_line(item["file"], line_num, text)


class Execution:
    def __init__(self, result_folder):
        self.result_folder = result_folder

    @staticmethod
    def _save_run_params(name, run_params, file_name):
        run_params = run_params.copy()
        with open(file_name, 'w') as run_file:
            run_params.append({"name": name})
            json_txt = json.dumps(run_params, indent=4)
            run_file.write(json_txt)

    @staticmethod
    def _get_execution_time():
        dirs = list(filter(os.path.isdir, [os.path.join("output", f) for f in os.listdir("output")]))
        newest_folder = max(dirs, key=os.path.getmtime)
        bubble_log_path = os.path.join(newest_folder, 'bubble_analysis.log')

        cores = []
        with open(bubble_log_path, 'r') as f:
            content = f.read()

        core_pattern = (
            r'\[(AIC_\d+|AIV_\d+)\] Execute task num:(\d+)\s+Core Total Work Time: ([\d.]+)\s+'
            r'Total Wait Time: ([\d.]+)\s+Wait Schedule Time: ([\d.]+)\s+'
            r'Wait Predecessor Time: ([\d.]+)'
        )

        matches = re.findall(core_pattern, content)

        for match in matches:
            total_work_time = float(match[2])
            cores.append(total_work_time)

        max_work_time = max(total_work_time for total_work_time in cores) if total_work_time else 0
        return max_work_time

    def measure_perf(self, json_config, run_params):
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)

        execution_times = []
        for _ in range(json_config["repeats"]):
            with Patcher(run_params) as patcher:
                patcher.apply_changes()
                Execution._save_run_params(
                    self.result_folder, run_params, self.result_folder + "/combination_params.json"
                )
                test_result = self._run_test(json_config)

            if test_result == 0:
                time = Execution._get_execution_time()
                execution_times.append(time)
            else:
                return None

        execution_times.sort()

        mid = len(execution_times) // 2
        if len(execution_times) % 2 == 1:
            return execution_times[mid]
        else:
            return (execution_times[mid] + execution_times[mid - 1]) / 2

    def _run_test(self, json_config):
        test = json_config["test_name"]
        run_command = f"python3 {test}"
        print(run_command)
        log_path = f"{self.result_folder}/tiling.log"
        print(f"Build logs: {log_path}")

        env = dict(os.environ)
        with open(log_path, "w") as f:
            test = subprocess.run(run_command.split(), stdout=f, stderr=subprocess.STDOUT, env=env)
        return test.returncode


class HeuristicTile:
    def __init__(self, l1_size=131072, l0a_size=65536, l0b_size=65536, l0c_size=524288, output_dt_bytes=4):
        self.l1_size = l1_size
        self.l0a_size = l0a_size
        self.l0b_size = l0b_size
        self.l0c_size = l0c_size
        self.output_dt_bytes = output_dt_bytes

    @staticmethod
    def greatest_bit(x):
        n = 0
        power = 1
        while power <= x:
            power *= 2
            n += 1
        return n - 1

    def is_good_tiling(self, tile, input_dt_bytes):
        m, m_big, k, k_big, n, n_big = tile
        r1 = k_big % k == 0 and n_big >= n
        r2 = m * n * self.output_dt_bytes <= self.l0c_size
        r3 = n * k * input_dt_bytes <= self.l0b_size
        r4 = m * k * input_dt_bytes <= self.l0a_size
        return r1 and r2 and r3 and r4

    def generate_tiles(self, shape, input_dt_bytes):
        shape_m, shape_k, shape_n = shape
        num_of_variants = 0

        min_m = min(3, max(HeuristicTile.greatest_bit(shape_m), 4))
        min_k = min(3, max(HeuristicTile.greatest_bit(shape_k), 4))
        min_n = min(3, max(HeuristicTile.greatest_bit(shape_n), 4))

        values_m = [2**i for i in range(min_m, self.greatest_bit(shape_m) + 1)]
        values_k = [2**i for i in range(min_k, self.greatest_bit(shape_k) + 1)]
        values_n = [2**i for i in range(min_n, self.greatest_bit(shape_n) + 1)]

        for m, k, n in product(values_m, values_k, values_n):
            m_big = m
            k_big = k
            n_big = n

            if self.is_good_tiling([m, m_big, k, k_big, n, n_big], input_dt_bytes):
                num_of_variants += 1
                yield [m, m_big, k, k_big, n, n_big]

            if self.is_good_tiling([m, m_big, k, 2 * k_big, n, n_big], input_dt_bytes):
                num_of_variants += 1
                yield [m, m_big, k, 2 * k_big, n, n_big]

            if self.is_good_tiling([m, m_big, k, k_big, n, 2 * n_big], input_dt_bytes):
                num_of_variants += 1
                yield [m, m_big, k, k_big, n, 2 * n_big]

    def get_score_for_tiling(self, tile, mx_shape, input_type_size):
        """
        Estimate score for tiling. The higher the score, the better it is.
        """
        m_dim = 0
        k_dim = 2
        n_dim = 4

        m = tile[m_dim]
        k = tile[k_dim]
        n = tile[n_dim]

        min_tile = 16

        balance_weight = 1

        whole_m_score = 2
        whole_k_score = 2
        whole_n_score = 2

        score = 0

        # If the tiling size = shape size -> the preferred option
        score = (score + whole_m_score) if tile[m_dim] == max(m, min_tile) else score
        score = (score + whole_k_score) if tile[k_dim] == max(k, min_tile) else score
        score = (score + whole_n_score) if tile[n_dim] == max(n, min_tile) else score

        # The more filled L0A, L0B, L0C is better
        utilization_l0a = (tile[m_dim] * tile[k_dim] * input_type_size) / self.l0a_size
        utilization_l0b = (tile[k_dim] * tile[n_dim] * input_type_size) / self.l0b_size
        utilization_l0c = (tile[m_dim] * tile[n_dim] * self.output_dt_bytes) / self.l0c_size
        score += min_tile * gmean([utilization_l0a, utilization_l0b, utilization_l0c])

        # The closer the ratio's is to 1, the better
        ratio_mk = (m / k) if (m > k) else (k / m)
        ratio_kn = (k / n) if (k > n) else (n / k)
        ratio_mn = (m / n) if (m > n) else (n / m)

        # Penalty for bad balance
        score -= balance_weight * gmean([ratio_mk, ratio_kn, ratio_mn])
        return score

    def get_best_k_tiles(self, tiles, shape, dt_bytes, best_k_tiles):
        tiles.sort(key=lambda tile: -self.get_score_for_tiling(tile, shape, dt_bytes))

        best_tiles = []
        d = set()
        i = 0

        while len(d) < best_k_tiles and i < len(tiles):
            shape = tiles[i]
            if (shape[0], shape[2], shape[4]) not in d:
                d.add((shape[0], shape[2], shape[4]))
                best_tiles.append(shape)
            i += 1

        return best_tiles

    def preproc_line_conf(self, line_conf):
        for key in line_conf.keys():
            if key != "string" and isinstance(line_conf[key], str):
                # param set like Matmul_int08_32_1536_783
                operation, datatype, m, k, n = line_conf[key].split("_")
                m = int(m)
                k = int(k)
                n = int(n)

                dt_bytes = int(re.search(r"\d+", datatype).group(0)) // 8
                tiles = list(self.generate_tiles([m, k, n], input_dt_bytes=dt_bytes))
                best_tiles = self.get_best_k_tiles(tiles, [m, k, n], dt_bytes, best_k_tiles=5)

                print(f"Autogenerated shape for {line_conf[key]}")
                for tile in best_tiles:
                    print(tile)
                print("--------------------------")

                line_conf[key] = best_tiles


def check_json(json_config):
    all_params = []

    for file_conf in json_config["files"]:
        _, lines_conf = next(iter(file_conf.items()))
        for line_conf in lines_conf:
            params = [k for k in line_conf if k not in ("string", "line")]
            all_params.extend(params)

    checked = set()
    duplicates = set()
    for item in all_params:
        if item in checked:
            duplicates.add(item)
        checked.add(item)

    if duplicates:
        raise NameError(f"Parameters {duplicates} are not unique in your json config")


def preproc_json(json_config):
    """
    Replaces parameters that are set using strings
    """
    ht = HeuristicTile()
    check_json(json_config)
    for file_conf in json_config["files"]:
        path_to_file, lines_conf = list(file_conf.items())[0]
        for line_conf in lines_conf:
            ht.preproc_line_conf(line_conf)


def run_values_to_run_params(run_values, json_config):
    vec_idx = 0
    run_params = []

    for file_conf in json_config["files"]:
        path_to_file, lines_conf = list(file_conf.items())[0]
        run_param = {"file": path_to_file}
        generated_lines_conf = dict()

        for line in lines_conf:
            if len(line) == 3:
                # if line with tune parameter
                run_value = run_values[vec_idx]
                vec_idx += 1
                line_no = line["line"]
                line_value = line["string"].format(**run_value)
            else:
                line_no = line["line"]
                line_value = line["string"]
            generated_lines_conf[line_no] = line_value

        run_param["lines"] = generated_lines_conf
        run_params.append(run_param)

    return run_params


class ExhaustiveGenerator:
    def __init__(self, json_config):
        self.json_config = json_config

    @staticmethod
    def _parse_file_conf(lines_conf):
        """
        Generates permutations for single line of code
        """
        lines_comb = []
        for line_conf in lines_conf:
            params = line_conf.copy()

            del params["line"]
            del params["string"]

            string_param_comb = []
            # if we have tunable params
            if params:
                for e in product(*params.values()):
                    param_comb = dict(zip(params.keys(), e))
                    result = line_conf["string"].format(**param_comb)
                    string_param_comb.append([(line_conf["line"], result), param_comb])
                lines_comb.append(string_param_comb)

        return list(product(*lines_comb))

    def generate_combinations(self):
        """
        Generates all possible permutations of tune values inside config.json
        """
        comb_inside_file = dict()
        for file_conf in self.json_config["files"]:
            path_to_file, lines_conf = list(file_conf.items())[0]
            comb_inside_file[path_to_file] = ExhaustiveGenerator._parse_file_conf(lines_conf)

        for e in product(*comb_inside_file.values()):
            single_param_values = []

            for cpp_file, line_params in zip(comb_inside_file.keys(), e):
                d = dict()
                d["file"] = cpp_file
                num_lines = dict()

                for line_param in line_params:
                    line, string = line_param[0]
                    names = line_param[1]
                    single_param_values.append(names)
                    num_lines[line] = string

                d["lines"] = num_lines

            single_run_params = run_values_to_run_params(single_param_values, self.json_config)
            yield single_run_params, single_param_values


def parse_json():
    parser = argparse.ArgumentParser(
        description='Iterate through the vector and cube tiles '
        'in the specified test and measure performance \n'
        'To specify which tiles and tile values to iterate through, use config.json \n'
        'To start the iteration: \n\n'
        'python tools/scripts/tiling_tool.py --json_path /path/to/config.json',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--json_path", help="path to config.json where described tiling configs", type=str, required=True
    )

    args = parser.parse_args()

    with open(args.json_path) as f:
        json_config = json.load(f)
        preproc_json(json_config)

    return json_config


def save_to_csv(path_to_file, res_params):
    with open(path_to_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=res_params[0].keys())
        writer.writeheader()
        for p in res_params:
            writer.writerow(p)


def sort_results(results):
    def compare_function(x):
        return 1e6 if x[1] == "Error" else x[1]  # set configs of error will be at the end

    results_sorted_by_time = dict(sorted(results.items(), key=compare_function))
    return results_sorted_by_time


def remove_worst_combination(result_folder_path, results):
    sorted_results = sort_results(results)
    worst_comb = list(sorted_results.items())[-1]
    worst_comb_name = worst_comb[0]
    del results[worst_comb_name]
    shutil.rmtree(result_folder_path + f"/{worst_comb_name}", ignore_errors=True)


def main():
    json_config = parse_json()
    results = dict()
    now = datetime.datetime.now(datetime.timezone.utc)
    today_folder_name = now.strftime("%d_%b_%H_%M_%S")

    os.makedirs(json_config["results_folder"], exist_ok=True)
    result_folder_path = json_config["results_folder"] + "/" + today_folder_name
    os.makedirs(result_folder_path, exist_ok=True)

    executor = Execution(result_folder_path)
    generator = ExhaustiveGenerator(json_config)

    idx = 0
    res_params = []

    for run_params, run_values in generator.generate_combinations():
        executor.result_folder = f"{result_folder_path}/combination_{idx}"
        time = executor.measure_perf(json_config, run_params)

        run_result = "Error" if time is None else time
        results[f"combination_{idx}"] = run_result
        run_values.append({"time(us)": run_result})

        param_val_flat = {"combination": f"combination_{idx}"}
        for d in run_values:
            param_val_flat.update(d)
        res_params.append(param_val_flat)

        if idx >= json_config["save_best_k"]:
            remove_worst_combination(result_folder_path, results)

        res_params.sort(key=lambda x: 1e6 if x["time(us)"] == "Error" else x["time(us)"])
        save_to_csv(result_folder_path + "/config_perf.csv", res_params)

        idx += 1


if __name__ == "__main__":
    main()
