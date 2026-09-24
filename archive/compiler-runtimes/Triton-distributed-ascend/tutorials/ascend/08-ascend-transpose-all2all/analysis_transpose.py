# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# This script parses msprof CSV output to analyze TD fusion kernel performance.

import csv
import re
import sys
import os
import glob

# ANSI 颜色和样式代码定义
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def visual_len(s):
    """
    计算字符串在终端中的视觉显示宽度（过滤 ANSI 颜色字符）。
    中文字符计为 2 个宽度，英文字符计为 1 个宽度。
    """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    plain_text = ansi_escape.sub('', s)
    return sum(2 if '\u4e00' <= c <= '\u9fff' else 1 for c in plain_text)


def pad_str(s, width, align='right'):
    """
    根据视觉显示宽度，对字符串进行完美填充对齐。
    """
    v_len = visual_len(s)
    pad_len = max(0, width - v_len)
    if align == 'right':
        return " " * pad_len + s
    else:
        return s + " " * pad_len


def calculate_median(data_list):
    """
    计算并返回列表的中位数
    """
    if not data_list:
        return 0.0
    sorted_data = sorted(data_list)
    n = len(sorted_data)
    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0


def isTargetName(op_name, op_type):
    """识别 Transpose-All2All golden 路径中的 torch 算子 + HCCL 通信算子。
    """
    return ("hcom_alltoall" in op_name or "hcom_alltoall" in op_type) or \
           ("ConcatD" in op_name or "ConcatD" in op_type) or \
           ("Slice" in op_name or "Slice" in op_type) or \
           ("OnesLike" in op_name or "OnesLike" in op_type)

def parse_single_msprof_raw_data(csv_path, test_cases):
    """
    解析单个卡对应的 CSV 文件，动态计算每个用例的执行次数并提取原始数据。
    """
    hcom_records = []
    td_records = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        header_idx = 0
        for idx, line in enumerate(lines):
            if "Op Name" in line or "OP Type" in line:
                header_idx = idx
                break
        
        f.seek(0)
        for _ in range(header_idx):
            f.readline()
            
        reader = csv.DictReader(f)
        durationFortorch = 0
        for row in reader:
            op_name = row.get("Op Name", "")
            op_type = row.get("OP Type", "")
            
            try:
                duration = float(row.get("Task Duration(us)", "0").strip())
                start_time = float(row.get("Task Start Time(us)", "0").strip())
            except (ValueError, TypeError):
                continue

            if isTargetName(op_name, op_type):
                durationFortorch += duration
                # Golden 路径终点：ReduceSum 是 scatter/mask/sum 的最后一步
                if ("hcom_alltoall_" in op_name or "hcom_alltoall_" in op_type):
                    hcom_records.append({"start_time": start_time, "duration": durationFortorch})
                    durationFortorch = 0

            elif "kernel_hccl_transpose_a2a" in op_name:
                td_records.append({"start_time": start_time, "duration": duration})

    hcom_records.sort(key=lambda x: x["start_time"])
    td_records.sort(key=lambda x: x["start_time"])

    num_cases = len(test_cases)
    if num_cases == 0:
        return []

    # 【核心改进】根据解析出的总记录数和用例数，动态推导每个用例的实际运行次数（不写死 10）
    chunk_size_hcom = len(hcom_records) // num_cases
    chunk_size_td = len(td_records) // num_cases

    # 兜底保护，防止因数据缺失导致除以 0 或分片异常
    if chunk_size_hcom == 0:
        chunk_size_hcom = 1
    if chunk_size_td == 0:
        chunk_size_td = 1

    hcom_by_case = []
    td_by_case = []

    for i in range(0, len(hcom_records), chunk_size_hcom):
        hcom_by_case.append(hcom_records[i:i+chunk_size_hcom])
    for i in range(0, len(td_records), chunk_size_td):
        td_by_case.append(td_records[i:i+chunk_size_td])

    card_raw_results = []
    for idx in range(num_cases):
        case_hcom = hcom_by_case[idx] if idx < len(hcom_by_case) else []
        case_td = td_by_case[idx] if idx < len(td_by_case) else []
        
        card_raw_results.append({
            "hcom_raw": [item["duration"] for item in case_hcom],
            "td_raw": [item["duration"] for item in case_td]
        })

    return card_raw_results


def remove_upper_outliers(data):
    """
    双重鲁棒去噪：
    1. 过滤掉绝对的异常大值（明显偏离中位数的数值）。
    2. 对剩余样本使用 IQR 四分位距法剔除偏大的高延迟噪点。
    """
    if len(data) <= 2:
        return data
    
    # 第一步：基于中位数的粗筛（剔除突变级别的超大异常值）
    sorted_data = sorted(data)
    median = sorted_data[len(sorted_data) // 2]
    # 如果正常耗时在几百 us，突变到几十万 us，直接过滤
    coarse_filtered = [x for x in data if x <= max(median * 5.0, 5000.0)]
    
    if not coarse_filtered:
        coarse_filtered = data

    # 第二步：精筛（IQR 剔除上限噪点）
    sorted_data2 = sorted(coarse_filtered)
    n = len(sorted_data2)
    if n <= 2:
        return sorted_data2
        
    q1 = sorted_data2[int(n * 0.25)]
    q3 = sorted_data2[int(n * 0.75)]
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    
    fine_filtered = [x for x in coarse_filtered if x <= upper_bound]
    return fine_filtered if fine_filtered else coarse_filtered


def analyze_multicard_performance(parent_dir):
    """
    自动查找目标目录下的多卡 CSV 文件，并将多卡数据混合进行去暖身、去噪和极值分析。
    """
    S_ranges = (
        list(range(2040, 2049)) +      # 2040 ~ 2048
        list(range(4088, 4097)) +      # 4088 ~ 4096
        list(range(8184, 8193))        # 8184 ~ 8192
    )
    topk_list = [0]                    # transpose 无 topk 维度，占位
    H_list = [1, 2, 3, 4, 5, 6, 7]     # Transpose 测 H=1..7

    test_cases = []
    for s_val in S_ranges:
        for h_val in H_list:
            test_cases.append({"S": s_val, "topk": 0, "H": h_val, "D": 128})

    search_pattern = os.path.join(parent_dir, "PROF_000001_*", "mindstudio_profiler_output", "op_summary_*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        search_pattern_alt = os.path.join(parent_dir, "**", "mindstudio_profiler_output", "op_summary_*.csv")
        csv_files = glob.glob(search_pattern_alt, recursive=True)

    if not csv_files:
        print(f"[ERROR] 未找到任何匹配的 op_summary CSV 性能文件。")
        sys.exit(1)

    csv_files.sort()

    num_cards = len(csv_files)
    print(f"[INFO] 成功检测到 {num_cards} 张卡的性能分析数据文件:")
    for idx, filepath in enumerate(csv_files):
        parts = filepath.split(os.sep)
        prof_folder_name = next((p for p in parts if p.startswith("PROF_000001_")), "UNKNOWN_DIR")
        print(f"  - Rank {idx}: {prof_folder_name} -> {os.path.basename(filepath)}")

    all_cards_raw = []
    for csv_file in csv_files:
        card_raw = parse_single_msprof_raw_data(csv_file, test_cases)
        if card_raw:
            all_cards_raw.append(card_raw)

    if not all_cards_raw:
        print("[ERROR] 所有 CSV 文件解析失败或数据为空。")
        sys.exit(1)

    results = []
    num_cases = len(test_cases)
    
    for idx in range(num_cases):
        case_info = test_cases[idx]
        
        mixed_hcom_warmup_removed = []
        mixed_td_warmup_removed = []
        
        for card_data in all_cards_raw:
            if idx < len(card_data):
                # 【动态过滤 Warmup】排除前 3 次执行，保留其余后续所有执行数据
                # 如果当前动态 chunk 大于 3，则取 [3:]，若不大于 3 则为空（保证不崩）
                mixed_hcom_warmup_removed.extend(card_data[idx]["hcom_raw"][3:])
                mixed_td_warmup_removed.extend(card_data[idx]["td_raw"][3:])

        if not mixed_hcom_warmup_removed or not mixed_td_warmup_removed:
            continue

        clean_hcom = remove_upper_outliers(mixed_hcom_warmup_removed)
        clean_td = remove_upper_outliers(mixed_td_warmup_removed)

        # 统计学计算
        hcom_min = min(clean_hcom)
        hcom_max = max(clean_hcom)
        hcom_med = calculate_median(clean_hcom)
        hcom_avg = sum(clean_hcom) / len(clean_hcom)

        td_min = min(clean_td)
        td_max = max(clean_td)
        td_med = calculate_median(clean_td)
        td_avg = sum(clean_td) / len(clean_td)

        # 计算不同维度的加速比
        speedup_min = hcom_min / td_min if td_min > 0 else 0.0
        speedup_max = hcom_max / td_max if td_max > 0 else 0.0
        speedup_med = hcom_med / td_med if td_med > 0 else 0.0
        speedup_avg = hcom_avg / td_avg if td_avg > 0 else 0.0

        results.append({
            "S": case_info["S"],
            "topk": case_info.get("topk", 0),
            "H": case_info["H"],
            "D": case_info["D"],
            "hcom_min": hcom_min,
            "hcom_max": hcom_max,
            "hcom_med": hcom_med,
            "hcom_avg": hcom_avg,
            "td_min": td_min,
            "td_max": td_max,
            "td_med": td_med,
            "td_avg": td_avg,
            "speedup_min": speedup_min,
            "speedup_max": speedup_max,
            "speedup_med": speedup_med,
            "speedup_avg": speedup_avg,
            "num_cards": num_cards  # 传递卡数，供报告生成使用
        })

    return results


def print_report(results):
    """
    在控制台打印完美对齐的多卡性能汇总报表。
    支持自适应多卡展示与中位数对比。
    """
    if not results:
        print("[WARNING] 没有可供分析的数据，请检查 msprof 文件的完整性。")
        return

    # 从结果中提取真实的卡数
    num_cards = results[0]["num_cards"]

    # 1. 精确定义每一个数据维度的绝对子列宽度
    w_s = 6
    w_topk = 5
    w_h = 4
    w_d = 4

    # 核心统计项每个子项的绝对宽度分配（Min, Max, Med, Avg）
    w_hcom_sub = 11   # 每个 Hcom 数值宽度
    w_td_sub = 11     # 每个 TD 数值宽度
    w_sp_sub = 10     # 每个加速比数值宽度

    # 数值子项之间的间隔大小
    gap = "  "  # 双空格间隔

    # 2. 计算合并表头大列所需要对应覆盖的精准宽度
    # (子列宽 * 4) + (间隔宽 * 3)
    w_hcom_total = (w_hcom_sub * 4) + (len(gap) * 3)
    w_td_total = (w_td_sub * 4) + (len(gap) * 3)
    w_sp_total = (w_sp_sub * 4) + (len(gap) * 3)

    # 3. 完美构造原样合并大标题表头（保持靠右对齐）
    header_cols = [
        pad_str("S", w_s, 'right'),
        pad_str("topk", w_topk, 'right'),
        pad_str("H", w_h, 'right'),
        pad_str("D", w_d, 'right'),
        pad_str("Hcom (Min / Max / Med / Avg) (us)", w_hcom_total, 'right'),
        pad_str("TD融合算子 (Min / Max / Med / Avg) (us)", w_td_total, 'right'),
        pad_str("Speedup (Min / Max / Med / Avg)", w_sp_total, 'right')
    ]
    header_line = " | ".join(header_cols)
    total_line_width = visual_len(header_line)

    print("\n" + "=" * total_line_width)
    title_text = f"昇腾 AICore Transpose-All2All Triton vs HCCL Golden 对比分析报告 ({num_cards}卡联合)"
    print(pad_str(f"{BOLD}{GREEN}{title_text}{RESET}", total_line_width, 'left'))
    print("=" * total_line_width)
    print(header_line)
    print("-" * total_line_width)
    
    better_avg_count = 0
    better_med_count = 0
    total_speedup_avg = 0.0
    total_speedup_med = 0.0
    
    for item in results:
        s, topk, h, d = item["S"], item.get("topk", 0), item["H"], item["D"]
        sp_min, sp_max, sp_med, sp_avg = item["speedup_min"], item["speedup_max"], item["speedup_med"], item["speedup_avg"]
        total_speedup_avg += sp_avg
        total_speedup_med += sp_med
        
        # 1. 严格按独立的子单元格宽度格式化 Hcom 耗时（Min / Max / Med / Avg）
        hcom_min_cell = pad_str(f"{item['hcom_min']:.1f}", w_hcom_sub, 'right')
        hcom_max_cell = pad_str(f"{item['hcom_max']:.1f}", w_hcom_sub, 'right')
        hcom_med_cell = pad_str(f"{item['hcom_med']:.1f}", w_hcom_sub, 'right')
        hcom_avg_cell = pad_str(f"{item['hcom_avg']:.1f}", w_hcom_sub, 'right')
        hcom_cell = f"{hcom_min_cell}{gap}{hcom_max_cell}{gap}{hcom_med_cell}{gap}{hcom_avg_cell}"

        # 2. 严格格式化 TD 融合算子耗时
        td_min_cell = pad_str(f"{item['td_min']:.1f}", w_td_sub, 'right')
        td_max_cell = pad_str(f"{item['td_max']:.1f}", w_td_sub, 'right')
        td_med_cell = pad_str(f"{item['td_med']:.1f}", w_td_sub, 'right')
        td_avg_cell = pad_str(f"{item['td_avg']:.1f}", w_td_sub, 'right')
        td_cell = f"{td_min_cell}{gap}{td_max_cell}{gap}{td_med_cell}{gap}{td_avg_cell}"
        
        # 3. 严格格式化 Speedup 及其 ANSI 颜色控制
        color_min = GREEN if sp_min > 1.0 else RED
        color_max = GREEN if sp_max > 1.0 else RED
        color_med = GREEN if sp_med > 1.0 else RED
        color_avg = GREEN if sp_avg > 1.0 else RED

        sp_min_raw = f"{sp_min:.2f}x"
        sp_max_raw = f"{sp_max:.2f}x"
        sp_med_raw = f"{sp_med:.2f}x"
        sp_avg_raw = f"{sp_avg:.2f}x"

        # 生成无颜色的空白填充，确保精确控制视觉显示宽度
        pad_min = " " * max(0, w_sp_sub - len(sp_min_raw))
        pad_max = " " * max(0, w_sp_sub - len(sp_max_raw))
        pad_med = " " * max(0, w_sp_sub - len(sp_med_raw))
        pad_avg = " " * max(0, w_sp_sub - len(sp_avg_raw))

        sp_min_cell = f"{pad_min}{color_min}{sp_min_raw}{RESET}"
        sp_max_cell = f"{pad_max}{color_max}{sp_max_raw}{RESET}"
        sp_med_cell = f"{pad_med}{color_med}{sp_med_raw}{RESET}"
        sp_avg_cell = f"{pad_avg}{color_avg}{sp_avg_raw}{RESET}"
        
        speedup_cell = f"{sp_min_cell}{gap}{sp_max_cell}{gap}{sp_med_cell}{gap}{sp_avg_cell}"
        
        if sp_avg > 1.0:
            better_avg_count += 1
        if sp_med > 1.0:
            better_med_count += 1
            
        row_cols = [
            pad_str(str(s), w_s, 'right'),
            pad_str(str(topk), w_topk, 'right'),
            pad_str(str(h), w_h, 'right'),
            pad_str(str(d), w_d, 'right'),
            hcom_cell,
            td_cell,
            speedup_cell
        ]
        print(" | ".join(row_cols))
        
    print("=" * total_line_width)
    avg_sp_avg = total_speedup_avg / len(results) if results else 0.0
    avg_sp_med = total_speedup_med / len(results) if results else 0.0
    pass_rate_avg = (better_avg_count / len(results)) * 100.0 if results else 0.0
    pass_rate_med = (better_med_count / len(results)) * 100.0 if results else 0.0
    
    print(f"{BOLD}📊 {num_cards}卡混合极值去噪分析摘要 (Summary):{RESET}")
    print(f"  - 总测试用例（Shape 组合）: {len(results)} / 189")
    print(f"  - 混合去噪后，TD融合算子 平均值(Avg) 超越 Hcom 标杆的用例数: {better_avg_count}")
    print(f"  - 混合去噪后，TD融合算子 中位数(Med) 超越 Hcom 标杆的用例数: {better_med_count}")
    print(f"  - 稳定状态平均性能超越率 (Avg Pass Rate): {GREEN if pass_rate_avg >= 50 else RED}{pass_rate_avg:.2f}%{RESET}")
    print(f"  - 稳定状态中位数性能超越率 (Med Pass Rate): {GREEN if pass_rate_med >= 50 else RED}{pass_rate_med:.2f}%{RESET}")
    print(f"  - 稳定状态联合平均性能加速比 (Average Speedup of Avg): {GREEN if avg_sp_avg >= 1.0 else RED}{avg_sp_avg:.3f}x{RESET}")
    print(f"  - 稳定状态联合中位数性能加速比 (Average Speedup of Med): {GREEN if avg_sp_med >= 1.0 else RED}{avg_sp_med:.3f}x{RESET}")
    print("=" * total_line_width + "\n")


def write_to_csv(results, output_path="performance_report_summary.csv"):
    """
    将分析结果保存到本地 CSV 报表中。
    包含中位数相关的各列。
    """
    if not results:
        return
    num_cards = results[0]["num_cards"]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "S", "topk", "H", "D",
            "Hcom_Min_us", "Hcom_Max_us", "Hcom_Med_us", "Hcom_Avg_us",
            "TD_Min_us", "TD_Max_us", "TD_Med_us", "TD_Avg_us",
            "Speedup_Min", "Speedup_Max", "Speedup_Med", "Speedup_Avg"
        ])
        for item in results:
            writer.writerow([
                item["S"], item.get("topk", 0), item["H"], item["D"],
                f"{item['hcom_min']:.2f}", f"{item['hcom_max']:.2f}", f"{item['hcom_med']:.2f}", f"{item['hcom_avg']:.2f}",
                f"{item['td_min']:.2f}", f"{item['td_max']:.2f}", f"{item['td_med']:.2f}", f"{item['td_avg']:.2f}",
                f"{item['speedup_min']:.3f}", f"{item['speedup_max']:.3f}", f"{item['speedup_med']:.3f}", f"{item['speedup_avg']:.3f}"
            ])
    print(f"[INFO] 详细的{num_cards}卡混合去噪对比数据表已成功导出至：{output_path}")


if __name__ == "__main__":
    parent_directory = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print(f"[START] 正在扫描并联合解析 msprof 性能数据目录: {os.path.abspath(parent_directory)} ...")
    
    perf_results = analyze_multicard_performance(parent_directory)
    print_report(perf_results)
    write_to_csv(perf_results)