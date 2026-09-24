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
"""Data plotting"""

import logging

import matplotlib.pyplot as plt
import numpy as np


class DataDiffAnalyzer:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler("app.log", encoding="utf-8")],
        )

        self.is_calc_inc = True
        self.is_calc_rel = True
        self.is_out_no_fig = False
        self.is_sort = False
        self.conf_seg_off = None
        self.conf_seg_len = None
        self.conf_fig_format = 'png'

    @staticmethod
    def calc_canb_dist_elemwise(data_a, data_b):
        abs_sum = np.abs(data_a) + np.abs(data_b)
        sub_abs = np.abs(data_a - data_b)
        arg_zeros = np.argwhere(np.equal(abs_sum, 0))
        np.put(abs_sum, arg_zeros, 1)
        diff_rel = sub_abs / abs_sum
        np.put(diff_rel, arg_zeros, 0)
        return diff_rel

    @staticmethod
    def get_shape_str(shape):
        shape_str = '_'.join(str(i) for i in shape)
        if shape_str == '':
            shape_str = 's'
        return shape_str

    # function to generate data averages
    @staticmethod
    def gen_max_min_avg(data, mode='all'):
        if mode in ['pos', 'pos2', 'pos4']:
            arg_data = np.argwhere(np.greater(data, 0))
            d = np.take(data, arg_data).reshape(-1)
        elif mode in ['neg', 'neg2', 'neg4']:
            arg_data = np.argwhere(np.less(data, 0))
            d = np.take(data, arg_data).reshape(-1)
        elif mode in ['all', 'avg', 'avg2']:
            d = data
        else:
            raise RuntimeError(f'Error: bad mode {mode}')

        if len(d) == 0:
            d = [0]
        if mode == 'pos2':
            min_ = np.min(d)
            avg = np.mean(d)
            return min_, avg
        elif mode == 'neg2':
            max_ = np.max(d)
            avg = np.mean(d)
            return max_, avg
        elif mode == 'pos4':
            max_ = np.max(d)
            min_ = np.min(d)
            avg = np.mean(d)
            arg_data = np.argwhere(np.greater_equal(d, avg))
            dmaxa = np.take(d, arg_data).reshape(-1)
            if len(dmaxa) == 0:
                dmaxa = [0]
            avg_max = np.mean(dmaxa)
            arg_data = np.argwhere(np.less(d, avg))
            dmina = np.take(d, arg_data).reshape(-1)
            if len(dmina) == 0:
                dmina = [0]
            avg_min = np.mean(dmina)
            return max_, min_, avg, avg_max, avg_min
        elif mode == 'neg4':
            max_ = np.max(d)
            min_ = np.min(d)
            avg = np.mean(d)
            arg_data = np.argwhere(np.less_equal(d, avg))
            dmina = np.take(d, arg_data).reshape(-1)
            if len(dmina) == 0:
                dmina = [0]
            avg_min = np.mean(dmina)
            arg_data = np.argwhere(np.greater(d, avg))
            dmaxa = np.take(d, arg_data).reshape(-1)
            if len(dmaxa) == 0:
                dmaxa = [0]
            avg_max = np.mean(dmaxa)
            return max_, min_, avg, avg_max, avg_min
        elif mode == 'avg':
            avg = np.mean(d)
            return avg
        elif mode == 'avg2':
            avg = np.mean(d)
            arg_data = np.argwhere(np.not_equal(data, 0))
            dnz = np.take(data, arg_data).reshape(-1)
            if len(dnz) == 0:
                dnz = [0]
            avg_nz = np.mean(dnz)
            return avg, avg_nz
        else:
            max_ = np.max(d)
            min_ = np.min(d)
            avg = np.mean(d)
            return max_, min_, avg

    def fix_input_and_compute(self, a, b, f_type, sort):
        a_dtype = f_type[0]
        b_dtype = f_type[1]
        self.is_sort = sort
        f_out_fig = 'sort-diff.' + self.conf_fig_format if self.is_sort else 'diff.' + self.conf_fig_format
        f_a_info = 'A_' + str(a_dtype)
        f_b_info = 'B_' + str(b_dtype)

        st_a_total_raw = a.size
        st_b_total_raw = b.size
        st_a_shape_raw = self.get_shape_str(a.shape)
        st_b_shape_raw = self.get_shape_str(b.shape)
        if st_a_shape_raw != st_b_shape_raw or st_a_total_raw != st_b_total_raw:
            st_a_shape_raw = '__' + st_a_shape_raw + '__'
            st_b_shape_raw = '__' + st_b_shape_raw + '__'

        a = a.reshape(-1)
        b = b.reshape(-1)

        # fix input
        st_a_total_aligned = st_a_total_raw
        _st_b_total_aligned = st_b_total_raw
        if st_a_total_raw > st_b_total_raw:
            b = np.append(b, np.zeros(st_a_total_raw - st_b_total_raw, b_dtype))
            _st_b_total_aligned = st_a_total_raw
        else:
            a = np.append(a, np.zeros(st_b_total_raw - st_a_total_raw, a_dtype))
            st_a_total_aligned = st_b_total_raw

        # data random pick
        st_pick_data_total = st_a_total_aligned

        # data sort
        if self.is_sort:
            _arg_sort = np.argsort(a)
            a = np.take(a, _arg_sort)
            b = np.take(b, _arg_sort)

        # data seg
        seg_begin = 0
        seg_end = st_pick_data_total
        my_is_data_seg = False
        if self.conf_seg_off is not None and self.conf_seg_off > 0:
            if self.conf_seg_off < st_pick_data_total:
                seg_begin = self.conf_seg_off
            my_is_data_seg = True
        if self.conf_seg_len is not None and self.conf_seg_len > 0:
            seg_end = seg_begin + self.conf_seg_len
            if seg_end > st_pick_data_total:
                seg_end = st_pick_data_total
            my_is_data_seg = True
        a = a[seg_begin:seg_end]
        b = b[seg_begin:seg_end]
        st_a_seg_data_total = len(a)
        st_b_seg_data_total = len(b)
        st_seg_data_total = st_a_seg_data_total

        a = a.astype(np.float64)
        b = b.astype(np.float64)

        # calc stat
        st_a_inf = len(np.argwhere(np.isinf(a)))
        st_a_nan = len(np.argwhere(np.isnan(a)))
        st_a_zero = len(np.argwhere(np.equal(a, 0)))
        st_a_pos = len(np.argwhere(np.greater(a, 0)))
        st_a_neg = len(np.argwhere(np.less(a, 0)))
        st_b_inf = len(np.argwhere(np.isinf(b)))
        st_b_nan = len(np.argwhere(np.isnan(b)))
        st_b_zero = len(np.argwhere(np.equal(b, 0)))
        st_b_pos = len(np.argwhere(np.greater(b, 0)))
        st_b_neg = len(np.argwhere(np.less(b, 0)))

        st_zamb = st_a_zero - st_b_zero
        st_pamb = st_a_pos - st_b_pos
        st_namb = st_a_neg - st_b_neg
        st_zamb_info = ' zamb=' + str(st_zamb)
        st_zbma_info = st_zamb_info
        st_pamb_info = ' pamb=' + str(st_pamb)
        st_pbma_info = st_pamb_info
        st_namb_info = ' namb=' + str(st_namb)
        st_nbma_info = st_namb_info
        st_zapb = st_a_zero + st_b_zero
        st_zabd = (abs(st_zamb) / st_zapb) if st_zapb != 0 else st_zapb
        st_zabd = float(st_zabd)
        st_zambd_info = ' zabd=' + str(st_zabd)
        _st_zbmad_info = st_zambd_info
        st_zdp = abs(st_zamb) / st_seg_data_total
        st_zdp_info = ' zdp=' + str(st_zdp)
        st_pdp = abs(st_pamb) / st_seg_data_total
        st_pdp_info = ' pdp=' + str(st_pdp)
        st_ndp = abs(st_namb) / st_seg_data_total
        st_ndp_info = ' ndp=' + str(st_ndp)

        st_a_seg_info = f"{st_a_seg_data_total}@{seg_begin}/{seg_end}#" if my_is_data_seg else ""
        st_a_info = (
            f"t={st_a_seg_info} {st_a_total_raw} "
            f"s={st_a_shape_raw} "
            f"inf={st_a_inf} nan={st_a_nan} z={st_a_zero} "
            f"p={st_a_pos} n={st_a_neg}"
            f"{st_zamb_info}{st_pamb_info}{st_namb_info}"
            f"{st_zambd_info}{st_zdp_info}{st_pdp_info}{st_ndp_info}"
        )
        logging.info(f"info_a: {st_a_info}")

        st_b_seg_info = f"{st_b_seg_data_total}@{seg_begin}/{seg_end}#" if my_is_data_seg else ""
        st_b_info = (
            f"t={st_b_seg_info} {st_b_total_raw} "
            f"s={st_b_shape_raw} "
            f"inf={st_b_inf} nan={st_b_nan} z={st_b_zero} "
            f"p={st_b_pos} n={st_b_neg}"
            f"{st_zbma_info}{st_pbma_info}{st_nbma_info}"
            f"{st_zambd_info}{st_zdp_info}{st_pdp_info}{st_ndp_info}"
        )
        logging.info(f"info_b: {st_b_info}")

        # process input
        g_aa = a
        g_bb = b

        # remove inf/nan before calc
        arg_aa_inf_nan = np.argwhere(np.logical_or(np.isinf(g_aa), np.isnan(g_aa)))
        arg_bb_inf_nan = np.argwhere(np.logical_or(np.isinf(g_bb), np.isnan(g_bb)))
        np.put(g_aa, arg_aa_inf_nan, 0)
        np.put(g_bb, arg_bb_inf_nan, 0)

        # calc data statistic
        g_aa_avg_s, g_aa_avg_nz_s = self.gen_max_min_avg(g_aa, mode='avg2')
        g_aa_pos_max_s, g_aa_pos_min_s, g_aa_pos_avg_s, g_aa_pos_max_avg_s, g_aa_pos_min_avg_s = self.gen_max_min_avg(
            g_aa, mode='pos4'
        )
        g_aa_neg_max_s, g_aa_neg_min_s, g_aa_neg_avg_s, g_aa_neg_max_avg_s, g_aa_neg_min_avg_s = self.gen_max_min_avg(
            g_aa, mode='neg4'
        )
        g_data_a_info = (
            f"pmax={g_aa_pos_max_s:.6e} nmin={g_aa_neg_min_s:.6e} "
            f"avg={g_aa_avg_s:.6e} avgnz={g_aa_avg_nz_s:.6e} "
            f"pavg={g_aa_pos_avg_s:.6e} navg={g_aa_neg_avg_s:.6e} "
            f"pmin={g_aa_pos_min_s:.6e} nmax={g_aa_neg_max_s:.6e} "
            f"pmaxa={g_aa_pos_max_avg_s:.6e} pmina={g_aa_pos_min_avg_s:.6e} "
            f"nmaxa={g_aa_neg_max_avg_s:.6e} nmina={g_aa_neg_min_avg_s:.6e}"
        )
        logging.info('data_a: ' + g_data_a_info)

        g_bb_avg_s, g_bb_avg_nz_s = self.gen_max_min_avg(g_bb, mode='avg2')
        g_bb_pos_max_s, g_bb_pos_min_s, g_bb_pos_avg_s, g_bb_pos_max_avg_s, g_bb_pos_min_avg_s = self.gen_max_min_avg(
            g_bb, mode='pos4'
        )
        g_bb_neg_max_s, g_bb_neg_min_s, g_bb_neg_avg_s, g_bb_neg_max_avg_s, g_bb_neg_min_avg_s = self.gen_max_min_avg(
            g_bb, mode='neg4'
        )
        g_data_b_info = (
            f"pmax={g_bb_pos_max_s:.6e} nmin={g_bb_neg_min_s:.6e} "
            f"avg={g_bb_avg_s:.6e} avgnz={g_bb_avg_nz_s:.6e} "
            f"pavg={g_bb_pos_avg_s:.6e} navg={g_bb_neg_avg_s:.6e} "
            f"pmin={g_bb_pos_min_s:.6e} nmax={g_bb_neg_max_s:.6e} "
            f"pmaxa={g_bb_pos_max_avg_s:.6e} pmina={g_bb_pos_min_avg_s:.6e} "
            f"nmaxa={g_bb_neg_max_avg_s:.6e} nmina={g_bb_neg_min_avg_s:.6e}"
        )
        logging.info('data_b: ' + g_data_b_info)

        # calc incremental diff
        if self.is_calc_inc:
            g_diff_inc = g_aa - g_bb
            g_diff_inc_pos_max_s, g_diff_inc_pos_min_s, g_diff_inc_pos_avg_s = self.gen_max_min_avg(
                g_diff_inc, mode='pos'
            )
            g_diff_inc_neg_max_s, g_diff_inc_neg_min_s, g_diff_inc_neg_avg_s = self.gen_max_min_avg(
                g_diff_inc, mode='neg'
            )
            arg_non_zeros = np.argwhere(np.not_equal(g_diff_inc, 0.0))
            diff_inc_diff_num = len(arg_non_zeros)
            g_diff_info = (
                f"diff_num={diff_inc_diff_num} "
                f"pmax={g_diff_inc_pos_max_s:.6e} nmin={g_diff_inc_neg_min_s:.6e} "
                f"pavg={g_diff_inc_pos_avg_s:.6e} navg={g_diff_inc_neg_avg_s:.6e} "
                f"pmin={g_diff_inc_pos_min_s:.6e} nmax={g_diff_inc_neg_max_s:.6e}"
            )
            logging.info('diff_inc: ' + g_diff_info)

        # calc canberra dist
        if self.is_calc_rel:
            g_diff_rel = self.calc_canb_dist_elemwise(g_aa, g_bb)
            g_diff_rel_max_s, g_diff_rel_min_s, g_diff_rel_avg_s = self.gen_max_min_avg(g_diff_rel, mode='all')
            g_diff_rel_pos_min_s, g_diff_rel_pos_avg_s = self.gen_max_min_avg(g_diff_rel, mode='pos2')
            g_diff_rel_pos_max_s = g_diff_rel_max_s
            diff_rel_info = (
                f"avg={g_diff_rel_avg_s:.6e} max={g_diff_rel_max_s:.6e} min={g_diff_rel_min_s:.6e} "
                f"pavg={g_diff_rel_pos_avg_s:.6e} pmax={g_diff_rel_pos_max_s:.6e} pmin={g_diff_rel_pos_min_s:.6e}"
            )
            logging.info('diff_rel: ' + diff_rel_info)

        # output figure
        if not self.is_out_no_fig:
            # output figure
            fig_title = f_a_info + '\n' + f_b_info
            fig_avg_linewidth = 0.5
            _fig_thresh_linewidth = 0.05
            fig_alpha = 0.5
            fig_markersize_factor = 1
            fig_markersize_a = 1.6 * fig_markersize_factor
            fig_markersize_b = 1.5 * fig_markersize_factor
            _fig_markersize_mod = 1.5 * fig_markersize_factor
            fig_markersize = 1.5 * fig_markersize_factor
            fig_legend_fontsize = 'xx-small'
            fig_legend_alpha = 0.6
            fig_legend_label_color = 'lightgray'

            subg_mark = [1, self.is_calc_inc, self.is_calc_rel]
            subg_num = sum(subg_mark)
            subg_id_min = subg_num * 100 + 11
            subg_id_max = subg_id_min + subg_num
            subg_id_list = list(range(subg_id_min, subg_id_max))

            plt.figure(1, figsize=(23, 11))

            subg_id_list_id = 0
            v = subg_id_list[subg_id_list_id]
            ax = plt.subplot(v, xbound=110)
            ax.xaxis.tick_top()
            ax.axhline(g_aa_pos_max_s, xmin=0.000, xmax=0.020, color='red', linewidth=fig_avg_linewidth, marker=None)
            ax.axhline(g_aa_pos_avg_s, xmin=0.000, xmax=0.015, color='red', linewidth=fig_avg_linewidth, marker=None)
            ax.axhline(g_aa_pos_min_s, xmin=0.000, xmax=0.010, color='red', linewidth=fig_avg_linewidth, marker=None)
            ax.axhline(g_aa_neg_max_s, xmin=0.000, xmax=0.010, color='red', linewidth=fig_avg_linewidth, marker=None)
            ax.axhline(g_aa_neg_avg_s, xmin=0.000, xmax=0.015, color='red', linewidth=fig_avg_linewidth, marker=None)
            ax.axhline(g_aa_neg_min_s, xmin=0.000, xmax=0.020, color='red', linewidth=fig_avg_linewidth, marker=None)
            ax.axhline(g_bb_pos_max_s, xmin=0.020, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
            ax.axhline(g_bb_pos_avg_s, xmin=0.025, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
            ax.axhline(g_bb_pos_min_s, xmin=0.030, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
            ax.axhline(g_bb_neg_max_s, xmin=0.030, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
            ax.axhline(g_bb_neg_avg_s, xmin=0.025, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
            ax.axhline(g_bb_neg_min_s, xmin=0.020, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
            plt.title(fig_title, loc='left', fontdict={'fontsize': 8})
            plt.plot(
                g_aa,
                label='data_a: ' + st_a_info + ' ' + g_data_a_info,
                linewidth=0,
                marker='.',
                markersize=fig_markersize_a,
                markeredgewidth=0,
                markerfacecolor='red',
                alpha=fig_alpha,
            )
            plt.plot(
                g_bb,
                label='data_b: ' + st_b_info + ' ' + g_data_b_info,
                linewidth=0,
                marker='.',
                markersize=fig_markersize_b,
                markeredgewidth=0,
                markerfacecolor='blue',
                alpha=fig_alpha,
            )
            leg = plt.legend(loc='upper left', frameon=True, fontsize=fig_legend_fontsize)
            leg.get_frame().set_alpha(fig_legend_alpha)
            plt.setp(leg.get_texts(), color=fig_legend_label_color)

            if self.is_calc_inc:
                subg_id_list_id += 1
                v = subg_id_list[subg_id_list_id]
                ax = plt.subplot(v)
                ax.xaxis.set_visible(False)
                ax.axhline(
                    g_diff_inc_pos_max_s, xmin=0.000, xmax=0.020, color='red', linewidth=fig_avg_linewidth, marker=None
                )
                ax.axhline(
                    g_diff_inc_pos_avg_s, xmin=0.000, xmax=0.015, color='red', linewidth=fig_avg_linewidth, marker=None
                )
                ax.axhline(
                    g_diff_inc_pos_min_s, xmin=0.000, xmax=0.010, color='red', linewidth=fig_avg_linewidth, marker=None
                )
                ax.axhline(
                    g_diff_inc_neg_max_s, xmin=0.000, xmax=0.010, color='red', linewidth=fig_avg_linewidth, marker=None
                )
                ax.axhline(
                    g_diff_inc_neg_avg_s, xmin=0.000, xmax=0.015, color='red', linewidth=fig_avg_linewidth, marker=None
                )
                ax.axhline(
                    g_diff_inc_neg_min_s, xmin=0.000, xmax=0.020, color='red', linewidth=fig_avg_linewidth, marker=None
                )
                plt.plot(
                    g_diff_inc,
                    label='diff_inc (=a-b): ' + g_diff_info,
                    linewidth=0,
                    marker='.',
                    markersize=fig_markersize,
                    markeredgewidth=0,
                    markerfacecolor='blue',
                    alpha=fig_alpha,
                )
                leg = plt.legend(loc='upper left', frameon=True, fontsize=fig_legend_fontsize)
                leg.get_frame().set_alpha(fig_legend_alpha)
                plt.setp(leg.get_texts(), color=fig_legend_label_color)

            if self.is_calc_rel:
                subg_id_list_id += 1
                v = subg_id_list[subg_id_list_id]
                ax = plt.subplot(v)
                ax.xaxis.set_visible(False)
                ax.axhline(
                    g_diff_rel_max_s, xmin=0.000, xmax=0.020, color='red', linewidth=fig_avg_linewidth, marker=None
                )
                ax.axhline(
                    g_diff_rel_avg_s, xmin=0.000, xmax=0.015, color='red', linewidth=fig_avg_linewidth, marker=None
                )
                ax.axhline(
                    g_diff_rel_min_s, xmin=0.000, xmax=0.010, color='red', linewidth=fig_avg_linewidth, marker=None
                )
                ax.axhline(
                    g_diff_rel_pos_max_s, xmin=0.020, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None
                )
                ax.axhline(
                    g_diff_rel_pos_avg_s, xmin=0.025, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None
                )
                ax.axhline(
                    g_diff_rel_pos_min_s, xmin=0.030, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None
                )
                plt.plot(
                    g_diff_rel,
                    label='diff_rel (=|a-b|/(|a|+|b|)): ' + diff_rel_info,
                    linewidth=0,
                    marker='.',
                    markersize=fig_markersize,
                    markeredgewidth=0,
                    markerfacecolor='blue',
                    alpha=fig_alpha,
                )
                leg = plt.legend(loc='upper left', frameon=True, fontsize=fig_legend_fontsize)
                leg.get_frame().set_alpha(fig_legend_alpha)
                plt.setp(leg.get_texts(), color=fig_legend_label_color)

            plt.subplots_adjust(hspace=0.1)
            plt.savefig(f_out_fig, bbox_inches='tight', transparent=True)
