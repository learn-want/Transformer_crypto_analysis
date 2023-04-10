import os
from backtest import run_classical_methods

#选取数据做 train 和 backtest
# INTERVALS = [(1990, y, y + 1) for y in range(2016, 2022)]
#改为
INTERVALS = [(2017, y, y + 1) for y in range(2020, 2023)]
#前两个数字是 tarin，后两个数字是 test，即（1990,2016），（2016,2017）
REFERENCE_EXPERIMENT = "experiment_quandl_100assets_tft_cpnone_len252_notime_div_v1"

features_file_path = os.path.join(
    "data",
    "quandl_cpd_nonelbw.csv",
)

run_classical_methods(features_file_path, INTERVALS, REFERENCE_EXPERIMENT)
