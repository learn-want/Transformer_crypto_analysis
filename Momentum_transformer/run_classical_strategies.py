import os
from backtest import run_classical_methods

#Select data for train and backtest
# INTERVALS = [(1990, y, y + 1) for y in range(2016, 2022)]
#revise to
INTERVALS = [(2017, y, y + 1) for y in range(2020, 2023)]
#The first two numbers are tarin, and the last two numbers are test, namely (1990,2016), (2016,2017)
REFERENCE_EXPERIMENT = "experiment_quandl_100assets_tft_cpnone_len252_notime_div_v1"

features_file_path = os.path.join(
    "data",
    "quandl_cpd_nonelbw.csv",
)

run_classical_methods(features_file_path, INTERVALS, REFERENCE_EXPERIMENT)
