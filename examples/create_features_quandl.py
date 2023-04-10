import argparse
import datetime as dt
from typing import List
import os

import pandas as pd
from tensorflow.config.experimental import list_physical_devices, set_memory_growth
physical_devices = list_physical_devices('GPU')
set_memory_growth(physical_devices[0], True)

# print(os.getcwd())

from pull_data import pull_quandl_sample_data
QUANDL_TICKERS=['btc','eth']
from default import (
    # QUANDL_TICKERS,
    CPD_QUANDL_OUTPUT_FOLDER,
    FEATURES_QUANDL_FILE_PATH,
)
from data_prep import (
    deep_momentum_strategy_features,
    include_changepoint_features,
)



def main(
    tickers: List[str],
    cpd_module_folder: str,
    lookback_window_length: int,
    output_file_path: str,
    extra_lbw: List[int],
):
    #将 tickers 改为 btc 和 eth
    tickers=['btc','eth']
    features = pd.concat(
        [
            deep_momentum_strategy_features(pull_quandl_sample_data(ticker)).assign(
                ticker=ticker
            )
            for ticker in tickers
        ]
    )
    
    features.date = features.index
    features.index.name = "Date"

    if lookback_window_length:
        features_w_cpd = include_changepoint_features(
            features, cpd_module_folder, lookback_window_length
        )

        if extra_lbw:
            for extra in extra_lbw:
                print('output_file_path',output_file_path)
                extra_data = pd.read_csv(              #读取已存在的两个文件夹的数据
                    output_file_path.replace(
                        f"quandl_cpd_{lookback_window_length}lbw.csv",
                        f"quandl_cpd_{extra}lbw.csv",
                    ),
                    index_col=0,
                    parse_dates=True,
                ).reset_index()[
                    ["date", "ticker", f"cp_rl_{extra}", f"cp_score_{extra}"]
                ]
                extra_data["date"] = pd.to_datetime(extra_data["date"])

                features_w_cpd = pd.merge(
                    features_w_cpd.set_index(["date", "ticker"]),
                    extra_data.set_index(["date", "ticker"]),
                    left_index=True,
                    right_index=True,
                ).reset_index()
                features_w_cpd.index = features_w_cpd["date"]
                features_w_cpd.index.name = "Date"
        else:
            features_w_cpd.index.name = "Date"
        features_w_cpd.to_csv(output_file_path)
    else:
        features.to_csv(output_file_path)


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(description="Run changepoint detection module")
        # parser.add_argument(
        #     "cpd_module_folder",
        #     metavar="c",
        #     type=str,
        #     nargs="?",
        #     default='',
        #     # choices=[],
        #     help="Input folder for CPD outputs.",
        # )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=21,
            # choices=[],
            help="Input folder for CPD outputs.",
        )
        # parser.add_argument(
        #     "output_file_path",
        #     metavar="f",
        #     type=str,
        #     nargs="?", #注释掉之后表示不能改，但是在 debug 的时候必须去掉注释，不然报参数错误
        #     default='',
        #     # choices=[],
        #     help="Output file location for csv.",
        # )
        parser.add_argument(
            "extra_lbw",
            metavar="-e",
            type=int,
            nargs="*",
            default=[],
            # choices=[],
            help="Fill missing prices.",
        )

        args = parser.parse_known_args()[0]
        # print(args.cpd_module_folder)
        # print(args.output_file_path)
        print(CPD_QUANDL_OUTPUT_FOLDER(args.lookback_window_length))
        print(FEATURES_QUANDL_FILE_PATH(args.lookback_window_length))
        return (
            QUANDL_TICKERS,
            CPD_QUANDL_OUTPUT_FOLDER(args.lookback_window_length),
            args.lookback_window_length,
            FEATURES_QUANDL_FILE_PATH(args.lookback_window_length),
            args.extra_lbw,
        )

    main(*get_args())
