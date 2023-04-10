import argparse
import datetime as dt

import pandas as pd
import os
print(os.getcwd())
import sys 
sys.path.append("..") 
import changepoint_detection as cpd
from data_prep import calc_returns
from pull_data import pull_quandl_sample_data

from default import CPD_DEFAULT_LBW, USE_KM_HYP_TO_INITIALISE_KC


def main(
    ticker: str, output_file_path: str, start_date: dt.datetime, end_date: dt.datetime, lookback_window_length :int
):
    data = pull_quandl_sample_data(ticker)
    data["daily_returns"] = calc_returns(data["close"])

    cpd.run_module(
        data, lookback_window_length, output_file_path, start_date, end_date, USE_KM_HYP_TO_INITIALISE_KC
    )


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(description="Run changepoint detection module")
        parser.add_argument(
            "ticker", #某种币
            metavar="t",
            type=str,
            # nargs="?",
            default="btc",
            # choices=[],
            help="Ticker type",
        )
        parser.add_argument(
            "output_file_path",
            metavar="f",
            type=str,
            # nargs="?",
            default="data/test1.csv",
            # choices=[],
            help="Output file location for csv.",
        )
        parser.add_argument(
            "start_date",
            metavar="s",
            type=str,
            # nargs="?",
            default="2017-08-17",
            help="Start date in format yyyy-mm-dd",
        )
        parser.add_argument(
            "end_date",
            metavar="e",
            type=str,
            # nargs="?",
            default="2023-01-24",
            help="End date in format yyyy-mm-dd",
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            # nargs="?",
            default=CPD_DEFAULT_LBW,
            # default=1,
            help="CPD lookback window length",
        )

        args = parser.parse_known_args()[0]

        start_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d")

        return (
            args.ticker,
            args.output_file_path,
            start_date,
            end_date,
            args.lookback_window_length
        )

    main(*get_args())
