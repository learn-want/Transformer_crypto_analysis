import os
coins=['btc','eth']
# coins=['eth']
# lbw=[7,14,21,30,60]
lbw=[90,120,180]
for coin in coins:
    for l in lbw:
        try:
            os.system(f'python cpd_quandl.py  -t {coin} -f /home/tao/transformer/resive_trading-momentum-transformer/data/quandl/{coin}_cpd_{l}.csv -s 2017-08-17 -t 2023-01-24 -l {l}')
        except Exception as e:
            print(e)