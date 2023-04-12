
import requests
import pandas as pd
import time
from multiprocessing.pool import ThreadPool

pd.set_option('expand_frame_repr', False)



#aaaaaaaaaa




BASE_URL = 'https://api.binance.com'
#
kline = '/api/v1/klines'

def get_history_data(coin,start_time=None,end_time=None):
    BASE_URL = 'https://api.binance.com'
    limit = 1000
    for time_level in ['1d','1h','1m']: 
        end_time = 1674553742000 # timestamp*1000,because timestamp use second and Binance use mircoseconds，1s=1000ms。Date and time (GMT): Saturday, 1 January 2022 00:00:00     #end_time=int(time.time() // 60 * 60 * 1000)    
        if  time_level=='1d' :
              start_time = int(end_time - limit*24*60*60*1000) 
        elif  time_level=='1h' :
              start_time = int(end_time - limit*60*60*1000)
        else:               
            start_time = int(end_time - limit*60*1000)                    
        df_total=pd.DataFrame()
        while True:
            url = BASE_URL + '/api/v3/klines' + f'?symbol={coin}USDT&interval={time_level}&limit=' + str(limit) + '&startTime=' + str(
                start_time) + '&endTime=' + str(end_time)
            #print(url)
            resp = requests.get(url)
            try:
                data = resp.json()
                df = pd.DataFrame(data,columns={'open_time': 0, 'open': 1, 'high': 2, 'low': 3, 'close': 4, 'volume': 5,
                                                'close_time': 6, 'quote_volume': 7, 'trades': 8, 'taker_base_volue': 9,
                                                'taker_quote_volume': 10, 'ignore': 11})
            except Exception as e:
                print(e)
            else:
                df_total=pd.concat([df_total,df],ignore_index=True)
            end_time = start_time
            if time_level=='1d':
                start_time = int(end_time - limit*60*24*60*1000)
            elif  time_level=='1h':
                start_time = int(end_time - limit*60*60*1000)
            else:  
                start_time = int(end_time - limit*60*1000)
        
            if end_time <1501286400000:
            #if end_time <1640822400000:# 30 December 2021 00:00:00
                break
        df_total['open_time']=pd.to_datetime(df_total['open_time'],unit='ms')
        df_total=df_total.sort_values(by='open_time')
        df_total['coin']=coin
        df_total.to_csv(f'/home/tao/transformer/{coin}_{time_level}_price1.csv')

      

if __name__ == '__main__':
    pool_size = 107
    coin_collect=['ETH']
    pool = ThreadPool(pool_size)
    pool.map(get_history_data,coin_collect)
    pool.close()
    pool.join()
    
    

