# Transformer_BTC_analysis
This is the project in NII 

Enviroment

Please set up TensorFlow and Pytorch environment.

Code Explanation 

First part:change point detection and Momentum transfer, this part will use TensorFlow. You can run the codes by following the steps:

1. The Bitcoin daily price and Ethereum daily price data are put in the folder /momentum-transformer/data/quandl/btc.csv and eth.csv. You can aslo update the data or get hourly and minute data using the script get_crypto_price.py.

2. Change point detection. Change path to the example folder, please run 'python cpd_quandl.py  -t btc -f /output_path/btc_cpd_200.csv -s 2017-08-17 -t 2023-01-24 -l 200' to get the change point infomation. you an point any length of the CPD window, for example 30,60.

3. Create Momentum Transformer input features with: 'python create_features_quandl.py 21'. Also，you can generate the input features without change point info by removing the CPD_WINDOW_LENGTH 21.

4. Run one of the Momentum Transformer python run_dmn_experiment.py -c <<EXPERIMENT_NAME>>.

Second part: We train a time series Transformer model from the vanilla Transformer,  this part will use Pytorch.
