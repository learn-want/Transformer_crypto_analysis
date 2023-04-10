import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import optuna
from optuna.trial import TrialState
torch.manual_seed(0)
np.random.seed(0)
import torch.optim as optim
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")
BACKUP_PATH='/home/tao/transformer/resive_trading-momentum-transformer/data/XTAO.pickle'
import pandas as pd
# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number
#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)

def loadp(filename):
    import pickle
    a = pickle.load(open(BACKUP_PATH, "rb"))
    return a
def get_data(price_or_return='return',multi_features=True,scaler_method=None):
    #怎样寻找超参数
    df=pd.read_csv('/home/tao/transformer/resive_trading-momentum-transformer/data/quandl_cpd_90_30lbw.csv')
    df=df[df['ticker']=='eth']
    df=df[['date','close','daily_returns','daily_vol','target_returns','norm_daily_return','norm_monthly_return','norm_quarterly_return','norm_biannual_return','norm_annual_return','macd_8_24','macd_16_48','macd_32_96','day_of_week','day_of_month','week_of_year','month_of_year','cp_rl_90','cp_score_90','cp_rl_30','cp_score_30']]
    df['date']=pd.to_datetime(df['date'])
    df=df.set_index('date')
    # df = pd.read_csv('/home/tao/transformer/resive_trading-momentum-transformer/data/quandl/BTC_1d_price.csv', usecols=['close'])#'open','high','low','volume'
    # if price_or_return=='return':
    #     df['return']=((df['close'].shift(1)-df['close'])/df['close'])
    #     new_order = ['return'] + [col for col in df.columns if col != 'return']
    #     df = df.reindex(columns=new_order)[1:]
    train_samples = int(0.8 * len(df))
    train_data = df[:train_samples]
    test_data = df[train_samples:]
    
    # train_data, test_data,y_train,y_test = loadp("XTAO") #第一列为 close price，最后一列为 close return 1596,387
    # train_data=train_data.drop(['SPY','^IXIC','DX-Y.NYB','GC=F'],axis=1)
    # test_data=test_data.drop(['SPY','^IXIC','DX-Y.NYB','GC=F'],axis=1)
    ##several import features
    # train_data=train_data[['close_price','low_price', 'RSI_14', 'high_low_diff', 'trend_last_7', 'trend_last_21','macd_7_21']]
    # test_data=test_data[['close_price','low_price', 'RSI_14', 'high_low_diff', 'trend_last_7', 'trend_last_21','macd_7_21']]
    # test_data=test_data[:50]
    if multi_features==True:
        train_data=train_data[21:] # Todo 选择具体变量
        test_data=test_data[21:]
        if price_or_return=='return':
            cols = train_data.columns.tolist() # 获取列名列表
            cols = cols[-1:] + cols[:-1] # 将最后一列移到第一列
            train_data = train_data[cols][1:] # 重新排列 DataFrame 的列
            
            cols = test_data.columns.tolist() # 获取列名列表 
            cols = cols[-1:] + cols[:-1] # 将最后一列移到第一列
            test_data = test_data[cols][1:] # 重新排列 DataFrame 的列
        else:
            pass  
    else:
        if  price_or_return=='return':
            train_data=train_data['close_returns'][1:].to_frame()
            test_data=test_data['close_returns'][1:].to_frame() #series 变成 dataframe
        else:
            train_data=train_data['close_price'].to_frame()
            test_data=test_data['close_price'].to_frame()

    global test_time_index
    test_time_index=test_data.index[input_window:]
    global scaler
    if scaler_method=='max_min':
        scaler = MinMaxScaler(feature_range=(-1, 1)) 
        train_data=scaler.fit_transform(train_data.values.reshape(-1, len(train_data.columns)))
        test_data=scaler.fit_transform(test_data.values.reshape(-1, len(test_data.columns)))

    elif scaler_method=='robust':
        scaler = RobustScaler() 
        train_data=scaler.fit_transform(train_data.values.reshape(-1, len(train_data.columns)))
        test_data=scaler.fit_transform(test_data.values.reshape(-1, len(test_data.columns)))
    else:
        train_data=(train_data.values).reshape(-1,len(train_data.columns))
        test_data=(test_data.values).reshape(-1,len(test_data.columns))
        # no scaler
    # convert our train data into a pytorch train tensor
    #train_tensor = torch.FloatTensor(train_data).view(-1)
    global train_data_rmsse
    train_data_rmsse=train_data
    train_sequence = create_inout_sequences( train_data,input_window)
    '''
    train_sequence = train_sequence[:-output_window] # todo: fix hack? -> din't think this through, looks like the last n sequences are to short, so I just remove them. Hackety Hack..
    # looks like maybe solved
    '''
    #test_data = torch.FloatTensor(test_data).view(-1) 
    test_data = create_inout_sequences(test_data,input_window)
    '''
    test_data = test_data[:-output_window] # todo: fix hack?
    '''
    # shape with (block , sql_len , 2 )
    return train_sequence.to(device),test_data.to(device)
def create_inout_sequences(input_data, tw):# train data 和 look back window
    inout_seq = []
    L = len(input_data)
    # if window is 100 and prediction step is 1
    # in -> [0..99]
    # target -> [1..100]
    for i in range(L - tw): #L - tw 是 sequence 的数量 L is the length of input_data, tw is the length of train_seq
        train_seq = input_data[i:i + tw] # train_seq is the input of the model
        train_label = input_data[i + output_window:i + tw + output_window] # train_label is the output of the model
        inout_seq.append((train_seq, train_label)) # inout_seq is a list of tuples, each tuple contains a train_seq and a train_label

    return torch.FloatTensor(inout_seq) 

def get_batch(input_data, i , batch_size):

    # batch_len = min(batch_size, len(input_data) - 1 - i) #  # Now len-1 is not necessary
    batch_len = min(batch_size, len(input_data) - i)
    data = input_data[ i:i + batch_len ]
    input = torch.stack([item[0] for item in data]).view((input_window,batch_len,data.shape[-1])) #data.shape[-1] 获取列数，即变量个数，这里把data[64.2,14,4] 变为 3 维[14,64,4]
    # ( seq_len, batch, 1 ) , 1 is feature size
    target = torch.stack([item[1] for item in data]).view((input_window,batch_len,data.shape[-1]))
    return input, target
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0).transpose(0, 1) # [5000, 1, d_model],so need seq-len <= 5000
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x): #x 来自于哪里
        # print(self.pe[:x.size(0), :].repeat(1,x.shape[1],1).shape ,'---',x.shape)
        # dimension 1 maybe inequal batchsize
        return x + self.pe[:x.size(0), :].repeat(1,x.shape[1],1) 
        #这行代码是对输入张量 x 和位置编码张量 pe 进行相加操作，其中 x 是一个形状为 (seq_len, batch_size, embedding_size) 的三维张量，pe 是一个形状为 (seq_len, 1, embedding_size) 的三维张量。
        # x: [14,64,250] x 本来是[14,64,4] 为什么最后一维变成了 250， self.input_embedding  = nn.Linear(4,feature_size) 这一行 代码将 4 维转换成了 250 维, 原来每个数据有 4 个特征，但每个数据点现在有 250 个特征。因此，转换后的张量仍然代表相同的数据集，但是在每个数据点上提供了更多的特征信息。
        # pe [5000,1,250]
        #pe[:x.size(0), :] [14,1,250]
        #然后使用 .repeat(1, x.shape[1], 1) 将其沿着第二维（即 batch_size）重复 batch_size 次，并沿着第三维（即 embedding_size）重复 1 次，以便与 x 的形状相匹配。最后，我们将这两个张量相加得到一个形状与 x 相同的张量，其中每个位置都加上了对应位置的位置编码。

class TransAm(nn.Module):
    def __init__(self,trial,feature_num=1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.num_layers = trial.suggest_int("num_layers", 1, 3)
        self.feature_size=trial.suggest_int("feature_size", 250, 260,270)
        self.dropout=trial.suggest_float("dropout", 0.05,0.1)
        self.input_embedding  = nn.Linear(feature_num,self.feature_size)# 关键操作，把原来的 4 维转换为feature_size维度，nn.Linear(4, feature_size) 表示创建一个具有 4 个输入特征和 feature_size 个输出特征的全连接层。
        self.src_mask = None
        
        self.pos_encoder = PositionalEncoding(self.feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=10, dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.decoder = nn.Linear(self.feature_size,feature_num) #表示创建一个具有 feature_size 个输入特征和 4 个输出特征的全连接层。
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        # src with shape (input_window, batch_len, 1)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.input_embedding(src) # linear transformation before positional embedding
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]



def objective(trial):
    train_data, val_data = get_data(price_or_return='price',multi_features=True,scaler_method='max_min')
    model=TransAm(trial,feature_num=20)
    # model.train() # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()
    batch_size=32
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)


    for batch, i in enumerate(range(0, len(train_data), batch_size)):  # Now len-1 is not necessary
        # data and target are the same shape with (input_window,batch_len,1)
        total_loss = 0
        start_time = time.time()
        criterion = nn.MSELoss()
        data, targets = get_batch(train_data, i , batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[-1,-1,0], targets[-1,-1,0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += np.sqrt(loss.item())
        log_interval = int(len(train_data) / batch_size / 5)
        
        cur_loss = total_loss / log_interval
        trial.report(cur_loss, batch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return cur_loss
            
if __name__ == "__main__":
    input_window = 14# number of input steps
    output_window = 1
    storage_name = "sqlite:///optuna.db"
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="minimize",
        study_name="BTC_transformer1", storage=storage_name,load_if_exists=True)
    
    study.optimize(objective, n_trials=20, timeout=1200)

    best_params = study.best_params
    best_value = study.best_value
    print("\n\nbest_value = "+str(best_value))
    print("best_params:")
    print(best_params)
    optuna.visualization.plot_optimization_history(study)


exit()
def RMSSE(true, pred, train): 
    '''
    true: np.array 
    pred: np.array
    train: np.array
    '''
    
    n = len(train)

    numerator = np.mean(np.sum(np.square(true - pred)))
    
    denominator = 1/(n-1)*np.sum(np.square((train[1:] - train[:-1])))
    
    msse = numerator/denominator
    
    return msse ** 0.5

def MAPE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean(np.abs((true-pred)/true))

def plot_and_loss(eval_model, data_source,epoch):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        # for i in range(0, len(data_source) - 1):
        for i in range(len(data_source)):  # Now len-1 is not necessary
            data, target = get_batch(data_source, i , 1) # one-step forecast,如果是多步预测，1 改为其他数字
            output = eval_model(data)            
            total_loss += np.sqrt(criterion(output[-1,-1,0], target[-1,-1,0]).item())
            test_result = torch.cat((test_result, output[-1].cpu()), 0)
            truth = torch.cat((truth, target[-1].cpu()), 0)
            
    #test_result = test_result.cpu().numpy() -> no need to detach stuff.. 
    test_result=scaler.inverse_transform(test_result.reshape(-1, feature_num_from_data)) #test 为矩阵
    # test_result=scaler.inverse_transform(test_result).reshape(-1)

    truth=scaler.inverse_transform(truth.reshape(-1, feature_num_from_data))
    len(test_result)
    # pyplot.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.5)
    pyplot.plot(test_time_index,test_result[:, 0],color="red",label='Prediction')
    pyplot.plot(test_time_index,truth[:, 0],color="blue",label='Real')
    pyplot.xticks(rotation=45)
    # pyplot.legend(loc='upper right')
    # pyplot.plot(test_result-truth,color="green")
    
    #inverse 数据之前
    #Rmse_value=np.sqrt(criterion(test_result, truth).item()) 
    #inverse 数据之后
    Rmse_value=np.sqrt(criterion(torch.from_numpy(test_result[:, 0]), torch.from_numpy(truth[:, 0])).item())
    Mape=MAPE(truth[:, 0],test_result[:, 0])
    Rmsse=RMSSE(truth[:, 0],test_result[:, 0],train_data_rmsse)
    pyplot.text(test_time_index[10], (truth[:, 0].max()-truth[:, 0].min())/2, f'Rmse:{Rmse_value}', fontsize=12, color='red', ha='center', va='top')
    pyplot.text(test_time_index[10], (truth[:, 0].max()-truth[:, 0].min())/3, f'Mape:{Mape}', fontsize=12, color='blue', ha='center', va='top')
    pyplot.text(test_time_index[10], (truth[:, 0].max()-truth[:, 0].min())/4, f'Rmsse:{Rmsse}', fontsize=12, color='black', ha='center', va='top')

    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('./transformer-epoch%d.png'%epoch)
    pyplot.close()
    return total_loss / i


# predict the next n steps based on the input data 
# def predict_future(eval_model, data_source,steps):
#     eval_model.eval() 
#     total_loss = 0.
#     test_result = torch.Tensor(0)    
#     truth = torch.Tensor(0)
#     data, _ = get_batch(data_source , 0 , 1)
#     with torch.no_grad():
#         for i in range(0, steps):            
#             output = eval_model(data[-input_window:])
#             # (seq-len , batch-size , features-num)
#             # input : [ m,m+1,...,m+n ] -> [m+1,...,m+n+1]
#             data = torch.cat((data, output[-1:])) # [m,m+1,..., m+n+1]

#     data = data.cpu().view(-1)
    
    # I used this plot to visualize if the model pics up any long therm structure within the data.
    pyplot.plot(data,color="red")       
    pyplot.plot(data[:input_window],color="blue")    
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('./transformer-future%d.png'%steps)
    pyplot.show()
    pyplot.close()
        

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 64
    with torch.no_grad():
        # for i in range(0, len(data_source) - 1, eval_batch_size): # Now len-1 is not necessary
        for i in range(0, len(data_source), eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)            
            total_loss += len(data[0]) * np.sqrt(criterion(output[-1,-1,0], targets[-1,-1,0]).cpu().item())
    return total_loss / len(data_source)

if __name__=='__main__':
    #设置超参数
    input_window = 14# number of input steps
    output_window = 1 # number of prediction steps, in this model its fixed to one
    lr = 0.005 #学习率是一个非常重要的超参数，它决定了每次参数更新的幅度。如果学习率设置过大，可能会导致模型无法收敛，损失函数不断波动或发生不稳定的震荡；如果学习率设置过小，则会导致模型收敛速度慢，需要更长的时间才能达到最优解。 
    block_len = input_window + output_window # for one input-output pair
    batch_size = 64 #数据里面包含了多少个 seq2seq，64 组 seq2seq
    train_size = 0.8
    train_data, val_data = get_data(price_or_return='price',multi_features=True,scaler_method='max_min')
    feature_num_from_data=train_data.shape[-1]
    model = TransAm(feature_num=feature_num_from_data).to(device)
    criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    best_val_loss = float("inf")
    epochs = 200 # The number of epochs
    best_model = None
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data)
        if ( epoch % 20 == 0 ):
            val_loss = plot_and_loss(model, val_data,epoch)
            # predict_future(model, val_data,200)
        else:
            val_loss = evaluate(model, val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} |'.format(epoch, (time.time() - epoch_start_time),
                                   val_loss))
        print('-' * 89)
        scheduler.step() 
