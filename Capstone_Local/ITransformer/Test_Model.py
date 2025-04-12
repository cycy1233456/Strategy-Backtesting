from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt


class Model_test:
    def __init__(self, configs):
        self.test_model = Model(configs)
        self.test_model.load_state_dict(torch.load(configs.model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        self.test_model = self.test_model.eval()

    def predict(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        result = self.test_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return result

if __name__ == '__main__':
    test_data_path1 = '/content/drive/MyDrive/168-Fintech-Captone/Capstone_data/Indices_data/data5Min/QQQ_test.csv'
    test_data_path2 = '/content/drive/MyDrive/168-Fintech-Captone/Capstone_data/Indices_data/data30S/QQQ_test.csv'

    model_save_path1 = '/content/drive/MyDrive/168-Fintech-Captone/iTransformer_result/5Min/iTransformer_model.pth'
    model_save_path2 = '/content/drive/MyDrive/168-Fintech-Captone/iTransformer_result/30S/iTransformer_model.pth'

    import argparse
    parser = argparse.ArgumentParser(description='ITransformer')

    parser.add_argument('--seq_len', type=int, required=False, default=288, help='input the sequence of length')
    parser.add_argument('--pred_len', type=int, required=False, default=96, help='output the sequence of length')
    parser.add_argument('--d_model', type=int, required=False, default=128, help='the dimension of model')
    parser.add_argument('--n_layers', type=int, required=False, default=4, help='the number of layers')
    parser.add_argument('--factor', type=int, required=False, default=7, help='the number of features')
    parser.add_argument('--n_heads', type=int, required=False, default=4, help='the number of heads')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--use_norm', type=int, default=False, help='use norm and denorm')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--lr', type=str, default=0.0001, help='learning rate')
    parser.add_argument('--freq', type=str, default='s', help='time frequency')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_path', type=str, default=model_save_path2)  #记得改

    args = parser.parse_args([])

    # 实例化测试模型
    iTransformer_ = Model_test(configs=args)

    # 最大-最小值
    history_train = pd.read_csv(test_data_path2)  #记得改


    # 测试数据
    t_d_process = test_data_process(path=test_data_path2, window_length=3, num_features=7) #记得改
    t_x, t_y = t_d_process.do()
    t_data_set = dataset(x=t_x, y=t_y)
    test_loader = DataLoader(dataset=t_data_set, num_workers=2, shuffle=False, batch_size=1)

    # 测试时间
    df = t_d_process.data['Datetime']
    df['timestamp'] = pd.to_datetime(t_d_process.data['Datetime'])
    # 提取日期部分，并获取唯一值
    unique_dates = df['timestamp'].dt.date.unique()
    # 格式化日期为 'YYYY-MM-DD'
    #test_time = [pd.Timestamp(date).strftime('%Y-%m-%d') for date in unique_dates][2:]
    test_time = [pd.Timestamp(date).strftime('%Y-%m-%d') for date in unique_dates]
    window_len = 3

    mape_list = []

    for idx, (x, y) in enumerate(test_loader):

        predict_mean = history_train.iloc[96*(idx+window_len-1):96*(idx+window_len), -3].mean()
        predict_std = history_train.iloc[96*(idx+window_len-1):96*(idx+window_len), -3].std()
        predict_result = iTransformer_.predict(x_enc=x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
        predict_result = predict_result * predict_std + predict_mean

        predict_mape = mean_absolute_percentage_error(predict_result[0].tolist(), y[0].tolist())
        mape_list.append(predict_mape)

        # 可视化预测和真实结果
        if idx <= 4:
          plt.figure(figsize=(8, 5))
          #plt.title('Date:' +  test_time[idx] +' \n MAE:'+str(predict_mae))
          plt.xlabel('Time')
          plt.ylabel('QQQ close price')
          plt.ylim(bottom=200)
          plt.ylim(top=300)
          plt.plot(range(0, 96), predict_result[0].tolist(), label='Predict')
          plt.plot(range(0, 96), y[0].tolist(), label='Real')
          plt.legend()
          plt.show()
    total_mae = np.mean(mape_list)
    print(f'Total MAPE on test data: {total_mae}')
