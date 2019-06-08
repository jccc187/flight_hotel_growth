import logging
import datetime
import time


class DefaultConfig():
    def __init__(self):
        # ------------ 数据路径 ------------
        # 数据集根目录
        # self.base_dir = '/home/q/tmp/f_algorithm_model/flight_growth/'
        # 预测数据路径
        self.predict_behavior_data = '../dataset/ftg_v3_behavior_0523_sample.zip'
        self.predict_pip_data = '../dataset/ftg_v3_pip_0523_sample.zip'
        self.predict_usertag_data = '../dataset/ftg_v3_usertag_0523_sample.zip'
        self.train_behavior_data = '../dataset/ftg_v3_behavior_0408_sample.zip'
        self.train_pip_data = '../dataset/ftg_v3_pip_0408_sample.zip'
        self.train_usertag_data = '../dataset/ftg_v3_usertag_0408_sample.zip'
        self.train_wideorder_data = '../dataset/ftg_v3_wideorder_0408_sample.zip'

        # 模型保存路径
        self.model_dir = '../model'
        # 模型名称
        self.model_pkl = 'ftg_model_v3'
        self.model_pkl_sample = 'ftg_model_v3_sample'
        # 结果保存文件名
        self.result_dir = '../result'
        self.result_filename = 'ftg_pred_0523_sample_result.csv'  # 修改保存的结果文件名

        # xgboost模型参数
        self.booster = 'gbtree'
        self.objective = 'binary:logistic'
        self.gamma = 0.03
        self.max_depth = 8
        self.lambda_xgb = 1
        self.alpha = 1
        self.subsample = 0.75
        self.colsample_bytree = 0.9
        self.colsample_bylevel = 0.9
        self.eval_metric = 'auc'
        self.min_child_weight = 0.8
        self.max_delta_step = 0
        self.silent = 0
        self.eta = 0.01
        self.seed = 123
        self.scale_pos_weight = 1
        self.tree_method = 'auto'
        self.nthread = -1
        self.early_stopping_rounds = 50
        self.num_boost_round = 500

        # 阈值设置
        self.threshold = 0.5

        # dt设置
        self.dt = '2019-05-23' #'2019-05-23'  #注意修改dt,用于确定预测数据集的dt

        # 数据采样的样本量设置
        self.n_sample = 500  #注意修改n_sample,用于确定训练集的正负采样样本量


        # 创建一个日志对象
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # 创建Handler，用于写入日志文件
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        fh = logging.FileHandler('../ftg-logger.log', mode='a')  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
        # a是追加模式，默认如果不写的话，就是追加模式
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 控制台输出日志
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)


    def getdate(self):
        yesterday = (datetime.date.today() + datetime.timedelta(days=-1)).strftime('%Y%m%d')
        return str(yesterday)

    def getUpdateDate(self):
        yesterday = (datetime.date.today() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
        return str(yesterday)