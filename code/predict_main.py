from config import DefaultConfig
import re
import os
import numpy as np
import gc
import pickle
import warnings
import xgboost as xgb
import pandas as pd
import time
warnings.filterwarnings('ignore')

def load_model(model_file):
    with open(model_file, 'rb') as fin:
        model = pickle.load(fin)
    return model
def load_data(path, logger):
    data = pd.read_csv(path, compression='zip', sep=',', encoding='utf8', low_memory='False')
    logger.info('数据[%s]加载成功,[%s] '%(path,str(data.shape)))
    return data

# 数据预处理
def data_process(opt, logger):
    # ---------------------- 读取数据 ----------------------
    logger.info('机火模型数据预处理...')
    logger.info('1.数据读取开始')
    start = time.time()
    df = load_data(opt.predict_behavior_data, logger)
    df_pip = load_data(opt.predict_pip_data, logger)
    df_usertag = load_data(opt.predict_usertag_data, logger)
    logger.info('数据读取完成，用时%f秒' % (time.time() - start))
    # ---------------------- 数据处理 ---------------------
    logger.info('2.数据清洗')
    # df = behavior_data_clean_process(df, logger)
    df.dropna(how='all', inplace=True)
    df.drop(['dt'], axis=1, inplace=True)
    # df_pip = pip_data_clean_process(df_pip, logger)
    # 过滤无用特征
    df_pip.dropna(how='all', inplace=True)
    # 删除全为object的列
    cols = ['label']  # pip中有个标签为'label'，需要将其删除.
    df_pip.drop(cols, axis=1, inplace=True)
    # df_usertag = usertag_data_clean_process(df_usertag, logger)
    # 过滤无用特征
    df_usertag.dropna(how='all', inplace=True)
    # 删除全为object的列
    df_usertag.drop(['dom_max_aircompany_ticket_rate'], axis=1, inplace=True)
    # 将变量值转换成numeric
    df_usertag['dom_passengers_buy_rate'] = pd.to_numeric(df_usertag['dom_passengers_buy_rate'], errors='coerce')
    df_usertag['blacklist'] = pd.to_numeric(df_usertag['blacklist'], errors='coerce')
    logger.info('usertag数据集清洗完成: ' + str(df_usertag.shape))
    # 缺失值处理
    logger.info('缺失值处理')
    df.fillna(-9)
    df_pip.fillna(-9)
    df_usertag.fillna(-9)
    # df = data_miss_process(df, logger)
    # df_pip = data_miss_process(df_pip, logger)
    # df_usertag = data_miss_process(df_usertag, logger)
    logger.info('数据合并')
    # 数据集合并
    # 1)behavior与usertag合并
    data = pd.merge(df, df_usertag, how='left', on='username')
    # 2)继续与pip合并
    data = pd.merge(data, df_pip, how='left', on='username')
    logger.info('数据预处理完成，用时%f秒' % (time.time() - start))
    return data


# ------------- 模型预测函数 ---------------
def predict(opt, logger, data):
    logger.info('机火模型预测:')
    logger.info('生成预测特征...')
    username = data.username
    objcols = [i for i in data.dtypes[data.dtypes == np.object].index]
    x_val = data.drop(objcols, axis=1) # 删除对象类型的字段
    gc.collect()
    # ---------------------- 模型预测 ----------------------
    logger.info('3.开始预测')
    # 模型预测
    load_path = '{}/{}.model'.format(opt.model_dir, opt.model_pkl)
    model = load_model(load_path)
    val = xgb.DMatrix(x_val)
    pred_xgb = model.predict(val, ntree_limit=1500)
    logger.info('模型预测完毕.')
    gc.collect()
    # ---------------------- 保存预测结果 -------------------------

    #pred = (pred_xgb >= threshold) * 1
    res_path = r'{}/{}'.format(opt.result_dir, opt.result_filename)
    res = pd.DataFrame({'qunar_username': username, 'q_ratio': pred_xgb,'true_label':None})
    res.to_csv(res_path, header=True, index=False, sep=',',columns=['qunar_username','true_label','q_ratio'])
    logger.info('保存预测结果文件成功：' + str(res_path))
    logger.info(res.head())


# 将用户登录平台类型编码
def get_source_type(source_type):
    if source_type == 'ios':
        return 1
    elif source_type == 'adr':
        return 0
    else:
        return -1


# 将用户登录机型类型编码
def get_model_type(model_type):
    if model_type == '低端':
        return 0
    elif model_type == '中端':
        return 1
    elif model_type == '高端':
        return 2
    else:
        return -1

if __name__ == '__main__':
    start = time.time()
    opt = DefaultConfig()
    logger = opt.logger
    data = data_process(opt, logger)
    predict(opt,logger,data)
    logger.info('总共耗时%f秒' % (time.time() - start))