from config import DefaultConfig
import pandas as pd
import numpy as np
import warnings
from sklearn import model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import xgboost as xgb
import pickle
import gc
import time
warnings.filterwarnings('ignore')


# 模型参数设置
def model_train(opt, train_x, train_y, val_x, val_y):
    xgb_train = xgb.DMatrix(train_x, train_y, silent=True)
    xgb_val = xgb.DMatrix(val_x, val_y, silent=True)
    params_xgb = {
        'booster': opt.booster,
        'objective': opt.objective,
        'gamma': opt.gamma,
        'max_depth': opt.max_depth,  # 树最大深度
        'lambda': opt.lambda_xgb,  # L2
        'alpha': opt.alpha,  # L1
        'subsample': opt.subsample,
        'colsample_bytree': opt.colsample_bytree,  # 在建立树时对特征采样的比例。缺省值为1
        'colsample_bylevel': opt.colsample_bylevel,
        'eval_metric': opt.eval_metric,
        'min_child_weight': opt.min_child_weight,
        'max_delta_step': opt.max_delta_step,
        'silent': opt.silent,  # 当这个参数值为1时，静默模式开启，不会输出任何信息。一般这个参数就保持默认的0，因为这样能帮我们更好地理解模型。
        'eta': opt.eta,  # 权重衰减因子eta为0.01~0.2
        'seed': opt.seed,  # 随机数的种子。缺省值为0。
        'scale_pos_weight': opt.scale_pos_weight,
        'tree_method': opt.tree_method,
        'nthread': opt.nthread,
        'early_stopping_rounds': opt.early_stopping_rounds}
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    num_boost_round = opt.num_boost_round  # 修改至2000
    plst = params_xgb.items()
    model_xgb = xgb.train(plst, xgb_train, num_boost_round, evals=watchlist, verbose_eval=100, maximize=1)
    return model_xgb


# 加载模型
def load_model(model_file):
    with open(model_file, 'rb') as fin:
        model = pickle.load(fin)
    return model


# 模型验证
def model_validate(X_val, y_val, model, threshold):
    val = xgb.DMatrix(X_val)
    print('model.best_iteration:', model.best_iteration)
    # pred_xgb_1 = model.predict(val,ntree_limit = model.best_iteration)
    pred_xgb_1 = model.predict(val, ntree_limit=1500)
    y_pred_1 = [1 if i > threshold else 0 for i in pred_xgb_1]

    print('预测结果集：', len(y_pred_1))
    print('阈值>%s 为正样本' % threshold)
    print(classification_report(y_val, y_pred_1))
    print('Accracy:', accuracy_score(y_val, y_pred_1))
    print('AUC: %.4f' % metrics.roc_auc_score(y_val, y_pred_1))
    print('ACC: %.4f' % metrics.accuracy_score(y_val, y_pred_1))
    print('Accuracy: %.4f' % metrics.accuracy_score(y_val, y_pred_1))
    print('Recall: %.4f' % metrics.recall_score(y_val, y_pred_1))
    print('F1-score: %.4f' % metrics.f1_score(y_val, y_pred_1))
    print('Precesion: %.4f' % metrics.precision_score(y_val, y_pred_1))

    return pred_xgb_1


def load_data(path, logger):
    data = pd.read_csv(path, compression='zip', sep=',', encoding='utf8', low_memory='False')
    logger.info('数据[%s]加载成功,[%s] '%(path,str(data.shape)))
    return data


def save_model(model, save_path):
    with open(save_path, 'wb') as fout:
        pickle.dump(model, fout)
    print('训练模型保存至：', str(save_path))


def data_process(opt, logger):
    logger.info('机火模型数据预处理...')
    logger.info('1.数据读取开始')
    start = time.time()
    df = load_data(opt.train_behavior_data, logger)
    df_pip = load_data(opt.train_pip_data, logger)
    df_usertag = load_data(opt.train_usertag_data, logger)
    df_pay = load_data(opt.train_wideorder_data, logger)
    logger.info('数据读取完成，用时%f秒' % (time.time() - start))

    logger.info('2.数据清洗...')
    # 数据清洗
    df.dropna(how='all', inplace=True)
    df.drop(['dt'], axis=1, inplace=True)
    df_pip.dropna(how='all', inplace=True)
    cols = ['label']
    df_pip.drop(cols, axis=1, inplace=True)
    df_usertag.dropna(how='all', inplace=True)
    df_usertag.drop(['dom_max_aircompany_ticket_rate'], axis=1, inplace=True)
    # 将变量值转换成numeric
    df_usertag['dom_passengers_buy_rate'] = pd.to_numeric(df_usertag['dom_passengers_buy_rate'], errors='coerce')
    df_usertag['blacklist'] = pd.to_numeric(df_usertag['blacklist'], errors='coerce')
    logger.info('usertag数据集清洗完成: ' + str(df_usertag.shape))
    logger.info('缺失值处理')
    df.fillna(-9)
    df_pip.fillna(-9)
    df_usertag.fillna(-9)

    # 数据集打标签
    logger.info('数据打标签...')
    userlist = df_pay['username'].unique()
    df['label'] = np.where(df.username.isin(userlist), 1, 0)

    logger.info('数据合并...')
    data = pd.merge(df, df_usertag, how='left', on='username')
    data = pd.merge(data, df_pip, how='left', on='username')
    logger.info('数据预处理完成，用时%f秒' % (time.time() - start))
    return data


def train(opt, logger, data):
    logger.info('机火预测模型训练:')
    # ---------------------- 样本拆分 ----------------------
    # 切分正负样本
    df_pos = data[data.label == 1].sample(n=opt.n_sample, random_state=1)
    df_neg = data[data.label == 0].sample(n=opt.n_sample, random_state=43)
    ## 验证集
    df_val = data.drop(index=df_pos.index)
    df_val = df_val.drop(index=df_neg.index)
    print('正样本量：%d,负样本量:%d, 测试集量:%d' % (len(df_pos), len(df_neg), len(df_val)))
    # 合并正负样本
    dfv1 = df_pos.append(df_neg)

    objcols = [i for i in data.dtypes[data.dtypes == np.object].index]
    objcols.append('label')  # 训练时需要将label删除
    df_x = dfv1.drop(objcols, axis=1)
    df_y = dfv1['label']
    print('df_x的数据集大小：\n', df_x.shape)
    print('df_y的数据集大小：\n', df_y.shape)

    # ---------------------- 模型训练 ----------------------

    # 提取训练数据
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.25, random_state=43)
    print('模型训练...')
    model = model_train(opt, x_train, y_train, x_test, y_test)
    print('模型训练结束...')

    # ---------------------- 模型评估 -------------------------
    threshold = 0.5
    model_name = model

    try:
        model
    except NameError:
        model = load_model(model_name)

    valset = df_val.sample(n=5 * (opt.n_sample), random_state=90)
    X_val = valset.drop(objcols, axis=1)
    y_val = valset['label']

    print('模型在测试集上的效果：')
    print('threshold = ', threshold)
    pred = model_validate(X_val, y_val, model, threshold=threshold)
    y_pred = [1 if i >= threshold else 0 for i in pred]

    print('混淆矩阵:')
    df_val = pd.DataFrame()
    df_val['true_label'] = y_val
    df_val['prediction_label'] = y_pred
    df_val['id'] = 1
    pivot_df = pd.crosstab(index=df_val.prediction_label,
                           columns=df_val.true_label,
                           values=df_val.id,
                           aggfunc='count')
    print(pivot_df)
    gc.collect()
    logger.info('模型训练完毕!')

    # ---------------------- 保存模型 ----------------------
    save_path = '{}/{}.model'.format(opt.model_dir, opt.model_pkl_sample)
    save_model(model, save_path)
    logger.info('机火xgb模型训练完毕,模型保存至：' + str(save_path))
    gc.collect()


if __name__ == '__main__':
    opt = DefaultConfig()
    logger = opt.logger
    data = data_process(opt, logger)
    train(opt, logger, data)