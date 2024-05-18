import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import numpy as np

# 从pickle文件中加载测试数据
with open("security_test.csv.pkl", "rb") as f:
    file_names = pickle.load(f)
    outfiles = pickle.load(f)

# 从pickle文件中加载训练数据
with open("security_train.csv.pkl", "rb") as f:
    labels = pickle.load(f)
    files = pickle.load(f)

print("start tfidf...")
# 使用TF-IDF提取特征
vectorizer = TfidfVectorizer(ngram_range=(1, 5), min_df=3, max_df=0.9)
train_features = vectorizer.fit_transform(files)
test_features = vectorizer.transform(outfiles)

# 打印特征矩阵的形状
print(train_features.shape)
print(test_features.shape)

# 将标签转换为整数类型
labels = labels.astype(int)

# 初始化meta数据，用于存储交叉验证的预测结果
meta_train = np.zeros(shape=(len(files), 8))
meta_test = np.zeros(shape=(len(outfiles), 8))

# 使用StratifiedKFold进行交叉验证
skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
for i, (tr_ind, te_ind) in enumerate(skf.split(train_features, labels)):
    X_train, X_train_label = train_features[tr_ind], labels[tr_ind]
    X_val, X_val_label = train_features[te_ind], labels[te_ind]

    print('FOLD: {}'.format(str(i)))
    print(len(te_ind), len(tr_ind))

    # 准备XGBoost的训练和测试数据
    dtrain = xgb.DMatrix(X_train, label=X_train_label)
    dtest = xgb.DMatrix(X_val, label=X_val_label)
    dout = xgb.DMatrix(test_features)

    # 设置XGBoost的参数
    param = {'max_depth': 6,
             'eta': 0.1,
             'eval_metric': 'mlogloss',
             'silent': 1,
             'objective': 'multi:softprob',
             'num_class': 8,
             'subsample': 0.8,
             'colsample_bytree': 0.85}

    # 训练模型并进行早期停止
    evallist = [(dtrain, 'train'), (dtest, 'val')]
    num_round = 300
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)

    # 对验证集和测试集进行预测
    pred_val = bst.predict(dtest)
    pred_test = bst.predict(dout)

    # 存储预测结果
    meta_train[te_ind] = pred_val
    meta_test += pred_test

# 平均测试集预测结果
meta_test /= 5.0

# 将结果保存到pickle文件中
with open("xgboost_result.pkl", 'wb') as f:
    pickle.dump(meta_train, f)
    pickle.dump(meta_test, f)
