import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 数据读取
path = './dataset/'
train = pd.read_csv(path + 'security_train.csv')
test = pd.read_csv(path + 'security_test.csv')

# 数据结构显示
print("Train Data:")
print(train.head())
print("Test Data:")
print(test.head())

# 基本数据分析
def basic_data_analysis(df):
    print("Data Info:")
    print(df.info())
    print("Data Describe:")
    print(df.describe())
    print("Data Null Values:")
    print(df.isnull().sum())
    print("Data different Values:")
    print(df.nunique())

basic_data_analysis(train)
basic_data_analysis(test)

# 数据label分布图
def label_distribution(df, column, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(title)
    plt.show()

label_distribution(train, 'label', 'Train Label Distribution')
label_distribution(test, 'label', 'Test Label Distribution')

# 数据集联合分析
train_fileids = train["file_id"].unique()
test_fileids = test["file_id"].unique()
len1 = len(set(train_fileids) - set(test_fileids))
len2 = len(set(test_fileids) - set(train_fileids))
print(len1, len2)

train_apis = train["api"].unique()
test_apis = test["api"].unique()
print(set(train_apis) - set(test_apis))
print(set(test_apis) - set(train_apis))

### 高级数据探索
# file_id与api之间的关系：调用API次数
train_analysis = train[['file_id', 'label']].drop_duplicates(subset=['file_id', 'label'], keep='last') # 首先进行重复数据清理
dic_ = train['file_id'].value_counts().to_dict()
train_analysis['file_id_cnt'] = train_analysis['file_id'].map(dic_).values
train_analysis['file_id_cnt'].value_counts()
sns.distplot(train_analysis['file_id_cnt'])
plt.show()
print('There are {} data are below 10000'.format(np.sum(train_analysis['file_id_cnt'] <= 1e4) / train_analysis.shape[0]))

# file_id_cnt & label 分析
def file_id_cnt_cut(x):
    if x < 15000:
        return x // 1e3
    else:
        return 15

train_analysis['file_id_cnt_cut'] = train_analysis['file_id_cnt'].map(file_id_cnt_cut).values

plt.figure(figsize=[16, 20])
plt.subplot(321)
train_analysis[train_analysis['file_id_cnt_cut'] == 0]['label'].value_counts().sort_index().plot(kind='bar')
plt.title('file_id_cnt_cut = 0')
plt.xlabel('label')
plt.ylabel('label_number')

plt.subplot(322)
train_analysis[train_analysis['file_id_cnt_cut'] == 1]['label'].value_counts().sort_index().plot(kind='bar')
plt.title('file_id_cnt_cut = 1')
plt.xlabel('label')
plt.ylabel('label_number')

plt.subplot(323)
train_analysis[train_analysis['file_id_cnt_cut'] == 14]['label'].value_counts().sort_index().plot(kind='bar')
plt.title('file_id_cnt_cut = 14')
plt.xlabel('label')
plt.ylabel('label_number')

plt.subplot(324)
train_analysis[train_analysis['file_id_cnt_cut'] == 15]['label'].value_counts().sort_index().plot(kind='bar')
plt.title('file_id_cnt_cut = 15')
plt.xlabel('label')
plt.ylabel('label_number')

plt.subplot(313)
train_analysis['label'].value_counts().sort_index().plot(kind='bar')
plt.title('All Data')
plt.xlabel('label')
plt.ylabel('label_number')
plt.show()

plt.figure(figsize=[16, 10])
sns.swarmplot(x=train_analysis.iloc[:1000]['label'], y=train_analysis.iloc[:1000]['file_id_cnt'])
plt.show()

# file_id与api之间的关系：调用API类别
dic_ = train.groupby('file_id')['api'].nunique().to_dict()
train_analysis['file_id_api_nunique'] = train_analysis['file_id'].map(dic_).values
print(train_analysis['file_id_api_nunique'].describe())
sns.distplot(train_analysis['file_id_api_nunique'])
plt.show()

train_analysis.loc[train_analysis.file_id_api_nunique >= 100]['label'].value_counts().sort_index().plot(kind='bar')
plt.title('File with api nunique >= 100')
plt.xlabel('label')
plt.ylabel('label_number')
plt.show()

plt.figure(figsize=[16, 10])
sns.boxplot(x=train_analysis['label'], y=train_analysis['file_id_api_nunique'])
plt.show()

# index与file_id之间的关系：寻找index极端
dic_ = train.groupby('file_id')['index'].nunique().to_dict()
train_analysis['file_id_index_nunique'] = train_analysis['file_id'].map(dic_).values
print(train_analysis['file_id_index_nunique'].describe())
sns.distplot(train_analysis['file_id_index_nunique'])
plt.show()

dic_ = train.groupby('file_id')['index'].max().to_dict()
train_analysis['file_id_index_max'] = train_analysis['file_id'].map(dic_).values
sns.distplot(train_analysis['file_id_index_max'])
plt.show()

plt.figure(figsize=[16, 8])
plt.subplot(121)
train_analysis.loc[train_analysis.file_id_index_nunique == 1]['label'].value_counts().sort_index().plot(kind='bar')
plt.title('File with index nunique = 1')
plt.xlabel('label')
plt.ylabel('label_number')

plt.subplot(122)
train_analysis.loc[train_analysis.file_id_index_nunique == 5001]['label'].value_counts().sort_index().plot(kind='bar')
plt.title('File with index nunique = 5001')
plt.xlabel('label')
plt.ylabel('label_number')
plt.show()

plt.figure(figsize=[16, 10])
sns.violinplot(x=train_analysis['label'], y=train_analysis['file_id_api_nunique'])
plt.show()

plt.figure(figsize=[16, 10])
sns.violinplot(x=train_analysis['label'], y=train_analysis['file_id_index_max'])
plt.show()

plt.figure(figsize=[16, 10])
sns.stripplot(x=train_analysis['label'], y=train_analysis['file_id_index_max'])
plt.show()

### 数据分析：file_id和tid之间的关系：借用nunique()
dic_ = train.groupby('file_id')['tid'].nunique().to_dict()
train_analysis['file_id_tid_nunique'] = train_analysis['file_id'].map(dic_).values
print(train_analysis['file_id_tid_nunique'].describe())
sns.distplot(train_analysis['file_id_tid_nunique'])
plt.show()

plt.figure(figsize=[16, 8])
plt.subplot(121)
train_analysis.loc[train_analysis.file_id_tid_nunique < 5]['label'].value_counts().sort_index().plot(kind='bar')
plt.title('File with tid nunique < 5')
plt.xlabel('label')
plt.ylabel('label_number')

plt.subplot(122)
train_analysis.loc[train_analysis.file_id_tid_nunique >= 20]['label'].value_counts().sort_index().plot(kind='bar')
plt.title('File with tid nunique >= 20')
plt.xlabel('label')
plt.ylabel('label_number')
plt.show()

plt.figure(figsize=[12, 8])
sns.boxplot(x=train_analysis['label'], y=train_analysis['file_id_tid_nunique'])
plt.show()

plt.figure(figsize=[12, 8])
sns.violinplot(x=train_analysis['label'], y=train_analysis['file_id_tid_nunique'])
plt.show()

### 数据分析：file_id和tid之间的关系：借用max()
dic_ = train.groupby('file_id')['tid'].max().to_dict()
train_analysis['file_id_tid_max'] = train_analysis['file_id'].map(dic_).values
print(train_analysis['file_id_tid_max'].describe())
sns.distplot(train_analysis['file_id_tid_max'])
plt.show()

plt.figure(figsize=[16, 8])
plt.subplot(121)
train_analysis.loc[train_analysis.file_id_tid_max >= 3000]['label'].value_counts().sort_index().plot(kind='bar')
plt.title('File with tid max >= 3000')
plt.xlabel('label')
plt.ylabel('label_number')

plt.subplot(122)
train_analysis['label'].value_counts().sort_index().plot(kind='bar')
plt.title('All Data')
plt.xlabel('label')
plt.ylabel('label_number')
plt.show()

### 分析API与LABEL之间的关系
train['api_label'] = train['api'] + '_' + train['label'].astype(str)
dic_ = train['api_label'].value_counts().to_dict()
df_api_label = pd.DataFrame.from_dict(dic_, orient='index').reset_index()
df_api_label.columns = ['api_label', 'api_label_count']
df_api_label['label'] = df_api_label['api_label'].apply(lambda x: int(x.split('_')[-1]))

labels = df_api_label['label'].unique()
for label in range(8):
    print('*' * 50, label, '*' * 50)
    print(df_api_label.loc[df_api_label.label == label].sort_values('api_label_count').iloc[-5:][['api_label', 'api_label_count']])
    print('*' * 103)
