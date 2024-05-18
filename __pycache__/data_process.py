import pandas as pd
import pickle
import numpy as np

train_path = r'./dataset/security_train.csv'
test_path = r'./dataset/security_test.csv'

def read_train_file(path):
    labels = []
    files = []
    data = pd.read_csv(path)
    goup_fileid = data.groupby('file_id')
    for file_name, file_group in goup_fileid:
        file_labels = file_group['label'].values[0]
        result = file_group.sort_values(['tid', 'index'], ascending=True)
        api_sequence = ' '.join(result['api'])
        labels.append(file_labels)
        files.append(api_sequence)
    with open(path.split('/')[-1] + ".txt", 'w') as f:
        for i in range(len(labels)):
            f.write(str(labels[i]) + ' ' + files[i] + '\n')

def read_test_file(path):
    names = []
    files = []
    data = pd.read_csv(path)
    goup_fileid = data.groupby('file_id')
    for file_name, file_group in goup_fileid:
        result = file_group.sort_values(['tid', 'index'], ascending=True)
        api_sequence = ' '.join(result['api'])
        names.append(file_name)
        files.append(api_sequence)
    with open("security_test.csv.pkl", 'wb') as f:
        pickle.dump(names, f)
        pickle.dump(files, f)

def load_train2h5py(path="security_train.csv.txt"):
    labels = []
    files = []
    with open(path) as f:
        for i in f.readlines():
            i = i.strip('\n')
            labels.append(i[0])
            files.append(i[2:])
    labels = np.asarray(labels)
    print(labels.shape)
    with open("security_train.csv.pkl", 'wb') as f:
        pickle.dump(labels, f)
        pickle.dump(files, f)

if __name__ == '__main__':
    print("read train file.....")
    read_train_file(train_path)
    load_train2h5py()
    print("read test file......")
    read_test_file(test_path)
