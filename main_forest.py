from deepforest import CascadeForestClassifier
from sklearn import preprocessing
import numpy as np
import glob
import os
import torch
from sklearn.metrics import accuracy_score
def _gen_dsuk_fold(num,SAVE_PATH):
    session_i=1
    class_num = 34
    subj_file_list = os.listdir(os.path.join(SAVE_PATH))
    subj_file_list.sort()
    train_samples = []
    test_samples = []
    test_samples1 = []
    test_samples2 = []
    # choose one subjects to test
    for subj_file_name in subj_file_list:
        for label in range(class_num):
            if subj_file_name == f"subject_{num}":
                break
            temp_train1 = glob.glob(os.path.join(SAVE_PATH, subj_file_name, f"session_{session_i}", str(label), "*"))
            temp_train2 = glob.glob(os.path.join(SAVE_PATH, subj_file_name, f"session_{session_i+1}", str(label), "*"))
            np.random.shuffle(temp_train1)
            np.random.shuffle(temp_train2)
            train_samples += temp_train1+temp_train2

    for label in range(class_num):
        temp_test1 = glob.glob(os.path.join(SAVE_PATH, f"subject_{num}", f"session_{session_i}", str(label), "*"))
        temp_test2 = glob.glob(os.path.join(SAVE_PATH, f"subject_{num}", f"session_{session_i+1}", str(label), "*"))
        np.random.shuffle(temp_test1)
        np.random.shuffle(temp_test2)
        # test_samples += temp_test1+temp_test2
        test_samples1 += temp_test1
        test_samples2 += temp_test2
    test_samples =test_samples1 + test_samples2
    train_length = len(train_samples)
    test_length = len(test_samples)
    print(f"train length: {train_length} \n test length:{test_length} ")
    # print(f"train length: {train_length}")
    return train_samples,test_samples


def pre_data(train_data_loader,test_data_loader):
    train_label = []
    train_data = []
    y_true_list = []
    y_prod_list = []
    for path in train_data_loader:
        train_y = np.load(path+'/data.npz')["y"]
        train_x = np.load(path + '/data.npz')["x"]
        train_label.append(train_y.astype('long'))
        train_data.append(train_x.astype('float32'))
    label_feed = np.hstack(train_label)
    data_feed = np.stack(train_data)

    nan_index = np.where(np.isnan(data_feed))[0]
    data_feed = np.delete(data_feed, nan_index, axis=0)
    label_feed = np.delete(label_feed, nan_index, axis=0)

    inf_index = np.where(np.isinf(data_feed))[0]
    data_feed = np.delete(data_feed, inf_index, axis=0)
    label_feed = np.delete(label_feed, inf_index, axis=0)
    # data_feed = np.nan_to_num(data_feed)

    for path in test_data_loader:
        test_y =np.load(path+'/data.npz')["y"]
        test_x =np.load(path + '/data.npz')["x"]
        y_true_list.append(test_y.astype('long'))
        y_prod_list.append(test_x.astype('float32'))

    y_true = np.hstack(y_true_list)
    y_prod = np.stack(y_prod_list)

    nan_index = np.where(np.isnan(y_prod))[0]
    y_prod = np.delete(y_prod, nan_index, axis=0)
    y_true = np.delete(y_true, nan_index, axis=0)

    inf_index = np.where(np.isinf(y_prod))[0]
    y_prod = np.delete(y_prod, inf_index, axis=0)
    y_true = np.delete(y_true, inf_index, axis=0)
    # y_prod = np.nan_to_num(y_prod)

    return data_feed,label_feed,y_prod,y_true

def normalize(data,channel_num,feature_num):
    normal_data=[]
    for i in range(feature_num):
        normal_data.append(preprocessing.scale(data[:,channel_num*5*i:channel_num*5*(i+1)],axis=0))
    data_out=np.hstack(normal_data)
    return data_out


def train_model(data_feed,label_feed,model_test,test_type,feature_order,channel_index):
    if test_type=='all_feature':
        feed = normalize(data_feed, 256, 39)
    elif test_type=='optimization':
        data_feed = data_feed.reshape(-1, 39, 5, 256)
        feature_optimization = data_feed[:, feature_order, :, :]
        channel_optimization = feature_optimization[:, :, :, channel_index.astype('int')] 
        feed = normalize(channel_optimization.reshape(-1, len(feature_order) * 5 * len(channel_index)), channel_optimization.shape[3], len(feature_order))
    # train
    model_test.fit(feed, label_feed)

    return model_test

def test_model(y_prod,y_true,model_test,test_type,feature_order,channel_index):
    if test_type=='all_feature':
        y_prod = normalize(y_prod, 256, 39)
    elif test_type=='optimization':
        y_prod=y_prod.reshape(-1, 39, 5, 256)
        y_prod_f = y_prod[:, feature_order, :, :]
        y_prod_c = y_prod_f[:, :, :, channel_index.astype('int')]
        y_prod = normalize(y_prod_c.reshape(-1,len(feature_order)*5*len(channel_index)), y_prod_c.shape[3], len(feature_order))

    y_pred = model_test.predict(y_prod) 
    return y_pred,y_true

def optimization(data_feed,label_feed,model_optimization, feature_sel, channel_sel):
    model_optimization.fit(data_feed, label_feed)
    feature = model_optimization.get_layer_feature_importances(0)
    # feature = model_optimization.feature_importances_
    feature = feature.reshape(39, 5, 256)
    feature_num, seg, channel_num = feature.shape
    feature_channel = np.mean(feature, axis=1)
    feature_num, channel_num = feature_channel.shape
    feature_name = ['mlf', 'wa', 'vare', 'ssi', 'myop', 'mmav2', 'mmav', 'ld', 'dasdv', ' aac', 'rms', 'wl', 'zc',
                    'ssc', 'mav', 'iemg', 'ae', 'var', 'sd', 'cov', 'kurt', 'skew', 'iqr', 'mad', 'damv', 'tm', 'vo', 'dvarv',
                    'ldamv', 'ldasdv', 'card', 'lcov', 'ltkeo', 'msr', 'ass', 'asm', 'fzc', 'ewl', 'emav']

    just_feature = np.mean(np.mean(feature, axis=1), axis=1)
    order = just_feature.argsort()[::-1]

    order_name = []
    for n in range(feature_sel):
        order_name.append(feature_name[order[n]])
    a = np.zeros((16, 16))
    for k in range(feature_sel):
        a = a + feature_channel[order[k]].reshape(16, 16)
    a=a.reshape(-1)
    # channel_index=np.where(a>a.mean())[0]
    channel_index=np.sort(np.argsort(a)[::-1][:channel_sel])
    return order[:feature_sel], channel_index

if __name__ == '__main__':
    max_layer_number=[1,5,10]
    estimator_number=2
    seed=0
    test_type='optimization'# optimization or all_feature
    data_path='/home/lijianfeng/essay/forest/open/feature_data'
    np.random.seed(seed)
    tree_number=100
    subject_num=20
    layer_acc=[]
    print(f'test type is {test_type}')
    for layer in max_layer_number:
        acc_list=[]
        for num in range(1,subject_num+1):

            model_test = CascadeForestClassifier(max_layers=layer, n_estimators=estimator_number,
                                                    n_trees=tree_number, random_state=1, n_jobs=-1, backend="sklearn",
                                                    use_predictor=True, delta=0, n_tolerant_rounds=10)
            train_samples, test_samples = _gen_dsuk_fold(num,data_path)
            print(f'subject_{num} ready')
            data_feed, label_feed, y_prod, y_true = pre_data(train_samples, test_samples)

            # optimization data
            if test_type=='optimization':
                model_optimization = CascadeForestClassifier(max_layers=1, n_estimators=estimator_number, n_trees=tree_number, random_state=1, n_jobs=-1, backend="sklearn",use_predictor=True)
                
                feature_order, channel_index = optimization(data_feed,label_feed,model_optimization, 4, 106)
            else:
                feature_order,channel_index=None,None
            # train model
            model_test = train_model(data_feed,label_feed,model_test,test_type,feature_order,channel_index)
            # test model
            y_pred,y_true = test_model(y_prod,y_true,model_test,test_type,feature_order,channel_index)
            acc = accuracy_score(y_true, y_pred) * 100
            print(f'subject_{num}')
            print("Testing Accuracy: {:.3f} %\n".format(acc))
            acc_list.append(acc)
        layer_acc.append([np.mean(acc_list),np.std(acc_list)])

    print(f'test type is {test_type}')
    for i in range(len(max_layer_number)):
        print(f'layer_{max_layer_number[i]}_mean_acc:{layer_acc[i][0]}\n layer_{max_layer_number[i]}_std:{layer_acc[i][1]}')       
        