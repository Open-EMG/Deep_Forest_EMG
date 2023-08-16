import os
import re
import glob
import numpy as np
import wfdb
import pathlib
import hyser_functions as hyser
def get_feature(data, windou_len, step_len, fs):
    feature_name = ['mfl', 'wa', 'vare', 'ssi', 'myop', 'mmav2', 'mmav', 'ld', 'dasdv', 'aac', 'rms', 'wl', 'zc', 'ssc',
                'mav', 'iemg', 'ae', 'var', 'sd', 'cov', 'kurt', 'skew', 'iqr', 'mad', 'damv', 'tm', 'vo', 'dvarv',
                'ldamv', 'ldasdv', 'card', 'lcov', 'ltkeo', 'msr', 'ass', 'asm', 'fzc', 'ewl', 'emav']
    thresh=0.0004
    feature_list=[]
    for i in feature_name:
        if i in ['zc','ssc']:
            feature=eval(f'hyser.get_{i}')(data, windou_len, step_len, thresh, fs)
        else:
            feature=eval(f'hyser.get_{i}')(data, windou_len, step_len, fs)
        feature_list.append(feature)
        if (np.isnan(feature)).any() or (np.isinf(feature)).any():
            print(i)
        del feature
    return np.stack(feature_list).reshape(-1)



def convert_feature(idx,fold_name,data_dir):
    a=np.arange(64)
    a1=a[::-1].reshape(8,8)
    a2=a1+64
    a3=a2+64
    a4=a3+64
    extensor_muscles=np.concatenate((a1,a2),axis=0)
    flexors=np.concatenate((a3,a4),axis=0)
    map=np.concatenate((flexors,extensor_muscles),axis=1)
    map=map.reshape(-1)
    counter = [0] * 34
    FS=2048
    SEGMENTATION_LENGTH= 0.25
    OVERLAP_RATIO=0.5
    DISCARD_LENGTH=0.25
    start_index=int(FS*SEGMENTATION_LENGTH)
    label_select=list(np.array([6, 7, 8, 9, 10, 11, 30, 31, 32, 34])-1)
    MODES='dynamic'
    m=re.search(r'subject(\d+)\_session(\d+)$',fold_name)
    subject_id=int(m[1])
    session=int(m[2])
    file_list=glob.glob(os.path.join(data_dir,fold_name,f"{MODES}_preprocess_sample*.hea"))
    labels=np.loadtxt(os.path.join(data_dir,fold_name,f"label_{MODES}.txt"),delimiter=",",dtype=np.int64)-1
    for j, file_name in enumerate(file_list):
        label_idx=int(re.search(r'\D+(\d+)\.hea$',file_name)[1])
        label=labels[label_idx-1]
        if label in label_select:
            signals, _ = wfdb.rdsamp(file_name[:-4])
            sig_map=np.zeros(signals.shape)
            for k in range(signals.shape[1]):
                sig_map[:,k]=signals[:,int(map[k])]
            feature=get_feature(sig_map[start_index:,:],SEGMENTATION_LENGTH, OVERLAP_RATIO*SEGMENTATION_LENGTH,FS)
            save_current_path=save_dir+f'/subject_{subject_id}'+f'/session_{session}'+f'/{int(label)}'+f"/{counter[label]}"
            pathlib.Path(save_current_path).mkdir(parents=True,exist_ok=True)
            np.savez(os.path.join(save_current_path,"data.npz"),x=feature,y=label_select.index(label))
            counter[label] = counter[label] + 1
    print(f'subject_{subject_id}_done')

if __name__ == '__main__':
    data_dir='/home/data/hyser_dataset_v1/physionet.org/files/hd-semg/1.0.0/pr_dataset'
    save_dir='./feature_data'
    folder_list = os.listdir(data_dir)
    folder_list.sort()
    for idx, fold_name in enumerate(folder_list[1:]):
        convert_feature(idx,fold_name,data_dir)