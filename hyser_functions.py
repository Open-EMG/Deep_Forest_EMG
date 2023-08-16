import numpy as np
import struct
from scipy import signal
import math
from scipy.stats import kurtosis,skew,iqr
from scipy.signal import welch

def load_1dof(path, subject_idx, session_idx, sig_type):
    if sig_type == 'force':
        M = 5
        N = 25 * 100 * 5
    else:
        M = 256
        N = 25 * 2048 * 256

    path = path + '1dof_dataset'

    data = np.empty((5, 3), dtype=object)
    for i in range(5):
        for j in range(3):
            path_file = path + '/subject' + subject_idx + '_session' + str(
                session_idx) + '/1dof_' + sig_type + '_finger' + str(i + 1) + '_sample' + str(j + 1) + '.dat'
            path_file_hea = path + '/subject' + subject_idx + '_session' + str(
                session_idx) + '/1dof_' + sig_type + '_finger' + str(i + 1) + '_sample' + str(j + 1) + '.hea'
            f = open(path_file, 'rb')
            parameter = '<' + str(N) + 'h'
            data_tmp = np.array(list(struct.unpack(parameter, f.read())))
            f.close()
            data_reshape_tmp = data_tmp.reshape(-1, M)
            data_reshape = data_reshape_tmp.astype('float64')

            fhea = open(path_file_hea, 'r')
            ch_idx = 0
            filename = '1dof_' + sig_type + '_finger' + str(i + 1) + '_sample' + str(j + 1) + '.dat'
            eachline = fhea.readline()
            while eachline:
                if eachline.find(filename) > -1:
                    idx1 = eachline.find('.dat 16 ')
                    idx2 = eachline.find('(')
                    idx3 = eachline.find(')')
                    gain = float(eachline[idx1 + 8:idx2])
                    baseline = float(eachline[idx2 + 1:idx3])
                    tmp = (data_reshape[:, ch_idx] - baseline) / gain
                    data_reshape[:, ch_idx] = list(tmp.reshape(-1, 1))
                    ch_idx = ch_idx + 1
                eachline = fhea.readline()
            fhea.close()
            data[i, j] = data_reshape
    return data


def load_ndof(path, subject_idx, session_idx, sig_type):
    if sig_type == 'force':
        M = 5
        N = 25 * 100 * 5
    else:
        M = 256
        N = 25 * 2048 * 256

    path = path + 'ndof_dataset'
    data = np.empty((15, 2), dtype=object)
    for i in range(15):
        for j in range(2):
            path_file = path + '/subject' + subject_idx + '_session' + str(
                session_idx) + '/ndof_' + sig_type + '_combination' + str(i + 1) + '_sample' + str(j + 1) + '.dat'
            path_file_hea = path + '/subject' + subject_idx + '_session' + str(
                session_idx) + '/ndof_' + sig_type + '_combination' + str(i + 1) + '_sample' + str(j + 1) + '.hea'
            f = open(path_file, 'rb')
            parameter = '<' + str(N) + 'h'
            data_tmp = np.array(list(struct.unpack(parameter, f.read())))
            f.close()
            data_reshape_tmp = data_tmp.reshape(-1, M)
            data_reshape = data_reshape_tmp.astype('float64')

            fhea = open(path_file_hea, 'r')
            ch_idx = 0
            filename = 'ndof_' + sig_type + '_combination' + str(i + 1) + '_sample' + str(j + 1) + '.dat'
            eachline = fhea.readline()
            while eachline:
                if eachline.find(filename) > -1:
                    idx1 = eachline.find('.dat 16 ')
                    idx2 = eachline.find('(')
                    idx3 = eachline.find(')')
                    gain = float(eachline[idx1 + 8:idx2])
                    baseline = float(eachline[idx2 + 1:idx3])
                    tmp = (data_reshape[:, ch_idx] - baseline) / gain
                    data_reshape[:, ch_idx] = list(tmp.reshape(-1, 1))
                    ch_idx = ch_idx + 1
                eachline = fhea.readline()
            fhea.close()
            data[i, j] = data_reshape
    return data


def load_mvc(path, subject_idx, session_idx, sig_type):
    if sig_type == 'force':
        M = 5
        N = 10 * 100 * 5
    else:
        M = 256
        N = 10 * 2048 * 256

    path = path + 'mvc_dataset'

    data = np.empty((10, 1), dtype=object)
    for i in range(10):
        for j in range(1):
            if i % 2 == 0:
                direction = 'flexion'
            else:
                direction = 'extension'
            path_file = path + '/subject' + subject_idx + '_session' + str(
                session_idx) + '/mvc_' + sig_type + '_finger' + str(
                int(np.ceil((i + 1) / 2))) + '_' + direction + '.dat'
            path_file_hea = path + '/subject' + subject_idx + '_session' + str(
                session_idx) + '/mvc_' + sig_type + '_finger' + str(
                int(np.ceil((i + 1) / 2))) + '_' + direction + '.hea'
            f = open(path_file, 'rb')
            parameter = '<' + str(N) + 'h'
            data_tmp = np.array(list(struct.unpack(parameter, f.read())))
            f.close()
            data_reshape_tmp = data_tmp.reshape(-1, M)
            data_reshape = data_reshape_tmp.astype('float64')

            fhea = open(path_file_hea, 'r')
            ch_idx = 0
            filename = 'mvc_' + sig_type + '_finger' + str(int(np.ceil((i + 1) / 2))) + '_' + direction + '.dat'
            eachline = fhea.readline()
            while eachline:
                if eachline.find(filename) > -1:
                    idx1 = eachline.find('.dat 16 ')
                    idx2 = eachline.find('(')
                    idx3 = eachline.find(')')
                    gain = float(eachline[idx1 + 8:idx2])
                    baseline = float(eachline[idx2 + 1:idx3])
                    tmp = (data_reshape[:, ch_idx] - baseline) / gain
                    data_reshape[:, ch_idx] = list(tmp.reshape(-1, 1))
                    ch_idx = ch_idx + 1
                eachline = fhea.readline()
            fhea.close()
            data[i, j] = data_reshape
    return data


def get_mvc(path, subject_idx, session_idx):
    force_data = load_mvc(path, subject_idx, session_idx, 'force')
    mvc = np.zeros((5, 2))
    for i in range(10):
        finger = int(np.floor(i / 2))
        direction = i - finger * 2
        force_tmp = np.abs(force_data[i, 0])
        force_tmp = force_tmp[:, finger]
        force_sort = sorted(force_tmp, reverse=True)
        mvc[finger, direction] = np.mean(force_sort[0:200])
    return mvc


def normalize_force(force, mvc):
    [row_num, column_num] = np.shape(force)
    force_norm = np.empty((row_num, column_num), dtype=object)
    for i in range(row_num):
        for j in range(column_num):
            force_tmp = force[i, j]
            force_norm_tmp = np.zeros(np.shape(force_tmp))
            for u in range(np.size(force_tmp, 0)):
                for v in range(np.size(force_tmp, 1)):
                    if force_tmp[u, v] < 0:
                        force_norm_tmp[u, v] = force_tmp[u, v] / mvc[v, 0]
                    else:
                        force_norm_tmp[u, v] = force_tmp[u, v] / mvc[v, 1]
            force_norm[i, j] = force_norm_tmp
    return force_norm


def preprocess_force(force, window_len, step_len, f_cutoff, fs_force, fs_emg):
    [row_num, column_num] = np.shape(force)
    force_preprocessed = np.empty((row_num, column_num), dtype=object)
    for i in range(row_num):
        for j in range(column_num):
            force_tmp = force[i, j]
            b, a = signal.butter(8, 10 / (fs_force / 2), 'lowpass')  # 10 Hz 8-order lowpass filter
            force_tmp_filter = signal.filtfilt(b, a, force_tmp, 0)
            force_tmp_filter_resample = signal.resample(force_tmp_filter,
                                                        int(np.size(force_tmp_filter, 0) / fs_force * fs_emg))
            [Nsample, Nchannel] = np.shape(force_tmp_filter_resample)
            window_sample = int(np.floor(window_len * fs_emg))
            step_sample = int(np.floor(step_len * fs_emg))
            force_preprocessed_tmp = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)),
                                               int(np.size(force_tmp_filter_resample, 1))))
            idx = 0
            for u in range(0, Nsample - window_sample + 1, step_sample):
                force_preprocessed_tmp[idx, :] = np.mean(force_tmp_filter_resample[u:u + window_sample, :], 0)
                idx = idx + 1
            force_preprocessed[i, j] = force_preprocessed_tmp
    return force_preprocessed

def my_mfl(sig):
    N = len(sig)
    Y = 0
    for i in range(N - 1):
        Y = Y + (sig[i + 1] - sig[i])**2
    mfl_value = np.log10(np.sqrt(Y))
    return mfl_value

def my_wa(sig):
    thres = 0.01
    N = len(sig)
    wa_value = 0
    for i in range(N - 1):
        if abs(sig[i] - sig[i+1]) > thres:
            wa_value = wa_value + 1
    return wa_value

def my_vare(sig):
    N = len(sig)
    vare_value = (1 / (N - 1)) * sum(sig * 2)
    return vare_value

def my_ssi(sig):
    ssi_value = sum(sig ** 2)
    return ssi_value

def my_myop(sig):
    thres = 0.016
    N = len(sig)
    Y = 0
    for i in range(N):
        if abs(sig[i]) >= thres:
            Y = Y + 1
    myop_value = Y / N
    return myop_value

def my_mmav2(sig):
    N = len(sig)
    Y = 0
    for i in range(N):
        if i >= 0.25 * N and i <= 0.75 * N:
            w = 1
        elif i < 0.25 * N:
            w = (4 * i) / N
        else:
            w = 4 * (i - N) / N
        Y = Y + (w * abs(sig[i]))
    mmav2_value = (1 / N) * Y
    return mmav2_value

def my_mmav(sig):
    N = len(sig)
    Y = 0
    for i in range(N):
        if i >= 0.25 * N and i <= 0.75 * N:
            w = 1
        else:
            w=0.5
        Y = Y + (w * abs(sig[i]))
    mmav_value = (1 / N) * Y
    return mmav_value

def my_ld(sig):
    N = len(sig)
    Y = 0
    for i in range(N):
        if sig[i]!=0:
            Y = Y + np.log(abs(sig[i]))
        else:
            continue
    ld_value = np.exp(Y / N)
    return ld_value

def my_dasdv(sig):
    N = len(sig)
    Y = 0
    for i in range(N-1):
        Y = Y + (sig[i+1] - sig[i])**2
    dasdv_value = np.sqrt(Y / (N - 1))
    return dasdv_value

def my_aac(sig):
    N = len(sig)
    Y = 0
    for i in range(N - 1):
        Y = Y + abs(sig[i+1] - sig[i])
    aac_value = Y / N
    return aac_value

def my_rms(sig):
    return math.sqrt(sum([x ** 2 for x in sig]) / len(sig))

def my_wl(sig, fs):
    N = len(sig)
    wl_value = 0
    for i in range(N - 1):
        wl_value = wl_value + abs(sig[i + 1] - sig[i])
    wl_value = wl_value / N * fs
    return wl_value


def my_zc(sig, thresh):
    N = len(sig)
    zc_value = 0
    for i in range(N - 1):
        if (abs(sig[i + 1] - sig[i]) > thresh) and (sig[i] * sig[i + 1] < 0):
            zc_value = zc_value + 1
    return zc_value


def my_ssc(sig, thresh):
    N = len(sig)
    ssc_value = 0
    for i in range(N - 1):
        if ((sig[i] - sig[i - 1]) * (sig[i] - sig[i + 1]) > 0) and (
                (abs(sig[i + 1] - sig[i]) > thresh) or (abs(sig[i - 1] - sig[i]) > thresh)):
            ssc_value = ssc_value + 1
    return ssc_value

def my_mav(sig):
    mav_value = np.mean(abs(sig))
    return mav_value

def my_iemg(sig):
    iemg_value = sum(abs(sig))
    return iemg_value

def my_ae(sig):
    ae_value = np.mean(sig**2)
    return ae_value

def my_var(sig):
    N = len(sig)
    mu = np.mean(sig)
    var_value = (1 / (N - 1)) * sum((sig - mu)**2)
    return var_value

def my_sd(sig):
    N = len(sig)
    mu = np.mean(sig)
    sd_value = np.sqrt((1 / (N - 1)) * sum((sig - mu)**2))
    return sd_value

def my_cov(sig):
    cov_value = np.std(sig) / np.mean(sig)
    return cov_value

def my_kurt(sig):
    kurt_value = kurtosis(sig)
    return kurt_value

def my_skew(sig):
    skew_value = skew(sig)
    return skew_value

def my_iqr(sig):
    iqr_value = iqr(sig)
    return iqr_value

def my_mad(sig):
    N = len(sig)
    mu = np.mean(sig)
    mad_value = (1 / N) * sum(abs(sig - mu))
    return mad_value

def my_ar(sig):
    order = 4

def my_damv(sig):
    N = len(sig)
    Y = 0
    for i in range(N - 1):
        Y = Y + abs(sig[i + 1] - sig[i])
    damv_value = Y / (N - 1)
    return damv_value

def my_tm(sig):
    order = 3
    N = len(sig)
    tm_value = abs((1 / N) * sum(sig**order))
    return tm_value

def my_vo(sig):
    order = 2
    N = len(sig)
    Y = (1 / N) * sum(sig ** order)
    vo_value = Y ** (1 / order)
    return vo_value

def my_dvarv(sig):
    N = len(sig)
    Y = 0
    for i in range(N - 1):
        Y = Y + (sig[i + 1] - sig[i])**2
    dvarv_value = Y / (N - 2)
    return dvarv_value

def my_ldamv(sig):
    N = len(sig)
    Y = 0
    for i in range(N - 1):
        Y = Y + abs((sig[i + 1] - sig[i]))
    ldamv_value = np.log(Y / N)
    return ldamv_value

def my_ldasdv(sig):
    N = len(sig)
    Y = 0
    for i in range(N - 1):
        Y = Y + (sig[i + 1] - sig[i]) ** 2
    ldasdv_value = np.log(np.sqrt(Y / (N - 1)))
    return ldasdv_value

def my_card(sig):
    thres = 0.01
    N = len(sig)
    Y = np.sort(sig)
    Z = np.zeros(N - 1)
    for i in range(N - 1):
        Z[i] = abs(Y[i] - Y[i + 1]) > thres
    card_value = np.sum(Z)
    return card_value

def my_lcov(sig):
    sig=abs(sig)
    mu = np.mean(sig)
    sd = np.std(sig)
    lcov_value = np.log(sd / mu)
    return lcov_value

def my_ltkeo(sig):
    N = len(sig)
    Y = 0
    for i in range(1,N - 1):
        Y = Y + ((sig[i] ** 2) - sig[i - 1] * sig[i + 1])
    ltkeo_value = np.log(Y)
    return ltkeo_value

def my_msr(sig):
    sig=abs(sig)
    N = len(sig)
    msr_value = (1 / N) * sum(sig ** (1 / 2))
    return msr_value

def my_ass(sig):
    sig=abs(sig)
    temp = sum(sig ** (1 / 2))
    ass_value = abs(temp)
    return ass_value

def my_asm(sig):
    sig=abs(sig)
    K = len(sig)
    Y = 0
    for n in range(K):
        if n >= 0.25 * K and n <= 0.75 * K:
            exp = 0.5
        else:
            exp = 0.75
        Y = Y + (sig[n] ** exp)
    asm_value = abs(Y / K)
    return asm_value

def my_fzc(sig):
    L = len(sig)
    fzc_value = 0
    T = 4 * ((1 / 10) * sum(sig[1 : 10]))
    for i in range(L-1):
        if (sig[i] > T and sig[i + 1] < T) or (sig[i] < T and sig[i + 1] > T):
            fzc_value = fzc_value + 1
    return fzc_value

def my_ewl(sig):
    L = len(sig)
    ewl_value = 0
    for i in range(1, L):
        if i >= 0.2 * L and i <= 0.8 * L:
            p = 0.75
        else:
            p = 0.5
        ewl_value = ewl_value + abs(abs(sig[i] - sig[i-1]) ** p)
    return ewl_value

def my_emav(sig):
    sig=abs(sig)
    L = len(sig)
    Y = 0
    for i in range(L):
        if i >= 0.2 * L and i <= 0.8 * L:
            p = 0.75
        else:
            p = 0.5
        Y = Y + abs(sig[i] ** p)
    emav_value = Y / L
    return emav_value

def my_TM(sig,order):
    L = len(sig)
    sig = sig*100
    TMfeature=abs(1/L*sum(sig**order))
    return TMfeature

def my_SM(sig,fs,order):
    pxx, f = welch(sig, fs, len(sig))

    SM=sum(pxx*(f**order))
    return SM

def my_VCF(sig,fs):
    SM0 = my_SM(sig, fs, 0)
    SM1 = my_SM(sig, fs, 1)
    SM2 = my_SM(sig, fs, 2)
    VCF=SM2/SM0-(SM1/SM0)**2
    return VCF

def my_Hjorth2(sig):
    sig_diff1=np.diff[sig]
    Hjorth2feature = np.std[sig_diff1] / np.std[sig]
    return Hjorth2feature

def my_FR(sig,fs,f1,fh):
    pxx, f = welch(sig, fs, len(sig))
    fl_idx1 = np.argmin(abs(f-f1[1]))
    fl_idx2 = np.argmin(abs(f-f1[2]))
    fh_idx1 = np.argmin(abs(f-fh[1]))
    fh_idx2 = np.argmin(abs(f-fh[2]))
    FR = sum(pxx[fl_idx1:fl_idx2]) / sum(pxx[fh_idx1: fh_idx2])
    return FR

def get_mfl(emg,window_len,step_len,fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    mfl =np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            mfl[fea_idx, j] = my_mfl(emg_window)
        fea_idx = fea_idx + 1
    return mfl

def get_wa(emg,window_len,step_len,fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    wa = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            wa[fea_idx, j] = my_wa(emg_window)
        fea_idx = fea_idx + 1
    return wa

def get_vare(emg,window_len,step_len,fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    vare = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            vare[fea_idx, j] = my_vare(emg_window)
        fea_idx = fea_idx + 1
    return vare

def get_ssi(emg,window_len,step_len,fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    ssi = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            ssi[fea_idx, j] = my_ssi(emg_window)
        fea_idx = fea_idx + 1
    return ssi

def get_myop(emg,window_len,step_len,fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    myop = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            myop[fea_idx, j] = my_myop(emg_window)
        fea_idx = fea_idx + 1
    return myop

def get_mmav2(emg,window_len,step_len,fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    mmav2 = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            mmav2[fea_idx, j] = my_mmav2(emg_window)
        fea_idx = fea_idx + 1
    return mmav2

def get_mmav(emg,window_len,step_len,fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    mmav = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            mmav[fea_idx, j] = my_mmav(emg_window)
        fea_idx = fea_idx + 1
    return mmav

def get_ld(emg,window_len,step_len,fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    ld = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            ld[fea_idx, j] = my_ld(emg_window)
        fea_idx = fea_idx + 1
    return ld

def get_dasdv(emg,window_len,step_len,fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    dasdv = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            dasdv[fea_idx, j] = my_dasdv(emg_window)
        fea_idx = fea_idx + 1
    return dasdv

def get_aac(emg,window_len,step_len,fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    aac = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            aac[fea_idx, j] = my_aac(emg_window)
        fea_idx = fea_idx + 1
    return aac

def get_rms(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    rms = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            rms[fea_idx, j] = my_rms(emg_window)
        fea_idx = fea_idx + 1
    return rms


def get_wl(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    wl = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            wl[fea_idx, j] = my_wl(emg_window, fs)
        fea_idx = fea_idx + 1
    return wl


def get_zc(emg, window_len, step_len, thresh, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    zc = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            zc[fea_idx, j] = my_zc(emg_window, thresh)
        fea_idx = fea_idx + 1
    return zc


def get_ssc(emg, window_len, step_len, thresh, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    ssc = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            ssc[fea_idx, j] = my_ssc(emg_window, thresh)
        fea_idx = fea_idx + 1
    return ssc

def get_mav(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    mav = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            mav[fea_idx, j] = my_mav(emg_window)
        fea_idx = fea_idx + 1
    return mav

def get_iemg(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    iemg = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            iemg[fea_idx, j] = my_iemg(emg_window)
        fea_idx = fea_idx + 1
    return iemg

def get_ae(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    ae = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            ae[fea_idx, j] = my_ae(emg_window)
        fea_idx = fea_idx + 1
    return ae

def get_var(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    var = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            var[fea_idx, j] = my_var(emg_window)
        fea_idx = fea_idx + 1
    return var

def get_sd(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    sd = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            sd[fea_idx, j] = my_sd(emg_window)
        fea_idx = fea_idx + 1
    return sd

def get_cov(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    cov = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            cov[fea_idx, j] = my_cov(emg_window)
        fea_idx = fea_idx + 1
    return cov

def get_kurt(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    kurt = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            kurt[fea_idx, j] = my_kurt(emg_window)
        fea_idx = fea_idx + 1
    return kurt

def get_skew(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    skew = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            skew[fea_idx, j] = my_skew(emg_window)
        fea_idx = fea_idx + 1
    return skew

def get_iqr(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    iqr = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            iqr[fea_idx, j] = my_iqr(emg_window)
        fea_idx = fea_idx + 1
    return iqr

def get_mad(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    mad = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            mad[fea_idx, j] = my_mad(emg_window)
        fea_idx = fea_idx + 1
    return mad

def get_ar(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    ar = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            ar[fea_idx, j] = my_ar(emg_window)
        fea_idx = fea_idx + 1
    return ar

def get_damv(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    damv = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            damv[fea_idx, j] = my_damv(emg_window)
        fea_idx = fea_idx + 1
    return damv

def get_tm(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    tm = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            tm[fea_idx, j] = my_tm(emg_window)
        fea_idx = fea_idx + 1
    return tm

def get_vo(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    vo = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            vo[fea_idx, j] = my_vo(emg_window)
        fea_idx = fea_idx + 1
    return vo

def get_dvarv(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    dvarv = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            dvarv[fea_idx, j] = my_dvarv(emg_window)
        fea_idx = fea_idx + 1
    return dvarv

def get_ldamv(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    ldamv = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            ldamv[fea_idx, j] = my_ldamv(emg_window)
        fea_idx = fea_idx + 1
    return ldamv

def get_ldasdv(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    ldasdv = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            ldasdv[fea_idx, j] = my_ldasdv(emg_window)
        fea_idx = fea_idx + 1
    return ldasdv

def get_card(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    card = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            card[fea_idx, j] = my_card(emg_window)
        fea_idx = fea_idx + 1
    return card

def get_lcov(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    lcov = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            lcov[fea_idx, j] = my_lcov(emg_window)
        fea_idx = fea_idx + 1
    return lcov

def get_ltkeo(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    ltkeo = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            ltkeo[fea_idx, j] = my_ltkeo(emg_window)
        fea_idx = fea_idx + 1
    return ltkeo

def get_msr(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    msr = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            msr[fea_idx, j] = my_msr(emg_window)
        fea_idx = fea_idx + 1
    return msr

def get_ass(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    ass = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            ass[fea_idx, j] = my_ass(emg_window)
        fea_idx = fea_idx + 1
    return ass

def get_asm(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    asm = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            asm[fea_idx, j] = my_asm(emg_window)
        fea_idx = fea_idx + 1
    return asm

def get_fzc(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    fzc = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            fzc[fea_idx, j] = my_fzc(emg_window)
        fea_idx = fea_idx + 1
    return fzc

def get_ewl(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    ewl = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            ewl[fea_idx, j] = my_ewl(emg_window)
        fea_idx = fea_idx + 1
    return ewl

def get_emav(emg, window_len, step_len, fs):
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    [Nsample, Nchannel] = np.shape(emg)
    emav = np.zeros((int(np.ceil((Nsample - window_sample + 1) / step_sample)), Nchannel))
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            emav[fea_idx, j] = my_emav(emg_window)
        fea_idx = fea_idx + 1
    return emav

def load_feature(path, subject_idx, session_idx, task_type, feature_type, process_type, reshape_option):
    if task_type == '1dof':
        mid_name = 'finger'
        mid_max_idx = 5
        sample_max_idx = 3
    else:
        mid_name = 'combination'
        mid_max_idx = 15
        sample_max_idx = 2
    data = np.empty((mid_max_idx, sample_max_idx), dtype=object)
    if reshape_option == 1:
        layout_array = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                layout_array[i, j] = 64 - i * 8 - j - 1
        layout = np.append(np.append(layout_array, layout_array + 64, axis=0), np.append(layout_array + 128, layout_array + 192, axis=0), axis=1)
        for i in range(mid_max_idx):
            for j in range(sample_max_idx):
                path_file = path + task_type + '_dataset/' + feature_type + '_' + process_type + '/subject' + subject_idx + '_session' + str(session_idx) + '/' + feature_type + '_' + mid_name + str(i + 1) + '_sample' + str(j + 1) + '.npy'
                tmp = np.load(path_file)
                data_tmp = np.zeros((np.size(tmp, 0), 16, 16))
                for u in range(16):
                    for v in range(16):
                        data_tmp[:, u, v] = tmp[:, int(layout[u, v])]
                data[i, j] = data_tmp
    else:
        for i in range(mid_max_idx):
            for j in range(sample_max_idx):
                path_file = path + task_type + '_dataset/' + feature_type + '_' + process_type + '/subject' + subject_idx + '_session' + str(session_idx) + '/' + feature_type + '_' + mid_name + str(i + 1) + '_sample' + str(j + 1) + '.npy'
                tmp = np.load(path_file)
                data[i, j] = tmp
    return data

def load_pca_feature(path, subject_idx, session_idx, task_type, feature_type, process_type):
    if task_type == '1dof':
        mid_name = 'finger'
        mid_max_idx = 5
        sample_max_idx = 3
    else:
        mid_name = 'combination'
        mid_max_idx = 15
        sample_max_idx = 2
    data = np.empty((mid_max_idx, sample_max_idx), dtype=object)

    for i in range(mid_max_idx):
        for j in range(sample_max_idx):
            path_file = path + task_type + '_dataset/' + feature_type + process_type + '/subject' + subject_idx + '_session' + str(session_idx) + '/' + feature_type + '_' + mid_name + str(i + 1) + '_sample' + str(j + 1) + '.npy'
            tmp = np.load(path_file)
            data[i, j] = tmp
    return data




