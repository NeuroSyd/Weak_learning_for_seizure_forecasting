import glob
import os
import mne
import matplotlib
import pandas as pd
import numpy as np
import stft
from scipy.signal import resample
from datetime import datetime, timedelta
import pickle

# matplotlib.use('TkAgg')
#%matplotlib
from threading import Thread
from multiprocessing.pool import ThreadPool
from utils.filters import butter_bandpass_filter, butter_highpass_filter, notch_filter, notch_filter_multi_harmonics
from utils.parser import sdy_extract, events_extract,seizure_table_extract
from utils.eeg_conversion import convert_AP_montage, create_mne_raw
from utils.preprocessing import detect_interupted_data, ica_arti_remove
from utils.pyst import read_edf,read_edf_elec



def resample_data(data, old_sample_rate,new_sample_rate):
    if new_sample_rate is not None:
        if new_sample_rate < old_sample_rate:
            print('Downsampling from {} Hz to {} Hz...'.format(old_sample_rate, new_sample_rate))
            num_samples = int(data.shape[1] / old_sample_rate * new_sample_rate)
            data = resample(data, num=num_samples, axis=1)
            final_sample_rate = new_sample_rate
        else:
            final_sample_rate = old_sample_rate
    else:
        final_sample_rate = old_sample_rate
    return data


def load_edf(fn):
    print ('EDF format')    
    raw = mne.io.read_raw_edf(fn, preload=True)    
    if 'T3(T7)' in raw.ch_names:
        with open('utils/10_20.txt', 'r') as f:
            chs = f.readlines()
    elif 'T3' in raw.ch_names:
        with open('utils/10_20_new.txt', 'r') as f:
            chs = f.readlines()
    else:
        with open('utils/10_20_old.txt', 'r') as f:
            chs = f.readlines()
    chs = [ch.strip() for ch in chs]        
    
    print (len(chs))
    print (chs)
    # load_data 
    raw_1020 = raw.pick_channels(chs, ordered=True)
    print (raw.ch_names)
    print (raw_1020.ch_names)
    print (len(raw_1020.ch_names))
    
    assert len(raw_1020.ch_names) == len(chs)
    del raw

    # raw_1020.plot(block=True, scalings=50e-6, remove_dc=True, lowpass=70, title='Checking raw')
    # raw_1020 = raw_1020.crop(tmin=59500, tmax=60500)
    
    data = raw_1020.to_data_frame()

    del raw_1020

    data = data.to_numpy()
    print (data.shape)
    # excerpt = data[-80000:-70000]
    # np.savetxt('excerpt.csv', excerpt, delimiter=',')
    
    return data[:,1:], chs


def load_csv(fn):
    print ('CSV format')
    with open('10_20.txt', 'r') as f:
        chs = f.readlines()
    chs = [ch.strip() for ch in chs]
    # with open('/home/nhantruong/0_ongoing/seizure-prediction/SZPred_biomarker_RPA/sample_csv.csv', 'r') as f:
    #     for i in range(10):
    #         line = f.readline()
    #         print (line.strip())
    
   
    # df = pd.read_csv('/home/nhantruong/0_ongoing/seizure-prediction/SZPred_biomarker_RPA/sample_csv_100k.csv', skiprows=1, encoding='utf-16-le', usecols=chs)   
    df = pd.read_csv(fn, skiprows=1, encoding='utf-16-le', usecols=chs) 
    df = df[chs] # to ensure the electrode order is correct
    print (df.head())
    print (df.columns)

    data = df.values
    print (data.shape)
       

    return data, chs

    # chunksize = 10 ** 6
    # for chunk in pd.read_csv(fn, chunksize=chunksize, skiprows=1, encoding='utf-16-le', usecols=chs):
    #     print (chunk.head())
    
    

def load_data(study_load_dir,ref_path,cache_dir):
    # for fn_study in study_list:
    print('--------------------------------------')
    fn_study = study_load_dir
    fn_eeg = locate_eeg_data(fn_study)
    print (fn_eeg)
    gap_time, creation_time, recording_start_time, recording_duration, eeg_sample_rate = sdy_extract(fn_study)
    print (gap_time, creation_time, recording_start_time, recording_duration, eeg_sample_rate)

    fn_study_name = fn_study.split('/')[-1]

    print(fn_study_name)
    data, chs = extract_signal(fn_eeg)
    print(data.shape)
    print('--------------------------------------')


    total_duration = int(data.shape[1])/float(eeg_sample_rate) - 1
    print(total_duration)

    return data,total_duration,eeg_sample_rate,chs,fn_study_name

    # sz_table = seizure_table_extract(fn_study_name, ref_path)
    # sz_table = events_extract(fn_study)

    # if sz_table.shape[0] > 0:
    #     extract_preictal(fn_eeg, sz_table, creation_time, int(eeg_sample_rate),cache_dir,new_sample_rate=256)
    # extract_interictal(fn_eeg, sz_table, creation_time, int(eeg_sample_rate),cache_dir,new_sample_rate=256)


def dc_notch_filter(s, chs, final_sample_rate):

    s = mne.filter.filter_data(s, sfreq=final_sample_rate, l_freq=0.1, h_freq=None, verbose=False)
    
    nyq_freq = (final_sample_rate-1)/2
    notch_upper = (nyq_freq - nyq_freq%50) + 1
    print ('notch_upper', notch_upper)
    s = mne.filter.notch_filter(s, Fs=final_sample_rate, freqs=np.arange(50,notch_upper,50), verbose=False)
    # s = s.transpose() 
    return s, chs


def set_montage_filter(s, chs, final_sample_rate):
    s_ap, chs_ap = convert_AP_montage(s, chs)
    chs_ap = [ch for ch in chs_ap]

    s_ap = mne.filter.filter_data(s_ap.transpose(), sfreq=final_sample_rate, l_freq=0.1, h_freq=None, verbose=False)
    
    nyq_freq = (final_sample_rate-1)/2
    notch_upper = (nyq_freq - nyq_freq%50) + 1
    print ('notch_upper', notch_upper)
    s_ap = mne.filter.notch_filter(s_ap, Fs=final_sample_rate, freqs=np.arange(50,notch_upper,50), verbose=False)
    s_ap = s_ap.transpose() 
    return s_ap, chs_ap

def extract_signal(fn_eeg):
    data = None
    if fn_eeg.split('.')[-1] == 'edf':
        fsamp, data = read_edf_elec(fn_eeg, parameters="utils/params_RPA_common_electrodes.txt")
        chs = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
        # chs_1 = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3(T7)', 'T4(T8)', 'T5(P7)', 'T6(P8)', 'Fz', 'Cz', 'Pz']
        # data, chs = load_edf(fn_eeg)
    elif fn_eeg.split('.')[-1] == 'csv':
        data, chs = load_csv(fn_eeg)
    else:
        print('Unknown format')

    assert data is not None
    return data, chs







def extract_interictal(data,sess_num, sz_table, cache_dir, new_sample_rate=256):
    
    
    assert data is not None
    assert data.shape == (3600*4*256,19)
    
    if new_sample_rate is not None:
        final_sample_rate = new_sample_rate


    numts = 30
    window_len = int(final_sample_rate * numts)
    step = int(window_len / 2)

    print('Checking sz table...')
    nsz = sz_table.shape[0]
    interictal_all = []
    start_time_all = []

    X = []
    t_stamps = []
    chs = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    if nsz == 0:  # there is no sz in the study
        print(data.shape)
        s = data[60*60 * final_sample_rate:-60*60 * final_sample_rate, :]  # skip first and last 60 minutes signals
        print('Consider signals in the whole study as interictal')
        s_ap, chs_ap = dc_notch_filter(s.transpose(), chs, final_sample_rate)
        s_ap = s_ap.transpose() 
        interictal_all.append(s_ap)
        # start_time_all.append(creation_time + timedelta(seconds=30))

    interictal_ind = 0

    for ind, row in sz_table.iterrows():
        print('SEIZURE {}:'.format(ind + 1), row['onset'], row['end'])

        fmt = '%d/%m/%Y %I:%M:%S %p'  # 26/07/2011 01:54:05 AM
        # onset_datetime = datetime.strptime(row['onset_datetime'], fmt)
        # end_datetime = datetime.strptime(row['end_datetime'], fmt)
        #
        # i = (onset_datetime - creation_time).total_seconds()
        # j = (end_datetime - creation_time).total_seconds()
        i = float(row['onset'])
        j = float(row['end'])

        print('Seizure', row['onset'], i, row['end'], j, int(i * final_sample_rate))

        # Here we define interictal period as at least 1.0 hour before sz onset and 1.0 hour after sz end
        if ind == 0:  # first sz in the table
            print('first sz in the table')
            interictal_sp = i - 1.5 * 60 * 60  # 1.5h before sz onset
            if interictal_sp >= 30:
                s = data[30 * final_sample_rate:int(interictal_sp * final_sample_rate),
                    :]  # from begining of the file to 1h before sz onset
                print('Consider signals from beginning of the file to {} as interictal'.format(interictal_sp))
                s_ap, chs_ap = dc_notch_filter(s.transpose(), chs, final_sample_rate)
                s_ap = s_ap.transpose() 
                interictal_all.append(s_ap)
                # start_time_all.append(creation_time + timedelta(seconds=30))

        interictal_st = j + 1.5 * 60 * 60  # 1.5 hour after sz end

        if ind < nsz - 1:  # there is another seizure after this one
            print('there is another seizure after this one at', sz_table.iloc[ind + 1]['onset'])
            # next_onset = datetime.strptime(sz_table.iloc[ind + 1]['onset_datetime'], fmt)
            # ii = (next_onset - creation_time).total_seconds()  # onset of the next sz

            ii = float(sz_table.iloc[ind + 1]['onset'])
            interictal_sp = ii - 1.5 * 60 * 60  # 1.5h before sz onset
            if interictal_sp >= interictal_st:
                s = data[int(interictal_st * final_sample_rate):int(interictal_sp * final_sample_rate), :]
                print('Consider signals from {} to {} as interictal'.format(interictal_st, interictal_sp))
                s_ap, chs_ap = dc_notch_filter(s.transpose(), chs, final_sample_rate)
                s_ap = s_ap.transpose() 
                interictal_all.append(s_ap)
                # start_time_all.append(creation_time + timedelta(seconds=interictal_st))
        elif ind == nsz - 1:  # this is the last sz
            print('this is the last sz')
            print(data.shape)
            s = data[int(interictal_st * final_sample_rate):, :]  # from 1.5 hour after sz end to end of the file
            print('Consider signals after {} as interictal'.format(interictal_st))
            print(s.shape)
            s_ap, chs_ap = dc_notch_filter(s.transpose(), chs, final_sample_rate)
            s_ap = s_ap.transpose() 
            interictal_all.append(s_ap)
            # start_time_all.append(creation_time + timedelta(seconds=interictal_st))

    # assert len(interictal_all) == len(start_time_all)
    for i_iterictal in range(len(interictal_all)):
        interictal_s = interictal_all[i_iterictal]
        i_s = 0
        while i_s * step + window_len < interictal_s.shape[0]:
            seg = interictal_s[i_s * step:i_s * step + window_len]
            # time_s = start_time_all[i_iterictal] + timedelta(seconds=i_s * step / final_sample_rate)
            i_s += 1
            # detect if signal is interupted, e.g., all dc, overflow
            if detect_interupted_data(seg, final_sample_rate):
                print('BAD DATA DETECTED! Skipping this {}-second segment due to interupted signals...'.format(numts))
                continue
            else:
                print('GOOD DATA!')

            #####ICA part#######################################

            # ica_filt_s = ica_arti_remove(seg.transpose(), final_sample_rate, chs_ap)
            #
            # if ica_filt_s is None:
            #     print ('Skipping this {}-second segment due to failed ICA...'.format(numts))
            #     continue

            #####ICA part#######################################

            # np.save('/mnt/data5_2T/tempdata/RPA_testing/interictal_{}.npy'.format(interictal_ind), ica_filt_s)
            # interictal_ind += 1

            X.append(seg)
            # print (time_s)
            # t_stamps.append(time_s)

            # s_ap, chs_ap_ = convert_AP_montage(seg, chs_ap)
            # chs_ap_ = [ch for ch in chs_ap_]
            # raw = create_mne_raw(s_ap.transpose(), final_sample_rate, chs_ap_)
            # raw.plot(block=True, scalings=50e-6, remove_dc=True, lowpass=70, title='Raw - 0.5-70 Hz')

            # s_ap, _ = convert_AP_montage(ica_filt_s, chs_ap)
            # # chs_ap = [ch for ch in chs_ap]
            # raw_ica = create_mne_raw(s_ap.transpose(), final_sample_rate, chs_ap_)
            # raw_ica.plot(block=True, scalings=50e-6, remove_dc=True, lowpass=70, title='ICA-denoised - 0.5-70')

    
    fn = '{}/interictal_{}.pickle'.format(cache_dir, sess_num)
    # fnt = '{}/interictal_{}_time.pickle'.format(cache_dir, fn_edf.split('.')[0])
    with open(fn, 'wb') as f:
        pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
    # with open(fnt, 'wb') as f:
    #     pickle.dump(t_stamps, f, pickle.HIGHEST_PROTOCOL)


def extract_preictal(data,sess_num, sz_table, cache_dir, new_sample_rate=256):
    '''
    This code does not consider the case that if the sz occurs at beginning of the file, preictal
    should be extracted from the previous recording.
    '''

    
    # assert data is not None
    assert data.shape == (3600*4*256,19)
    final_sample_rate = new_sample_rate
    numts = 30
    window_len = int(final_sample_rate * numts)
    step = int(window_len / 10)

    preictal_ind = 0
    X = []
    t_stamps = []
    chs = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    for ind, row in sz_table.iterrows():
        print('SEIZURE {}:'.format(ind + 1), row['onset'], row['end'] )

        fmt = '%d/%m/%Y %I:%M:%S %p'  # 26/07/2011 01:54:05 AM
        # onset_datetime = datetime.strptime(row['onset_datetime'], fmt)
        # end_datetime = datetime.strptime(row['end_datetime'], fmt)
        #
        # i = (onset_datetime - creation_time).total_seconds()
        # j = (end_datetime - creation_time).total_seconds()
        i = float(row['onset'])
        j = float(row['end'])

        print('Seizure', row['onset'], i, row['end'], j, int(i * final_sample_rate))

        # Here we define preictal period as 35 min - 5 min before seizure onset
        preictal_st = i - 35 * 60
        preictal_sp = i - 5 * 60

        preictal_st = max(30, preictal_st)

        # s_time = begin_rec + timedelta(seconds=int(i*numts))
        if sz_table.shape[0] > 1 and ind > 0:  # check previous sz end time
            # prv_end_datetime = datetime.strptime(sz_table.iloc[ind - 1]['end_datetime'], fmt)
            # prv_j = (prv_end_datetime - creation_time).total_seconds()
            prv_j = float(sz_table.iloc[ind - 1]['end'])
            preictal_st = max(preictal_st, prv_j)

        s = data[int(preictal_st * final_sample_rate):int(preictal_sp * final_sample_rate), :]
        print('preictal_st', preictal_st)
        # start_time = creation_time + timedelta(seconds=preictal_st)
        print('preictal shape', s.shape)

        s_ap, chs_ap = dc_notch_filter(s.transpose(), chs, final_sample_rate)
        s_ap = s_ap.transpose() 
        
        i_s = 0
        while i_s * step + window_len < s_ap.shape[0]:
            seg = s_ap[i_s * step:i_s * step + window_len]
            # time_s = start_time + timedelta(seconds=i_s * step / final_sample_rate)
            i_s += 1

            # detect if signal is interupted, e.g., all dc, overflow
            if detect_interupted_data(seg, final_sample_rate):
                print('BAD DATA DETECTED! Skipping this {}-second segment due to interupted signals...'.format(numts))
                continue
            else:
                print('GOOD DATA!')

            #####ICA part#######################################
            # ica_filt_s = ica_arti_remove(seg.transpose(), final_sample_rate, chs_ap)
            #
            # if ica_filt_s is None:
            #     print ('Skipping this {}-second segment due to failed ICA...'.format(numts))
            #     continue
            #####ICA part#######################################

            # np.save('/mnt/data5_2T/tempdata/RPA_testing/interictal_{}.npy'.format(preictal_ind), ica_filt_s)
            # preictal_ind += 1

            X.append(seg)
            # t_stamps.append(time_s)

            # s_ap, chs_ap_ = convert_AP_montage(seg, chs_ap)
            # chs_ap_ = [ch for ch in chs_ap_]
            # raw = create_mne_raw(s_ap.transpose(), final_sample_rate, chs_ap_)
            # raw.plot(block=True, scalings=50e-6, remove_dc=True, lowpass=70, title='Raw - 0.1-70 Hz')

            # s_ap, _ = convert_AP_montage(ica_filt_s, chs_ap)
            # # chs_ap = [ch for ch in chs_ap]
            # raw_ica = create_mne_raw(s_ap.transpose(), final_sample_rate, chs_ap_)
            # raw_ica.plot(block=True, scalings=50e-6, remove_dc=True, lowpass=70, title='ICA-denoised - 0.1-70')

    # if len(X) > 0:
        
    fn = '{}/preictal_{}.pickle'.format(cache_dir, sess_num)
    # fnt = '{}/preictal_{}_time.pickle'.format(cache_dir, fn_edf.split('.')[0])
    with open(fn, 'wb') as f:
        pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
    # with open(fnt, 'wb') as f:
    #     pickle.dump(t_stamps, f, pickle.HIGHEST_PROTOCOL)


def locate_eeg_data(fn_study):
    study_name = fn_study.split('/')[-1]
    # print ('study_name', study_name)
    fn_eeg = glob.glob('{}/EEGData/Temp/{}*'.format(fn_study, study_name))
    print('----yikai-------')
    print(fn_eeg)
    print('----yikai-------')
    if len(fn_eeg) > 0:
        for dir in fn_eeg:
            if '.edf' in dir:
                return dir
        return fn_eeg[0]
    else:
        return None


if __name__ == '__main__':
    import json

    year = '2013'
    # pat = '1'
    pats = [
        # '1',
        # '2',
        # '4',
        # '5',
        # '6',
        # '7',
        '8',
        # '9',
        # '10',
        # '11'
    ]
    for pat in pats:
        cache_dir = '/mnt/data7_M2/tempdata/RPA_pred_detect/true_label/'+year+'_'+pat
        os.makedirs(cache_dir, exist_ok=True)
        with open('../json_file/SETTINGS_'+year+'.json') as f:
            settings = json.load(f)
        print (settings['datadir'])

        study_pd = pd.read_csv('event_list/'+year+'_st_list.csv')

        study_pd = study_pd[(study_pd['pat_num'] == int(pat)) & (study_pd['anno_quality'] == 'Y')]
        print(study_pd)
        study_list = study_pd['file_name'].values

        print(study_list)

        study_list = [os.path.join(settings['datadir'], study) for study in study_list]
        ref_path = '/home/yikai/code/prediction_detection/ref/ref_' + year + '/' + year + '_pat' + pat + '.txt'
        data,total_duration,eeg_sample_rate,chs = load_data(study_list, ref_path ,cache_dir)
