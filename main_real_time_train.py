import os
import pandas as pd
import numpy as np
import pickle
import stft
from sklearn.model_selection import train_test_split
from utils.data_gen_new import DataGenerator
from models.convlstm import ConvNN
from utils.load_pred_data import load_pred_data

def stft_cal(X_test, type='interictal'):
    y_test = []
    X_test_new = []
    for s in X_test:
        stft_data = stft.spectrogram(s, framelength=256, centered=False)
        stft_data = np.transpose(stft_data, (2, 1, 0))
        stft_data = np.abs(stft_data) + 1e-6
        stft_data = stft_data[:, 1:57, 1:]
        stft_data = np.log10(stft_data)
        indices = np.where(stft_data <= 0)
        stft_data[indices] = 0
        stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                      stft_data.shape[1],
                                      stft_data.shape[2])
        X_test_new.append(stft_data)

        if type == 'interictal':
            y_test.append(0)
        elif type == 'preictal':
            y_test.append(1)
        else:
            raise Exception("Sorry, type is neither interictal or preictal")
    return X_test_new, y_test


def preprocess(data_dir,year,pat,test_list):
    
    temp_dir = data_dir

    X_test_interictal = []
    X_test_preictal = []
    for study in test_list:
        fn = temp_dir + '/interictal_' + study.split('.')[0] + '.pickle'
        print(fn)
        if os.path.exists(fn):
            with open(fn, 'rb') as f:
                x = pickle.load(f)
                try:
                    print(len(x), x[0].shape)
                except:
                    print('empty inside')
                X_test_interictal += x

        fn = temp_dir + '/preictal_' + study.split('.')[0] + '.pickle'
        print(fn)
        if os.path.exists(fn):
            with open(fn, 'rb') as f:
                x = pickle.load(f)
                try:
                    print(len(x), x[0].shape)
                except:
                    print('empty inside')
                X_test_preictal += x

    if args.mode == 'train':
        # calculate the class_weights_train
        total_sample = len(X_test_interictal)+len(X_test_preictal)
        class_weights = {}
        class_weights[0] = total_sample / (len(X_test_interictal) + 1e-6)
        class_weights[1] = total_sample / (len(X_test_preictal) + 1e-6)
        # print('----------------------down sampling-----------------------------')
        # '''
        # Downsampling interictal training set so that the 2 classes
        # are balanced
        # '''
        # if len(X_test_preictal)!=0:
        #     down_spl = int(np.floor(len(X_test_interictal) / len(X_test_preictal)))
        #     print('down sampling rate:', down_spl)
        #     if down_spl > 1:
        #         X_test_interictal = X_test_interictal[::down_spl]
        #     elif down_spl == 1:
        #         X_test_interictal = X_test_interictal[:len(X_test_preictal)]


    if len(X_test_interictal) ==0 and len(X_test_preictal)==0:
        print('no useful information yet')

    if len(X_test_preictal)>0:
        X_test_preictal_stft, y_test_preictal = stft_cal(X_test_preictal, type='preictal')
        X_test_preictal_stft = np.concatenate(X_test_preictal_stft, axis=0)
        y_test_preictal = np.array(y_test_preictal)
        print(X_test_preictal_stft.shape, y_test_preictal.shape)

    if len(X_test_interictal)>0:
        X_test_interictal_stft, y_test_interictal = stft_cal(X_test_interictal, type='interictal')
        X_test_interictal_stft = np.concatenate(X_test_interictal_stft, axis=0)
        y_test_interictal = np.array(y_test_interictal)
        print(X_test_interictal_stft.shape, y_test_interictal.shape)

    if len(X_test_preictal)>0 and len(X_test_interictal)>0:
        X_test = np.concatenate((X_test_preictal_stft, X_test_interictal_stft), axis=0)
        y_test = np.concatenate((y_test_preictal, y_test_interictal), axis=0)
    
    if len(X_test_preictal) ==0 and len(X_test_interictal)>0:
        print('only interiactal yet')
        X_test = X_test_interictal_stft
        y_test = y_test_interictal

    if len(X_test_interictal) ==0 and len(X_test_preictal)>0:
        print('only preictal yet')
        X_test = X_test_preictal_stft
        y_test = y_test_preictal

    X_test = np.transpose(X_test, (0, 2, 1, 3))

    X_test = X_test[..., np.newaxis]

    return X_test, y_test,class_weights


def RPA_train(X_train, y_train,savepath,class_weights_train,batch_size,is_first_time):

    params = {'dim': (56, 19, 128, 1),
              'batch_size': batch_size,
              'n_classes': 2,
              'n_channels': 19,
              'shuffle': True}

    X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42)

    training_generator = DataGenerator(X_train_new, y_train_new, **params)
    validation_generator = DataGenerator(X_val, y_val, **params)

    model = ConvNN(epochs=20,batch_size=batch_size)
    model.setup(X_train.shape)

    is_load = False
    while is_load != True:
        print('Begin load')
        if is_first_time:
            try:
                model.load_trained_weights('models/pre_trained_model/predict/weights_clstm_goodpat.h5')
                print('Load successful the first time model')
                is_load =True
            except:
                pass
        else:
            try:
                model.load_trained_weights(savepath)
                print('Load successful the new model')
                is_load =True
            except:
                pass
    #model.fit(X_train, y_train)
    model.fit_generator(training_generator, validation_generator, savepath,class_weights_train)
    print('finish training and begin evaluate ----------------------------')
    print('--------------finish training!----------------')














if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="RPA",
                        help="FB, EpilepsiaSurf, CHBMIT or Kaggle2014Det")
    parser.add_argument("--mode", default="train",
                        help="online training")
    parser.add_argument("--sph", type=int, default=5,
                        help="seizure prediction horizon in seconds")

    args = parser.parse_args()
    assert args.dataset in ["RPA", "FB", "CHBMIT", "Kaggle2014Det", "EpilepsiaSurf"]
    assert args.mode in ['cv', 'train', 'test']

    year = '2013'
    pats = [
        # '1',
        #  '2',
        #  '4',
        # '5',
        # '6',
        # '7',
        # '8',
        '9',
        # '10',
        # '11',
        # '12',
        # '21',
        # '22'
    ]
    for pat in pats:
        cache_dir = '/raid/neurosyd/RPA_online_train_buffer/real_time_data/' + year + '_' + pat
        os.makedirs(cache_dir, exist_ok=True)

        study_pd = pd.read_csv('utils/event_list/' + year + '_st_list.csv')
        study_pd = study_pd[(study_pd['pat_num'] == int(pat)) & (study_pd['anno_quality'] == 'Y')]
        study_list = study_pd['file_name'].values
        print(study_list)
        #need to change to automtical next step 
        # sess_num = fn_study = study_list[2].split('.')[0] 
        
        while True:
            try:
                with open ('FLAG_'+year+'_'+pat+'.txt', 'r') as f:
                    info = f.readlines()[0].split('_')
                    sess_num = fn_study = info[0]
                    flag = info[1]
                    detect_res_idx = info[-1]
                if flag == 'True': 
                    print('read detection result index:   ', str(detect_res_idx))
                    load_pred_data(sess_num, year, pat, cache_dir,detect_res_idx)
                    #load_data
                    assert args.mode == 'train'
                    data_dir = cache_dir
                    test_li = list(study_list)

                    #only test one now
                    test_li = [sess_num]
                    print(test_li)
                    X_train, y_train,class_weights_train = preprocess(data_dir,year,pat,test_li)
                    if y_train.shape[0] <500:
                        batch_size = 5
                    elif y_train.shape[0]>= 500:
                        batch_size = 10
                    elif y_train.shape[0]>= 1000:
                        batch_size = 20
                    elif y_train.shape[0]>= 2000:
                        batch_size = 32
                    else:
                        batch_size = 32
                    print(' batch_size:', batch_size)
                    #begin training 
                    savepath = 'models/pre_trained_model/predict/weights_update_by_detection_clstm_' + year + '_' + pat + '.h5'
                    if int(detect_res_idx) ==0:
                        is_first_time = True
                    else:
                        is_first_time = False
                    RPA_train(X_train, y_train,savepath,class_weights_train,batch_size, is_first_time=is_first_time)
                    print('rewrite the FLAG into False')
                    with open ('FLAG_'+year+'_'+pat+'.txt', 'w+') as f:
                        f.write(sess_num+'_'+'False'+'_'+str(detect_res_idx))
            except:
                pass
            