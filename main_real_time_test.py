import os
import pickle
from models.convlstm import ConvNN
import stft
import numpy as np
import pandas as pd
import json
from sklearn.utils import shuffle
from utils.data_gen_new import DataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from multiprocessing.pool import ThreadPool
from utils.loaddata_2011 import load_data, dc_notch_filter, resample_data
from utils.preprocessing import detect_interupted_data, ica_arti_remove
import gc

class real_time_test:

    def __init__(self,model, data, total_duration, eeg_sample_rate, chs,fn_sdy_name, cache_dir,pred_model_save_path,res_savepath,sess_num):
        """
        model: seizure detection and prediction model, both are conv-lstm in this test
        data: the real-time data
        total_duration: total duration of one session test
        eeg_sample_rate: the inital data eeg sample rate
        chs: the channel and eletrode information for detection pretrained model
        chs_pred: the channel and eletrode information for prediction pretrained model
        fn_sdy_name: the file study name
        cache_dir: the cache_dir to save the real-time buffer
        pred_model_save_path: the new prediction model saving path
        res_savepath: the result saving path
        """
        self.model = model
        self.data = data
        self.total_duration = total_duration
        self.eeg_sample_rate = eeg_sample_rate
        self.chs = chs
        self.chs_pred = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
        self.fn_sdy_name = fn_sdy_name
        self.cache_dir = cache_dir
        self.pred_model_save_path = pred_model_save_path
        self.res_savepath = res_savepath
        self.sess_num = sess_num


    def RPA_detect(self, X_test, idx):
        
        # model = ConvNN()
        self.model.setup(X_test.shape)
        self.model.load_trained_weights("models/pre_trained_model/detect/nhan_ICA_12_train.h5")
        predictions = self.model.predict_proba(X_test)
        with open(self.res_savepath + 'detect_res_'+str(idx)+'.txt', 'a') as f:
            f.write(str(predictions))

    def RPA_predict(self, X_test):
        
        self.model.setup(X_test.shape)

        try:
            self.model.load_trained_weights(self.pred_model_save_path)
            print('success load the new model')
        except:
            self.model.load_trained_weights('models/pre_trained_model/predict/weights_clstm_goodpat.h5')
            print('success load the old model')
        
        predictions = self.model.predict_proba(X_test)
        with open(self.res_savepath + 'pred_res.txt', 'a') as f:
            f.write(str(predictions))

    def feed_data_test(self):
        real_time_window = 1
        ###############array define to test#######################
        data_detect_temp = []
        data_pred_temp = []
        self.data_buffer = []
        #detection record index
        idx =0
        index_Flag = False
        detect_Flag = False
        fn = '{}/buffer_{}.pickle'.format(self.cache_dir, self.fn_sdy_name.split('.')[0])
        ###############array define to test#######################
        for i in range(int(self.total_duration)):
            print('-------------------------------------------')
            print(i)
            print('---------------------------------------')
            data_temp = self.data[:,i * int(self.eeg_sample_rate): (i + real_time_window) * int(self.eeg_sample_rate)]
            data_detect = data_temp
            
            
            # reset the prediction electrode seqence and reorder deat_Pred electrode information
            data_pred = self.re_order(data_temp,self.chs,self.chs_pred)
            # resample the data
            data_pred = resample_data(data_pred, int(self.eeg_sample_rate),256)
            # prediction append to buffer
            # back to previous mne read shape to save on buffer
            data_pred_mne = np.moveaxis(data_pred, 0, -1)
            self.data_buffer.append(data_pred_mne)
            if index_Flag == True and detect_Flag==True:
                idx +=1
                index_Flag = False
                detect_Flag = False
            # rewrite if the buffer is exceed two hour information
            # 4hour is 3600*4
            if len(self.data_buffer) >=3600*4:
                # write buffer to disk for ech 2 hour information
                with open(fn, 'wb') as f:
                    pickle.dump(self.data_buffer, f, pickle.HIGHEST_PROTOCOL)
                #reset to zero
                
                del self.data_buffer[:3600*2]
               
                # self.data_buffer =[]
                # set a flag on a txt file
                with open ('FLAG_'+year+'_'+pat+'.txt', 'w+') as f:
                    f.write(self.sess_num+'_'+'True'+'_'+str(idx))
                # increase the detection result index Flag
                index_Flag = True
                gc.collect()
                # with open(self.res_savepath + 'detect_res.txt', 'w+') as f:
                #     pass
                # os.system('CUDA_VISIBLE_DEVICES=3 python main_real_time_train.py')
            #####################################do the test###################################
            # print(data_pred.shape, data_detect.shape)
            if len(data_detect_temp) < 12:
                data_detect_temp.append(data_detect)
                detect_Flag = False
            else:
                X_detect_test = np.hstack(data_detect_temp)
                chs_ap = self.chs
                X_detect_test = self.ICA_process(X_detect_test, self.eeg_sample_rate, chs_ap)
                X_detect_test = resample_data(X_detect_test, int(self.eeg_sample_rate),250)
                X_detect_test= self.stft_cal(X_detect_test,framelength=250,type='detect')
                X_detect_test = np.moveaxis(X_detect_test,0,-1)
                X_detect_test = np.expand_dims(X_detect_test, 0)
                self.RPA_detect(X_detect_test,idx)
                data_detect_temp = []
                data_detect_temp.append(data_detect)
                detect_Flag = True
            if len(data_pred_temp) < 30:
                data_pred_temp.append(data_pred)
            else:
                X_pred_test = np.hstack(data_pred_temp) 
                # resample the data
                X_pred_test, chs_ap = dc_notch_filter(X_pred_test, self.chs_pred, 256)
                X_pred_test = self.stft_cal(X_pred_test, framelength=256,type='pred')
                X_pred_test = np.moveaxis(X_pred_test, 0, -1)
                X_pred_test = np.expand_dims(X_pred_test, 0)
                self.RPA_predict(X_pred_test)
                data_pred_temp = []
                data_pred_temp.append(data_pred)
            #######################################################################################


    def stft_cal(self, X_test,framelength=256, type='pred'):

        X_test = X_test.transpose()
        stft_data = stft.spectrogram(X_test, framelength=framelength, centered=False)
        
        stft_data = np.transpose(stft_data, (1, 2, 0))
        stft_data = np.abs(stft_data) + 1e-6
        if type == 'pred':
            stft_data = stft_data[1:57,:, 1:]
        if type == 'detect':
            stft_data = stft_data[:, :, 1:]
        stft_data = np.log10(stft_data)
        indices = np.where(stft_data <= 0)
        stft_data[indices] = 0
        stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                      stft_data.shape[1],
                                      stft_data.shape[2])


        return stft_data

    def ICA_process(self, data_seg, fsamp, chs_ap):

        if detect_interupted_data(data_seg.transpose(), fsamp):
            print('BAD DATA DETECTED! Skipping this 12-second segment due to interupted signals...')
            ica_filt_s = data_seg
        else:
            print('GOOD DATA!')
            ica_filt_s = ica_arti_remove(data_seg, fsamp, chs_ap)
            if ica_filt_s is None:
                print('Skipping this 12-second segment due to failed ICA...')
                ica_filt_s = data_seg
                
        return ica_filt_s

    def re_order(self,sig,chs_old,chs_new):
        """
        sig: input signal array
        chs_old: old detection electrode sequence
        chs_new: prediction electrode sequence
        """
        sig_new = np.empty([19,int(self.eeg_sample_rate)*1])
        for elec_idx in range(sig.shape[0]):
            old_elec = chs_old[elec_idx]
            sig_new[chs_new.index(old_elec),:] = sig[elec_idx,:]
        return sig_new




def RPA_test(X_test, y_test, pred_model_save_path):
    params = {'dim': (56, 19, 128, 1),
              'batch_size': 32,
              'n_classes': 2,
              'n_channels': 19,
              'shuffle': False}
    test_generator = DataGenerator(X_test, y_test, **params)
    model = ConvNN()
    model.setup(X_test.shape)

    model.load_trained_weights(pred_model_save_path)
    # model.load_trained_weights("weights_clstm_goodpat.h5")
    predictions = model.evaluate_generator(test_generator)
    number = y_test.shape[0] - y_test.shape[0] % 32
    y_test = y_test[:number]

    auc_test = roc_auc_score(y_test, predictions)
    print('Test AUC is:', auc_test)
    print('AUC is: {}'.format(auc_test * 100))
    return auc_test


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="RPA",
                        help="FB, EpilepsiaSurf, CHBMIT or Kaggle2014Det")
    parser.add_argument("--mode", default="test",
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
    #define the model
    model = ConvNN()
    for pat in pats:
        
        #define the cache_dir
        cache_dir = '/raid/neurosyd/RPA_online_train_buffer/real_time_data/' + year + '_' + pat
        os.makedirs(cache_dir, exist_ok=True)
        with open('json_file/SETTINGS_' + year + '.json') as f:
            settings = json.load(f)
        print(settings['datadir'])

        #get the patient and study information 
        study_pd = pd.read_csv('utils/event_list/' + year + '_st_list.csv')
        study_pd = study_pd[(study_pd['pat_num'] == int(pat)) & (study_pd['anno_quality'] == 'Y')]
        study_list = study_pd['file_name'].values
        print(study_list)
        #test one patient entirely
        for study_idx in range(len(study_list)):
            gc.collect()
            # study number
            # sess_num = fn_study = study_list[2].split('.')[0]
            sess_num = fn_study = study_list[study_idx].split('.')[0]
            #set inial flag= False
            with open ('FLAG_'+year+'_'+pat+'.txt', 'w+') as f:
                f.write(sess_num+'_'+'False'+'_'+str(0))

            study_list_dir = [os.path.join(settings['datadir'], study) for study in study_list]
            study_load_dir = study_list_dir[study_idx]
            print(study_load_dir)
            #define and create the result and model save path 
            ref_path = '/home/yikaiy/prediction_detection/ref/ref_' + year + '/' + year + '_pat' + pat + '.txt'
            pred_model_save_path = 'models/pre_trained_model/predict/weights_update_by_detection_clstm_' + year + '_' + pat + '.h5'
            #make the result save path
            seiz_info_dir = 'result/'+year
            os.makedirs(seiz_info_dir, exist_ok=True)
            seiz_info_dir = 'result/'+year+'/pat_'+pat
            os.makedirs(seiz_info_dir, exist_ok=True)
            seiz_info_dir = 'result/'+year+'/pat_'+pat+'/'+sess_num
            os.makedirs(seiz_info_dir, exist_ok=True)
            res_savepath = 'result/'+year+'/pat_'+pat+'/'+sess_num+'/'
            # load the real-time data
            data, total_duration, eeg_sample_rate, chs ,fn_study_name = load_data(study_load_dir, ref_path, cache_dir)
            
            #begin the test
            trial = real_time_test(model,data, total_duration, eeg_sample_rate, chs, fn_study_name, cache_dir,pred_model_save_path,res_savepath,sess_num)
            trial.feed_data_test()
            

        #####################################code example of define two thread-pool#########################
        # pool_1 = ThreadPool(processes=1)
        # pool_2 = ThreadPool(processes=1)
        # async_result_1 = pool_1.apply_async(resample_data,
        #                                     (data_temp, int(self.eeg_sample_rate), 256))  # tuple of args for foo
        # async_result_2 = pool_2.apply_async(resample_data, (data_temp, int(self.eeg_sample_rate), 250))
        # data_pred = async_result_1.get()
        # data_detect = async_result_2.get()
            
        #######################################################################################