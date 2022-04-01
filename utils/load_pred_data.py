import os
import pandas as pd
import numpy as np
import pickle
from utils.loaddata_2011 import extract_interictal, extract_preictal

def post_process(thre, space_time, add_prev_res,file_old, file_in, file_out):

    #open and read the detection result
    with open(file_in) as f:
        res = f.readlines()
    #get the result list
    res_li = res[0][1:-1].split('][')

    # print('------------------')
    # print(file_in)
    # print(len(res_li))
    if add_prev_res:
        with open(file_old) as f_old:
            res_old = f_old.readlines()
        res_li_old = res_old[0][1:-1].split('][')
        # print('------------------')
        # print(file_old)
        # print(len(res_li_old))
        res_li = res_li_old[-600:] +res_li
    

    #create high seiz prob list and time list
    prob_li = []
    st_time_li = []
    end_time_li = []
    for i, prob in enumerate (res_li):
        prob = float(prob)
        if prob >= thre:
            st_time = i * 12
            end_time = (i+1) * 12
            st_time_li.append(st_time)   
            end_time_li.append(end_time)
            prob_li.append(prob)
    # print(st_time_li)
    # print(end_time_li)
    if len(prob_li)==0:
        st_time_li_new =[]
        end_time_li_new =[]
    else:
        # do the combine post processing
        if len(prob_li)>=2:
            flag_li = [True]*len(prob_li)
            i = 1
            # print(st_time_li)
            while i < len(prob_li):
                acc = 0
                while st_time_li[i] - st_time_li[i-1] <= space_time:
                    flag_li[i] = False
                    acc += 1
                    i += 1 
                    if i ==len(prob_li):
                        end_time_li[i-1-acc] = end_time_li[i-1]
                        break
                else:
                    end_time_li[i-1-acc] = end_time_li[i-1]
                    i +=1
        elif len(prob_li)==1:
            flag_li = [True]*len(prob_li)
        # only get the falg='true' 
        prob_li_new = []
        st_time_li_new =[]
        end_time_li_new =[]
        for i in range(len(flag_li)):
            if flag_li[i] == True:
                prob_li_new.append(prob_li[i])
                st_time_li_new.append(st_time_li[i])
                end_time_li_new.append(end_time_li[i])
    
    # save to csv
    df = pd.DataFrame()
    df['onset'] = st_time_li_new
    df['end'] = end_time_li_new
    print('-------------------------------------------')
    print(df)
    print('-------------------------------------------')
    df.to_csv(file_out, index=False)
    return df

def load_pred_data(sess_num, year, pat, cache_dir,detect_res_idx):
    #need to change to automtical next step 
        fn_study = sess_num
        # print(fn_study)
        
        # # post processing information
        thre = 0.65
        space_time = 60
        # make the dir if not exsit
        seiz_info_dir = 'result/'+year
        os.makedirs(seiz_info_dir, exist_ok=True)
        seiz_info_dir = 'result/'+year+'/pat_'+pat
        os.makedirs(seiz_info_dir, exist_ok=True)
        seiz_info_dir = 'result/'+year+'/pat_'+pat+'/'+sess_num
        os.makedirs(seiz_info_dir, exist_ok=True)

        if int(detect_res_idx) >=1:
            add_prev_res = True
        else:
            add_prev_res = False
        print('add_prev_res: ', add_prev_res)
        #set the result path
        detect_file_dir_old = 'result/'+year+'/pat_'+pat+'/'+sess_num+'/'+'detect_res_'+str(int(detect_res_idx)-1)+'.txt'
        detect_file_dir = 'result/'+year+'/pat_'+pat+'/'+sess_num+'/'+'detect_res_'+str(detect_res_idx)+'.txt'
        seiz_info_dir = 'result/'+year+'/pat_'+pat+'/'+sess_num+'/'+'detect_seiz_'+str(detect_res_idx)+'.csv'
        sz_table = post_process(thre, space_time, add_prev_res=add_prev_res,file_old=detect_file_dir_old, file_in = detect_file_dir, file_out =seiz_info_dir)

        #read buffer eeg data 
        buffer_data_dir = '{}/buffer_{}.pickle'.format(cache_dir, sess_num)
        if os.path.exists(buffer_data_dir):
            with open(buffer_data_dir, 'rb') as f:
                x = pickle.load(f)
                try:
                    print(len(x), x[0].shape)
                except:
                    print('empty inside')
        buffer_eeg = np.vstack(x)
        #load the data from the buffer and extract preictal and intericatal 
        # if sz_table.shape[0] > 0:
        #if nothing is there still rewrite the .pickle file
        extract_preictal(buffer_eeg,sess_num, sz_table, cache_dir,new_sample_rate=256)
        extract_interictal(buffer_eeg,sess_num, sz_table ,cache_dir,new_sample_rate=256)











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

    year = '2011'
    pats = [
        '1',
        #  '2',
        #  '4',
        # '5',
        # '6',
        # '7',
        # '8'
        # '10',
        # '11'
    ]
    for pat in pats:
        cache_dir = '/mnt/data7_M2/tempdata/RPA_online_train_buffer/real_time_data/' + year + '_' + pat
        os.makedirs(cache_dir, exist_ok=True)

        study_pd = pd.read_csv('utils/event_list/' + year + '_st_list.csv')
        study_pd = study_pd[(study_pd['pat_num'] == int(pat)) & (study_pd['anno_quality'] == 'Y')]
        study_list = study_pd['file_name'].values

        #need to change to automtical next step 
        sess_num = fn_study = study_list[2].split('.')[0]
        print(fn_study)
        
        # # post processing information
        thre = 0.65
        space_time = 60
        # make the dir if not exsit
        seiz_info_dir = 'result/'+year
        os.makedirs(seiz_info_dir, exist_ok=True)
        seiz_info_dir = 'result/'+year+'/pat_'+pat
        os.makedirs(seiz_info_dir, exist_ok=True)
        seiz_info_dir = 'result/'+year+'/pat_'+pat+'/'+sess_num
        os.makedirs(seiz_info_dir, exist_ok=True)

        #set the result path
        detect_file_dir = 'result/'+year+'/pat_'+pat+'/'+sess_num+'/'+'detect_res_1.txt'
        seiz_info_dir = 'result/'+year+'/pat_'+pat+'/'+sess_num+'/'+'detect_seiz_1.csv'
        sz_table = post_process(thre, space_time, file_in = detect_file_dir, file_out =seiz_info_dir)

        #read buffer eeg data 
        buffer_data_dir = '{}/buffer_{}.pickle'.format(cache_dir, sess_num)
        if os.path.exists(buffer_data_dir):
            with open(buffer_data_dir, 'rb') as f:
                x = pickle.load(f)
                try:
                    print(len(x), x[0].shape)
                except:
                    print('empty inside')
        buffer_eeg = np.vstack(x)
        print(buffer_eeg.shape)

        #load the data from the buffer and extract preictal and intericatal 
        # if sz_table.shape[0] > 0:
        #     extract_preictal(buffer_eeg,sess_num, sz_table, cache_dir,new_sample_rate=256)
        # else:
        #     print('sz_table is none')
        # extract_interictal(buffer_eeg,sess_num, sz_table ,cache_dir,new_sample_rate=256)



#set detail information this is for examle test only
# year = '2011'
# pat = '1'
# sess_num = '20110303{968424F4-75AD-4902-8CF1-9D8335408A90}'


