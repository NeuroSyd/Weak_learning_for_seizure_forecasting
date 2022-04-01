import numpy as np
import mne
import pandas as pd

def convert_AP_montage(m, elec_names):
    # Referencing: using AP montages
    m_s_ref = []
    if 'T3(T7)' in elec_names:
        mon_fn = 'utils/AP.txt'
    elif 'T3' in elec_names:
        mon_fn = 'utils/AP_new.txt'
    else:
        mon_fn = 'utils/AP_old.txt'
    ap = pd.read_csv(mon_fn,header=0,index_col=None)
    # print (ap, ap.index)
    # print (ap['input'])
    # print (ap['input'][0])
    for i in range(len(ap.index)):
        input_ind = elec_names.index(ap['input'][i])
        ref_ind = elec_names.index(ap['ref'][i])
        tmp = m[:,input_ind] - m[:,ref_ind]
        # print (tmp.shape)
        m_s_ref.append(tmp[:,np.newaxis])
        # print (input_ind, ref_ind)
    m_s_ref = np.concatenate(m_s_ref, axis=1)
    return m_s_ref, ap['name'].values


def create_mne_raw(data, sfreq, chs=None):
    '''
    data: signal with shape (channel x samples)
    '''
    if chs is None:
        chs_ = ['ch{}'.format(i) for i in range(data.shape[0])]
    else:
        print (data.shape[0], len(chs))
        assert data.shape[0] == len(chs)
        chs_ = chs    
    ch_types = ['eeg' for i in range(len(chs_))]    
    info = mne.create_info(ch_names=chs_, sfreq=sfreq, ch_types=ch_types, verbose=False)
    raw = mne.io.RawArray(data*1e-6, info)
    return raw