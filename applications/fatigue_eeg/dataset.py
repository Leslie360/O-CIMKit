import os
import sys
import numpy as np
import mne

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_real_sleep_edf_2ch(data_dir=None):
    """
    Loads EEG PSG and Hypnogram annotations from Sleep-EDF dataset.
    Returns downsampled epochs and stage labels.
    """
    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, "data", "datasets", "sleep_edf")
        
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Sleep-EDF dataset directory not found at: {data_dir}")
        
    print(f"📖 Loading Sleep-EDF dataset from: {data_dir}")
    
    psg_files = sorted([f for f in os.listdir(data_dir) if f.endswith('-PSG.edf')])
    hyp_files = sorted([f for f in os.listdir(data_dir) if f.endswith('-Hypnogram.edf')])
    
    pairs = []
    for psg in psg_files:
        prefix = psg.split('-')[0][:6]
        for hyp in hyp_files:
            if hyp.startswith(prefix):
                pairs.append((os.path.join(data_dir, psg), os.path.join(data_dir, hyp)))
                break
                
    all_epochs, all_labels = [], []
    stage_map = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 2, 'Sleep stage 4': 2}
    sfreq_target = 100
    epoch_samples = 30 * sfreq_target
    
    for psg_file, hyp_file in pairs:
        try:
            raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
            eeg_picks = mne.pick_types(raw.info, eeg=True)
            if len(eeg_picks) < 2:
                continue
            raw.pick(eeg_picks[:2])
            if raw.info['sfreq'] != sfreq_target:
                raw.resample(sfreq_target, verbose=False)
                
            annotations = mne.read_annotations(hyp_file)
            raw.set_annotations(annotations)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=30,
                               baseline=None, preload=True, verbose=False)
                              
            for epoch_data, event in zip(epochs.get_data(), epochs.events):
                event_id_val = event[2]
                stage_name = None
                for name, eid in event_id.items():
                    if eid == event_id_val:
                        stage_name = name
                        break
                if stage_name is None or stage_name not in stage_map:
                    continue
                label = stage_map[stage_name]
                if epoch_data.shape[1] >= epoch_samples:
                    epoch_segment = epoch_data[:, :epoch_samples].T
                else:
                    pad_len = epoch_samples - epoch_data.shape[1]
                    epoch_segment = np.pad(epoch_data.T, ((0, pad_len), (0, 0)), mode='constant')[:epoch_samples, :]
                all_epochs.append(epoch_segment)
                all_labels.append(label)
        except Exception:
            continue
            
    X = np.array(all_epochs)
    y = np.array(all_labels)
    # Downsample sequence to save memory
    X = X[:, ::4, :]  
    
    print(f"✅ Sleep-EDF dataset loaded: {X.shape[0]} samples, shape: {X.shape}, labels distribution: {np.bincount(y)}")
    return X, y
