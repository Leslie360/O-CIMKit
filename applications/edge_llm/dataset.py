import os
import sys
import wfdb
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_mitbih_data(data_dir=None):
    """
    Loads MIT-BIH Arrhythmia Database files from data_dir.
    Packs beats into 200-sample sequences centered at annotations.
    """
    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, "data", "datasets", "mitdb")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ MIT-BIH dataset directory not found at: {data_dir}")

    print(f"📖 Loading MIT-BIH data from: {data_dir}")

    normal_beats = []
    anomaly_beats = []

    for f in sorted(os.listdir(data_dir)):
        if f.endswith('.dat'):
            rec = f[:-4]
            try:
                sig = wfdb.rdrecord(os.path.join(data_dir, rec)).p_signal[:, 0]
                ann = wfdb.rdann(os.path.join(data_dir, rec), 'atr')

                for peak, sym in zip(ann.sample, ann.symbol):
                    if 150 < peak < len(sig) - 150:
                        beat = sig[peak - 100:peak + 100]
                        if len(beat) == 200:
                            if sym == 'N':
                                normal_beats.append(beat)
                            else:
                                anomaly_beats.append(beat)
            except Exception as e:
                continue

    print(f"  Loaded: Normal beats = {len(normal_beats)}, Anomaly beats = {len(anomaly_beats)}")
    return normal_beats, anomaly_beats
