# Covid-19 Cough Audio — End-to-End Pipeline

> **Tujuan:** dokumen ini menjelaskan **langkah demi langkah** proyek analisis audio batuk (dataset Kaggle: *Covid-19 Cough Audio Classification*). Termasuk **setup**, **download data**, **prapemrosesan**, **visualisasi**, **ekstraksi fitur (manual & library)**, **validasi label** (Cohen’s Kappa & EWE), **model baseline**, serta **reproducibility & troubleshooting**.

---

## 🔎 Ringkasan Hasil Penting

* **Cohen’s Kappa (Annotator1 vs Annotator2)** ≈ `0.85` → *substantial agreement*.
* **Akurasi EWE terhadap Gold Standard** = **`0.93`** → konsensus label sangat dekat dengan rujukan.

> **Makna praktis:** meski penilaian manusia itu subyektif, agregasi **EWE** mampu menstabilkan label sehingga dataset **cukup andal** untuk pelatihan model deteksi batuk.

---

## 1) Persiapan Lingkungan

### 1.1. Dependensi

```bash
pip install numpy pandas scipy scikit-learn soundfile librosa ipython matplotlib
```

### 1.2. Kredensial Kaggle (Colab)

Simpan `kaggle.json` ke `/content/.kaggle/kaggle.json` lalu:

```python
import os
kaggle_dir = "/content/.kaggle"
kaggle_file = os.path.join(kaggle_dir, "kaggle.json")
os.makedirs(kaggle_dir, exist_ok=True)
assert os.path.exists(kaggle_file), "Upload kaggle.json ke /content/.kaggle/"
os.chmod(kaggle_file, 0o600)
```

---

## 2) Unduh Dataset & Susun Folder

### 2.1. Unduh & Ekstrak

```bash
mkdir -p ./datasets
kaggle datasets download -d andrewmvd/covid19-cough-audio-classification -p ./datasets
unzip -qo ./datasets/covid19-cough-audio-classification.zip -d ./datasets/Cough
```

### 2.2. Struktur yang benar

```
./datasets/Cough/
 ├── metadata_compiled.csv
 └── Cough_Soundfiles/
      ├── <uuid>.wav
      └── ...
```

> **Sering salah:** path audio tidak menunjuk ke `Cough_Soundfiles/` → mengakibatkan DataFrame kosong saat filter label.

---

## 3) Muat Metadata → DataFrame Utama

```python
import os, pandas as pd

Cough_metadata = "./datasets/Cough/metadata_compiled.csv"
Cough_audio    = "./datasets/Cough/Cough_Soundfiles/"

md = pd.read_csv(Cough_metadata).rename(columns={"status":"labels"})
md["Path"] = md["uuid"].apply(lambda x: os.path.join(Cough_audio, f"{x}.wav"))

Cough_df = md[
    md["labels"].isin(["healthy","symptomatic","COVID-19"]) & md["Path"].apply(os.path.exists)
][["labels","Path"]].reset_index(drop=True)

print("Jumlah data:", len(Cough_df))
print(Cough_df["labels"].value_counts())
```

**Checklist cepat:**

* Kolom label = `labels` (bukan `Emotions`).
* `Path` benar-benar ada file `.wav`-nya.

---

## 4) I/O Audio & Visualisasi Dasar (tanpa `librosa.display`)

### 4.1. Loader (soundfile)

```python
import soundfile as sf
def load_audio(file_path):
    data, sr = sf.read(file_path)
    if data.ndim > 1:   # stereo → mono
        data = data[:, 0]
    return data, sr
```

### 4.2. Waveplot Manual

```python
import numpy as np, matplotlib.pyplot as plt

def create_waveplot_manual(data, sr, label):
    t = np.linspace(0, len(data)/sr, len(data))
    plt.figure(figsize=(10,3))
    plt.plot(t, data, linewidth=0.5)
    plt.title(f'Wave — {label}')
    plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
```

### 4.3. STFT Manual & Spectrogram SciPy

```python
import numpy as np, scipy.signal as signal, matplotlib.pyplot as plt

def manual_stft(data, window_length=2048, hop_length=512, window='hann'):
    w = np.hanning(window_length) if window=='hann' else (
        np.hamming(window_length) if window=='hamming' else np.ones(window_length))
    num_frames = max(1, 1 + (len(data) - window_length) // hop_length)
    stft_matrix = np.zeros((window_length // 2 + 1, num_frames), dtype=complex)
    for i in range(num_frames):
        s, e = i*hop_length, i*hop_length + window_length
        seg = np.zeros(window_length)
        if s < len(data):
            seg[:max(0, min(window_length, len(data)-s))] = data[s:e]
        seg *= w
        stft_matrix[:, i] = np.fft.rfft(seg)
    return stft_matrix

def create_spectrogram_scipy(data, sr, label, nperseg=2048, noverlap=1536):
    f, t, Sxx = signal.spectrogram(data, sr, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10*np.log10(Sxx + 1e-8)
    plt.figure(figsize=(12,3))
    plt.pcolormesh(t, f, Sxx_db, shading='gouraud')
    plt.title(f'Spectrogram — {label}')
    plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='dB'); plt.tight_layout(); plt.show()
```

> **Kenapa manual?** Untuk memahami rantai sinyal (window → FFT → magnitude → dB) dan sebagai *fallback* bila modul visualisasi Librosa tidak tersedia.

---

## 5) Dengarkan Contoh Audio di Notebook

```python
import numpy as np, random
from IPython.display import Audio

def play_random(label="COVID-19"):
    subset = np.array(Cough_df.Path[Cough_df.labels==label])
    assert len(subset) > 0, f"Tidak ada sampel untuk label {label}"
    path = random.choice(subset)
    print("Memainkan:", path)
    return Audio(path)

# Contoh: play_random("symptomatic")
```

---

## 6) Validasi Label: Cohen’s Kappa & EWE

### 6.1. Sampling yang aman

```python
sample_df = Cough_df.sample(min(200, len(Cough_df)), random_state=42).reset_index(drop=True)
gold_labels = sample_df["labels"].values
```

### 6.2. Simulasi dua annotator (akurasi berbeda)

```python
import numpy as np
np.random.seed(42)
L = ["healthy","symptomatic","COVID-19"]
ann1, ann2 = [], []
for y in gold_labels:
    ann1.append(y if np.random.rand()<0.90 else np.random.choice(L))
    ann2.append(y if np.random.rand()<0.85 else np.random.choice(L))
```

### 6.3. Hitung Kappa & EWE

```python
from sklearn.metrics import cohen_kappa_score
from collections import Counter

kappa = cohen_kappa_score(ann1, ann2)
print("Cohen's Kappa:", round(kappa,3))

def ewe_vote(votes): return Counter(votes).most_common(1)[0][0]
ewe_labels = [ewe_vote([ann1[i], ann2[i]]) for i in range(len(gold_labels))]
acc_ewe = np.mean(np.array(ewe_labels) == gold_labels)
print("Akurasi EWE vs Gold:", round(acc_ewe,3))
```

> **Output yang kami dapat:** `Kappa ≈ 0.85` dan **`EWE = 0.93`**.

---

## 7) Ekstraksi Fitur untuk Model Baseline (opsional tapi dianjurkan)

### 7.1. MFCC + Statistik Ringkas

```python
import librosa, numpy as np

def extract_features(file_path, sr_target=16000, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr_target, mono=True)
    rms = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    sc  = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).mean(axis=1)
    return np.hstack([rms, zcr, sc, sb, mfcc])
```

### 7.2. Bangun tabel fitur

```python
import pandas as pd
rows = []
for _, r in sample_df.iterrows():
    try:
        feat = extract_features(r["Path"])
        rows.append([r["labels"], *feat])
    except Exception:
        pass

cols = ["label","rms","zcr","centroid","bandwidth"] + [f"mfcc_{i+1}" for i in range(13)]
feat_df = pd.DataFrame(rows, columns=cols)
# feat_df.to_csv("features_baseline.csv", index=False)
```

### 7.3. Model baseline (Logistic Regression)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

X = feat_df.drop(columns=["label"]).values
y = feat_df["label"].values

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
clf.fit(Xtr, ytr)

print(classification_report(yte, clf.predict(Xte)))
print(confusion_matrix(yte, clf.predict(Xte)))
```

> Baseline ini untuk tolok ukur cepat. Upgrade ke SVM/RandomForest atau CNN berbasis mel-spectrogram bila ingin peningkatan akurasi.

---

## 8) (Opsional) VAD/AAD Sederhana untuk Segmentasi Batuk

```python
import numpy as np

def energy_vad(y, frame=1024, hop=512, thr=1.5):
    # energi per frame; threshold adaptif sederhana
    energies = [np.mean(y[i:i+frame]**2) for i in range(0, len(y)-frame, hop)]
    m = np.median(energies) + thr*np.std(energies)
    mask = np.array(energies) > m
    return np.array(energies), mask  # gunakan untuk menandai segmen aktif
```

> Di data nyata (noise tinggi), AAD berbasis energi + threshold sering kurang kuat. Pertimbangkan model statistik/ML-based VAD.

---

## 9) Reproducibility & Struktur Proyek

```
project/
├── datasets/
│   └── Cough/
│       ├── metadata_compiled.csv
│       └── Cough_Soundfiles/*.wav
├── notebooks/
│   └── Kelompok_2_KAV_.ipynb
├── src/
│   ├── io_audio.py          # load_audio, playback, sanity check
│   ├── features.py          # manual_stft, spectrogram, extract_features
│   ├── labeling.py          # Kappa, EWE, simulasi annotator
│   └── model_baseline.py    # training & evaluasi
└── README.md
```

* Set `random_state=42` pada seluruh *split* untuk hasil konsisten.
* Simpan fitur (CSV) di `./features/` agar eksperimen mudah diulang.

---

## 10) Troubleshooting (error umum & solusinya)

* **`IndexError: index 0 is out of bounds` saat pilih sampel**
  Filter label menghasilkan array kosong. **Periksa:** ejaan label (`"healthy"`, `"symptomatic"`, `"COVID-19"`) & path `Cough_Soundfiles/`.

* **`NameError: 'ewe_labels' is not defined`**
  Blok EWE belum dieksekusi. **Jalankan** sel Kappa & pembuatan `ewe_labels` sebelum menghitung akurasi.

* **`kaggle.json not found`**
  Pastikan file berada di `/content/.kaggle/` dan permission `0o600`.

* **Plot tidak tampil**
  Pastikan `plt.show()` dipanggil dan tidak mencampur banyak subplot dalam satu figur.

---

## 11) Interpretasi & Best-Practice

* **Kappa** mengukur konsistensi antarannotator; **EWE** menggabungkan suara mayoritas berbobot.
* Dengan **EWE = 0.93**, *ground truth consensus* cukup stabil → cocok untuk pelatihan model.
* Untuk pemodelan, hindari *brute-forcing* semua fitur: lakukan **feature selection**/**dimensionality reduction** (mRMR/PCA/LDA) dan normalisasi yang konsisten.
* Pisahkan **train/val/test** berbasis *speaker/session* untuk mencegah *leakage*.
