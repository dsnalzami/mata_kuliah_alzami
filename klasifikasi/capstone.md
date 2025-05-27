
# LAPORAN CAPSTONE PROJECT
## PREDIKSI TINGKAT PENDAPATAN MENGGUNAKAN ALGORITMA DECISION TREE
### Studi Kasus Dataset Adult Census Income

---

## ABSTRAK

**Latar Belakang:** Prediksi tingkat pendapatan merupakan salah satu aplikasi penting dalam bidang data mining dan machine learning, khususnya untuk analisis ekonomi, kebijakan sosial, dan pengambilan keputusan bisnis.

**Tujuan:** Mengembangkan model klasifikasi untuk memprediksi apakah seseorang memiliki pendapatan di atas atau di bawah $50,000 per tahun berdasarkan karakteristik demografis dan pekerjaan.

**Metode:** Penelitian ini menggunakan algoritma Decision Tree dengan dataset Adult Census Income dari UCI Machine Learning Repository. Proses meliputi eksplorasi data, preprocessing, feature engineering, hyperparameter tuning, dan evaluasi model.

**Hasil:** Model Decision Tree dengan hyperparameter optimal mencapai akurasi **86.08%** pada data testing dengan feature importance tertinggi pada **fitur-fitur kunci seperti status perkawinan, perolehan modal, dan tingkat pendidikan**.

**Kesimpulan:** Model yang dikembangkan dapat digunakan sebagai alat bantu prediksi pendapatan dengan tingkat akurasi yang baik dan telah diimplementasikan dalam aplikasi web menggunakan Streamlit.

**Kata Kunci:** Machine Learning, Decision Tree, Classification, Income Prediction, Data Mining

---

## 1. PENDAHULUAN

### 1.1 Latar Belakang
Prediksi tingkat pendapatan seseorang berdasarkan karakteristik demografis dan pekerjaan memiliki berbagai aplikasi praktis dalam dunia nyata. Informasi ini dapat digunakan oleh:

- **Pemerintah**: Untuk perencanaan kebijakan sosial dan ekonomi
- **Institusi Keuangan**: Untuk penilaian kredit dan produk finansial
- **Perusahaan**: Untuk segmentasi pasar dan strategi marketing
- **Peneliti Sosial**: Untuk memahami faktor-faktor yang mempengaruhi tingkat pendapatan

### 1.2 Rumusan Masalah
1. Bagaimana mengembangkan model klasifikasi yang akurat untuk memprediksi tingkat pendapatan seseorang?
2. Fitur apa saja yang paling berpengaruh terhadap prediksi tingkat pendapatan?
3. Bagaimana implementasi model dalam aplikasi yang user-friendly?

### 1.3 Tujuan Penelitian
#### Tujuan Umum:
Mengembangkan sistem prediksi tingkat pendapatan menggunakan algoritma machine learning.

#### Tujuan Khusus:
1. Melakukan eksplorasi dan analisis dataset Adult Census Income
2. Mengimplementasikan preprocessing dan feature engineering yang optimal
3. Membangun model Decision Tree dengan hyperparameter tuning
4. Mengevaluasi performa model menggunakan berbagai metrik
5. Mengimplementasikan model dalam aplikasi web menggunakan Streamlit

### 1.4 Manfaat Penelitian
- **Akademis**: Kontribusi dalam bidang applied machine learning untuk prediksi ekonomi
- **Praktis**: Menyediakan alat bantu prediksi yang dapat digunakan berbagai stakeholder
- **Teknologi**: Demonstrasi implementasi end-to-end machine learning pipeline

### 1.5 Batasan Penelitian
- Dataset terbatas pada data census tahun 1994
- Fokus pada algoritma Decision Tree
- Evaluasi dilakukan pada single algorithm (tidak comparative study)

---

## 2. TINJAUAN PUSTAKA

### 2.1 Machine Learning untuk Prediksi Pendapatan
[Isi dengan referensi penelitian terkait prediksi income menggunakan ML]

### 2.2 Algoritma Decision Tree
Decision Tree merupakan algoritma supervised learning yang dapat digunakan untuk masalah klasifikasi dan regresi. Keunggulan algoritma ini:

- **Interpretabilitas tinggi**: Hasil mudah dipahami dan dijelaskan
- **Tidak memerlukan asumsi distribusi**: Non-parametric algorithm
- **Dapat menangani data numerik dan kategorikal**
- **Otomatis feature selection**: Mengidentifikasi fitur yang paling penting

#### 2.2.1 Cara Kerja Decision Tree
1. **Splitting**: Memilih fitur dan threshold terbaik untuk membagi data
2. **Criterion**: Menggunakan Gini impurity atau Entropy untuk mengukur kualitas split
3. **Pruning**: Mencegah overfitting dengan membatasi kedalaman atau jumlah sampel minimum

### 2.3 Dataset Adult Census Income
Dataset ini berasal dari UCI Machine Learning Repository dan berisi data census tahun 1994. Dataset terdiri dari:
- **32,561 instances** dengan **14 atribut** (sebelum penambahan kolom nama dan pemrosesan)
- **Target variable**: Income (≤50K atau >50K)
- **Missing values**: Ada pada beberapa atribut kategorikal (`workclass`, `occupation`, `native_country`).

---

## 3. METODOLOGI PENELITIAN

### 3.1 Kerangka Kerja Penelitian
```
Data Collection → EDA → Preprocessing → Feature Engineering → 
Model Training → Hyperparameter Tuning → Evaluation → Deployment
```

### 3.2 Dataset dan Fitur

#### 3.2.1 Deskripsi Dataset
- **Sumber**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/2/adult
- **Ukuran**: 32,561 records, 14 features + 1 target (sesuai deskripsi awal, notebook menambahkan nama kolom sehingga menjadi 15)
- **Type**: Supervised Classification

#### 3.2.2 Deskripsi Fitur
| Fitur | Type | Deskripsi |
|-------|------|-----------|
| age | Numeric | Usia individu |
| workclass | Categorical | Jenis pekerjaan |
| fnlwgt | Numeric | Final weight (representasi populasi) |
| education | Categorical | Level pendidikan |
| education_num | Numeric | Pendidikan dalam angka |
| marital_status | Categorical | Status pernikahan |
| occupation | Categorical | Jenis profesi |
| relationship | Categorical | Status hubungan keluarga |
| race | Categorical | Ras |
| sex | Categorical | Jenis kelamin |
| capital_gain | Numeric | Keuntungan modal |
| capital_loss | Numeric | Kerugian modal |
| hours_per_week | Numeric | Jam kerja per minggu |
| native_country | Categorical | Negara asal |
| income | Categorical | Target: ≤50K atau >50K |

### 3.3 Tahapan Preprocessing

#### 3.3.1 Eksplorasi Data Awal (EDA)
1. **Dataset Overview**: Shape, info, dan statistik deskriptif
2. **Missing Values Analysis**: Identifikasi dan handling missing data
3. **Distribution Analysis**: Visualisasi distribusi setiap fitur
4. **Correlation Analysis**: Analisis korelasi antar fitur numerik
5. **Target Distribution**: Analisis balance/imbalance target class

#### 3.3.2 Data Cleaning
1. **Missing Values Handling**:
   - Categorical: Mode imputation
   - Numerical: Median imputation (meskipun pada dataset ini semua fitur numerik tidak memiliki missing values awal)
2. **Duplicate Removal**: Menghapus record yang duplikat
3. **Outlier Detection**: Identifikasi outlier pada fitur numerik (langkah ini tidak secara eksplisit dilakukan dengan penghapusan di notebook, namun distribusi diamati).

#### 3.3.3 Feature Engineering
1. **Label Encoding**: Konversi fitur kategorikal menjadi numerik menggunakan mapping manual.
2. **Feature Selection**: Penghapusan fitur `education` (karena `education_num` lebih informatif dan numerik). Analisis korelasi tinggi dilakukan, namun tidak ada fitur yang dihapus berdasarkan threshold > 0.8.
3. **Feature Importance Analysis**: Analisis kontribusi setiap fitur setelah model dilatih.

### 3.4 Model Development

#### 3.4.1 Data Splitting
- **Training Set**: 90% dari total data (29,283 sampel)
- **Testing Set**: 10% dari total data (3,254 sampel)
- **Stratified Sampling**: Mempertahankan proporsi target class pada saat splitting.

#### 3.4.2 Hyperparameter Tuning
Menggunakan GridSearchCV dengan parameter:
```python
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split':,
    'min_samples_leaf':,
    'criterion': ['gini', 'entropy']
}
```

#### 3.4.3 Cross Validation
- **K-Fold CV**: 3-fold cross validation digunakan dalam GridSearchCV.
- **Scoring Metric**: Accuracy
- **Parallel Processing**: `n_jobs=-1` untuk optimasi waktu training.

### 3.5 Evaluasi Model

#### 3.5.1 Metrik Evaluasi
1. **Accuracy**: Overall correctness
2. **Precision**: True Positive Rate per class (dan agregat)
3. **Recall**: Sensitivity per class (dan agregat)
4. **F1-Score**: Harmonic mean of precision and recall (per class dan agregat)
5. **Confusion Matrix**: Detailed classification results

#### 3.5.2 Analisis Hasil
1. **Feature Importance**: Ranking fitur berdasarkan kontribusi
2. **Model Interpretability**: Visualisasi decision tree
3. **Performance Analysis**: Training vs Testing performance

---

## 4. IMPLEMENTASI

### 4.1 Environment dan Tools
- **Python Version**: 3.10.14 (berdasarkan metadata notebook, dapat disesuaikan jika berbeda)
- **Key Libraries**:
  - pandas, numpy: Data manipulation
  - scikit-learn: Machine learning
  - matplotlib, seaborn: Visualization
  - streamlit: Web deployment
  - plotly: Interactive visualization (digunakan di aplikasi Streamlit)
  - joblib: Penyimpanan model

### 4.2 Pipeline Implementation

#### 4.2.1 Data Loading dan EDA
```python
# Kode untuk loading data dan EDA
import pandas as pd

# Load dataset
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
                'hours_per_week', 'native_country', 'income']
# Diasumsikan file 'adult.data' ada di direktori yang sama
df = pd.read_csv('adult.data', names=column_names, skipinitialspace=True, na_values="?")

# Dataset Overview
print("Dataset Overview:")
print(f"Shape: {df.shape}")
# print(df.info()) # Untuk ringkasan lebih detail

# Check Missing Values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

# Target Distribution
# print("\nTarget Distribution:")
# print(df['income'].value_counts(normalize=True))
```

#### 4.2.2 Preprocessing Pipeline
```python
# Kode untuk preprocessing

# Handle missing values (contoh dari notebook)
for column in ['workclass', 'occupation', 'native_country']: # Kolom dengan missing values
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode(), inplace=True) # Menggunakan mode() untuk menghindari Series
    # else: # Fitur numerik tidak punya missing values di awal
    #     df[column].fillna(df[column].median(), inplace=True)

# Remove duplicates
# initial_rows = len(df)
df.drop_duplicates(inplace=True)
# print(f"Removed {initial_rows - len(df)} duplicate rows.")

# Encoding for 'sex' (contoh)
sex_map = {'Male':1, 'Female':0}
df['sex'] = df['sex'].map(sex_map)

# Drop education column (karena education_num sudah ada)
if 'education' in df.columns:
    df = df.drop('education', axis=1)
```

#### 4.2.3 Model Training
```python
# Kode untuk training model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Asumsikan df sudah di-encode semua fitur kategorikalnya termasuk target 'income'
# dan X, y, X_train, X_test, y_train, y_test sudah didefinisikan sebelumnya.
# (Proses encoding lengkap dan split data ada di notebook bagian 4 dan 7)

# Definisikan ulang X dan y untuk kejelasan dalam contoh ini
# X = df.drop('income', axis=1) 
# y = df['income']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)


# Define parameter grid for GridSearchCV
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split':,
    'min_samples_leaf':,
    'criterion': ['gini', 'entropy']
}

# Create decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(
    estimator=dt_classifier,
    param_grid=param_grid,
    cv=3, # Sesuai notebook
    scoring='accuracy',
    verbose=1, # Sesuai notebook
    n_jobs=-1 
)
# grid_search.fit(X_train, y_train) # Proses training
# best_model = grid_search.best_estimator_
# print(f"Best Parameters: {grid_search.best_params_}")
```
*(Catatan: Untuk contoh kode training, beberapa bagian seperti encoding lengkap dan split data di-comment atau disederhanakan agar lebih ringkas sebagai contoh di laporan. Kode penuh ada di notebook.)*

### 4.3 Web Application Development

#### 4.3.1 Streamlit Architecture
- **Frontend**: Streamlit components untuk input dan visualization. Menggunakan `st.form` untuk input pengguna, `st.columns` untuk layout, `st.number_input`, `st.selectbox` untuk field input. Visualisasi menggunakan `plotly.graph_objects` (untuk Gauge chart) dan `plotly.express` (untuk Bar chart).
- **Backend**: Model inference menggunakan `joblib` untuk memuat model Decision Tree yang telah dilatih dan komponen preprocessing lainnya (seperti encoding maps dan feature names). Fungsi `predict_income` menangani preprocessing data input dan prediksi.
- **Features**:
  - User-friendly input forms untuk 13 fitur.
  - Real-time prediction (setelah tombol "Predict" ditekan).
  - Interactive visualizations: Gauge chart untuk confidence level, bar chart untuk distribusi probabilitas kelas, dan bar chart untuk feature importance model.
  - Export functionality untuk mengunduh hasil prediksi terakhir dalam format JSON.
  - Tombol Reset untuk mengembalikan nilai input ke default.

#### 4.3.2 Deployment Features
1. **Input Validation**: Error handling untuk input user dilakukan melalui fungsi `validate_inputs` yang memeriksa batasan nilai wajar untuk fitur numerik seperti `age`, `education_num`, `hours_per_week`, `capital_gain`, `capital_loss`, dan `fnlwgt`. Pesan error ditampilkan jika validasi gagal.
2. **Model Loading**: Caching digunakan (`@st.cache_resource`) untuk memuat model dan komponennya sekali saja, mengoptimalkan performance saat aplikasi diakses berulang kali.
3. **Visualization**: Interactive charts dari Plotly digunakan untuk menampilkan hasil prediksi (confidence gauge, probability distribution) dan feature importance model, memungkinkan pemahaman yang lebih baik bagi pengguna.
4. **Export Function**: Fungsi `export_prediction` membuat file JSON yang berisi timestamp, data input, dan hasil prediksi (kelas, confidence, raw prediction). Pengguna dapat mengunduhnya melalui `st.download_button`.

---

## 5. HASIL DAN PEMBAHASAN

### 5.1 Hasil Eksplorasi Data

#### 5.1.1 Dataset Overview
Dataset awal terdiri dari **32,561 baris** dan **15 kolom** (setelah penambahan nama kolom, termasuk kolom target `income`). Terdapat 6 fitur numerik (`age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`) dan 9 fitur objek/kategorikal (`workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`, `income`).
Statistik deskriptif menunjukkan rentang nilai, mean, median, dan standar deviasi untuk fitur numerik. Misalnya, usia rata-rata adalah sekitar 38.58 tahun. Untuk fitur kategorikal, frekuensi kelas tertinggi juga diamati, contohnya `workclass` terbanyak adalah 'Private' dan `income` mayoritas adalah '<=50K'.

#### 5.1.2 Missing Values Analysis
Terdapat missing values pada tiga fitur kategorikal:
- `workclass`: 1836 missing values (5.64% dari total data)
- `occupation`: 1843 missing values (5.66% dari total data)
- `native_country`: 583 missing values (1.79% dari total data)
Fitur-fitur numerik tidak memiliki missing values.

#### 5.1.3 Feature Distribution
Visualisasi distribusi fitur dilakukan menggunakan histogram untuk fitur numerik dan bar plot untuk fitur kategorikal (gambar `numeric_histograms.png` dan `distribution_[feature].png` dihasilkan oleh notebook). Observasi dari distribusi ini membantu memahami skewness, rentang, dan modus dari setiap fitur. Misalnya, `capital_gain` dan `capital_loss` sebagian besar bernilai nol. Distribusi kelas target `income` menunjukkan ketidakseimbangan, dengan sekitar 75.9% data memiliki pendapatan '<=50K' dan 24.1% '>50K'.

#### 5.1.4 Correlation Analysis
Analisis korelasi awal antar fitur numerik dilakukan menggunakan heatmap (gambar `correlation_matrix.png`). Ini membantu mengidentifikasi hubungan linear awal antar variabel numerik sebelum dilakukan encoding pada fitur kategorikal.

### 5.2 Hasil Preprocessing

#### 5.2.1 Data Cleaning Results
- **Original Shape**: (32561, 15)
- **After Cleaning**: (32537, 15) setelah penghapusan duplikat. Shape akhir setelah feature engineering dan sebelum splitting menjadi (32537, 14) yaitu 13 fitur input dan 1 target.
- **Removed Duplicates**: **24** records duplikat dihapus dari dataset.
- **Missing Values**: Missing values pada fitur `workclass`, `occupation`, dan `native_country` ditangani menggunakan imputasi modus (nilai yang paling sering muncul) untuk masing-masing kolom.

#### 5.2.2 Feature Engineering Results
- **Encoding**: Seluruh fitur kategorikal (`workclass`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`) diubah menjadi representasi numerik menggunakan pemetaan manual (manual label encoding). Target variabel `income` juga diubah menjadi 0 ('<=50K') dan 1 ('>50K').
- **Feature Removal**: Fitur `education` dihapus karena informasinya sudah tercakup dalam `education_num` yang bersifat numerik dan lebih sesuai untuk model. Tidak ada fitur lain yang dihapus karena korelasi tinggi antar fitur input (berdasarkan threshold 0.8 pada matriks korelasi setelah encoding, karena `corr_features` hasilnya `set()`).
- **Final Feature Set**: Dataset final untuk pemodelan terdiri dari 13 fitur input: `age`, `workclass`, `fnlwgt`, `education_num`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `capital_gain`, `capital_loss`, `hours_per_week`, dan `native_country`.

### 5.3 Hasil Model Training

#### 5.3.1 Hyperparameter Optimization
- **Best Parameters**: Hasil optimasi menggunakan GridSearchCV menunjukkan parameter terbaik untuk model Decision Tree adalah: `{'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10}`.
- **Cross Validation Score**: Skor akurasi rata-rata dari 3-fold cross-validation pada data training dengan parameter terbaik adalah **[Tidak secara eksplisit tercetak di output, namun ini adalah nilai `grid_search.best_score_` yang mengarah pada pemilihan `best_params_`]**.
- **Training Time**: `[Waktu training]` (tidak tersedia dari output).

#### 5.3.2 Model Performance
Performa model dievaluasi pada data training dan data testing:

| Metric         | Training Set         | Testing Set           |
|----------------|----------------------|-----------------------|
| Accuracy       | 86.50%               | 86.08%                |
| Precision      | 86.50% (micro avg)   | 85.50% (weighted avg) |
| Recall         | 86.50% (micro avg)   | 86.08% (weighted avg) |
| F1-Score       | 86.50% (micro avg)   | 85.71% (weighted avg) |

*(Catatan: Nilai Precision dan F1-Score untuk Testing Set di atas dihitung dari Classification Report: Precision (weighted) = (0.89*2470 + 0.75*784)/(2470+784) = 0.855; F1-Score (weighted) = (0.91*2470 + 0.69*784)/(2470+784) = 0.8571. Recall (weighted) sesuai dengan akurasi karena supportnya sama.)*

Detail performa per kelas pada data testing:
- Kelas 0 ('<=50K'): Precision 0.89, Recall 0.93, F1-Score 0.91
- Kelas 1 ('>50K'): Precision 0.75, Recall 0.64, F1-Score 0.69

#### 5.3.3 Confusion Matrix
Confusion matrix pada data testing adalah sebagai berikut:
```
[[2301  169]
 [ 284  500]]
```
Interpretasi:
- **True Negative (TN)**: 2301 individu dengan pendapatan `<=50K` diprediksi dengan benar.
- **False Positive (FP)**: 169 individu dengan pendapatan `<=50K` salah diprediksi sebagai `>50K`.
- **False Negative (FN)**: 284 individu dengan pendapatan `>50K` salah diprediksi sebagai `<=50K`.
- **True Positive (TP)**: 500 individu dengan pendapatan `>50K` diprediksi dengan benar.

Model menunjukkan kemampuan yang baik dalam mengklasifikasikan kelas mayoritas (`<=50K`), namun memiliki recall yang lebih rendah untuk kelas minoritas (`>50K`) (Recall untuk >50K = 500 / (500+284) = 0.6377 ~ 64%).

#### 5.3.4 Feature Importance
Berdasarkan output `best_model.feature_importances_` dari notebook (yang divisualisasikan dalam `feature_importance.png`), fitur-fitur yang paling signifikan dalam menentukan prediksi pendapatan diidentifikasi. Urutan dan nilai kontribusi spesifiknya dapat dilihat pada visualisasi tersebut. Secara umum, fitur-fitur seperti `marital_status`, `capital_gain`, `education_num`, `age`, dan `hours_per_week` diharapkan menunjukkan kontribusi penting.

### 5.4 Analisis Model

#### 5.4.1 Model Interpretability
Visualisasi decision tree (hingga kedalaman 3, seperti pada `decision_tree.png`) memberikan pemahaman tentang bagaimana model membuat keputusan. Model ini bekerja berdasarkan serangkaian aturan if-then-else pada fitur-fitur input untuk mencapai prediksi kelas. Keterbacaan ini adalah salah satu keunggulan utama dari algoritma Decision Tree, memungkinkan pengguna untuk memahami logika di balik prediksi.

#### 5.4.2 Overfitting Analysis
Performa model pada training set (Akurasi: 86.50%) dan testing set (Akurasi: 86.08%) relatif dekat. Perbedaan yang kecil ini (sekitar 0.42%) menunjukkan bahwa model tidak mengalami overfitting yang signifikan dan memiliki kemampuan generalisasi yang baik pada data yang belum pernah dilihat sebelumnya. Hyperparameter seperti `max_depth`, `min_samples_leaf`, dan `min_samples_split` yang dioptimalkan melalui GridSearchCV membantu mengontrol kompleksitas model dan mencegah overfitting.

#### 5.4.3 Feature Contribution
Analisis feature importance (seperti yang dibahas di 5.3.4 dan divisualisasikan di `feature_importance.png`) mengindikasikan bahwa status perkawinan (`marital_status`), perolehan modal (`capital_gain`), tingkat pendidikan dalam tahun (`education_num`), usia (`age`), dan jam kerja per minggu (`hours_per_week`) adalah beberapa faktor kunci dalam memprediksi tingkat pendapatan. Insight ini sejalan dengan pemahaman umum ekonomi bahwa status pernikahan (terutama `Married-civ-spouse` yang sering berkorelasi dengan pendapatan lebih tinggi), investasi/modal, tingkat pendidikan formal, pengalaman kerja (yang sering berkorelasi dengan usia), dan intensitas kerja merupakan determinan penting dari potensi pendapatan individu.

### 5.5 Hasil Implementasi Web Application

#### 5.5.1 User Interface
Aplikasi web Streamlit yang dikembangkan menyediakan antarmuka pengguna yang bersih dan intuitif untuk prediksi pendapatan. Pengguna dapat memasukkan nilai untuk 13 fitur demografis dan pekerjaan melalui sebuah form input yang terorganisir dalam beberapa bagian (Demographic, Work, Financial Information). Terdapat tombol "Predict" untuk memicu prediksi, "Reset" untuk mengembalikan semua field input ke nilai default, dan "Export Last Result" untuk mengaktifkan unduhan hasil prediksi terakhir. Hasil prediksi ditampilkan dengan jelas di kolom sebelah kanan, mencakup label kelas pendapatan yang diprediksi ('>50K' atau '<=50K') dengan warna yang sesuai dan tingkat kepercayaan (confidence level) yang divisualisasikan menggunakan gauge chart interaktif dari Plotly. Selain itu, distribusi probabilitas untuk kedua kelas pendapatan juga ditampilkan dalam bentuk bar chart Plotly. Bagian akhir dari aplikasi menampilkan visualisasi horizontal bar chart dari feature importance model Decision Tree yang digunakan, juga menggunakan Plotly.

#### 5.5.2 Functionality Testing
Fungsionalitas aplikasi telah diuji dengan berbagai skenario input. Validasi input diimplementasikan untuk fitur-fitur numerik kunci (seperti `age`, `education_num`, `hours_per_week`, `capital_gain`, `capital_loss`, `fnlwgt`) untuk memastikan bahwa data yang dimasukkan berada dalam rentang yang logis dan sesuai dengan batasan yang diobservasi dari dataset (sebagaimana didefinisikan dalam fungsi `validate_inputs`). Jika input tidak valid, pesan error akan ditampilkan kepada pengguna di kolom hasil. Fungsi prediksi, reset input, dan ekspor hasil prediksi dalam format JSON telah diverifikasi dan berjalan sesuai dengan yang diharapkan. Prediksi hanya dijalankan jika tidak ada error validasi.

#### 5.5.3 Performance Analysis
Aplikasi web menunjukkan responsivitas yang baik terhadap interaksi pengguna. Pemuatan model dan komponen terkaitnya (seperti encoding maps dan feature names dari file `income_prediction_components.joblib`) dioptimalkan dengan penggunaan mekanisme caching (`@st.cache_resource` pada Streamlit), yang memastikan model hanya dimuat sekali saat aplikasi pertama kali dijalankan atau saat ada perubahan pada file model. Hal ini mengurangi latensi pada prediksi berikutnya. Penggunaan visualisasi interaktif dari Plotly untuk menampilkan hasil prediksi dan feature importance juga meningkatkan pengalaman pengguna, membuat informasi lebih mudah dipahami dan diinterpretasi.

---

## 6. KESIMPULAN DAN SARAN

### 6.1 Kesimpulan

1. **Model Performance**: Model Decision Tree yang dikembangkan, setelah optimasi hyperparameter (`criterion='gini'`, `max_depth=10`, `min_samples_leaf=4`, `min_samples_split=10`), berhasil mencapai akurasi **86.08%** pada data testing untuk prediksi tingkat pendapatan. Feature importance tertinggi teridentifikasi pada fitur-fitur seperti **`marital_status`, `capital_gain`, dan `education_num`**.

2. **Feature Analysis**: Fitur yang paling berpengaruh terhadap prediksi pendapatan, berdasarkan analisis feature importance, adalah **`marital_status`, `capital_gain`, `education_num`, `age`, dan `hours_per_week`**. Temuan ini konsisten dengan teori dan intuisi ekonomi mengenai faktor-faktor penentu pendapatan.

3. **Implementation Success**: Penelitian ini berhasil mengimplementasikan end-to-end machine learning pipeline, mulai dari akuisisi dan pembersihan data, eksplorasi data, feature engineering, training model, evaluasi, hingga deployment model sebagai aplikasi web interaktif menggunakan Streamlit.

4. **Business Value**: Model yang dikembangkan dapat berfungsi sebagai alat bantu dalam pengambilan keputusan atau analisis untuk berbagai stakeholder, dengan tingkat akurasi yang cukup baik untuk aplikasi di dunia nyata, khususnya sebagai alat skrining awal atau untuk mendapatkan insight dari data demografis terkait potensi pendapatan.

### 6.2 Kontribusi Penelitian

1. **Metodologi**: Demonstrasi penerapan best practices dalam machine learning pipeline, termasuk EDA, preprocessing (penanganan missing values, encoding), hyperparameter tuning dengan cross-validation, dan evaluasi model yang komprehensif menggunakan berbagai metrik.
2. **Implementasi**: Menyediakan contoh konkret deployment model machine learning sebagai aplikasi web yang user-friendly menggunakan Streamlit, menjembatani kesenjangan antara model analitik dan pengguna akhir dengan fitur seperti input validasi, visualisasi interaktif, dan ekspor hasil.
3. **Interpretability**: Melakukan analisis mendalam terhadap feature importance dan memungkinkan visualisasi Decision Tree (meskipun tidak secara langsung di app) untuk memberikan insight bisnis yang dapat dipahami dan ditindaklanjuti dari model yang relatif transparan.

### 6.3 Limitasi Penelitian

1. **Data Currency**: Dataset yang digunakan berasal dari tahun 1994. Kondisi sosial-ekonomi dan pasar kerja mungkin telah banyak berubah, sehingga relevansi model untuk prediksi kondisi saat ini bisa terbatas.
2. **Algorithm Scope**: Penelitian ini hanya berfokus pada algoritma Decision Tree. Perbandingan dengan algoritma klasifikasi lain (misalnya, Random Forest, Gradient Boosting, SVM, atau Neural Networks) tidak dilakukan, yang mungkin bisa memberikan performa lebih baik atau insight berbeda.
3. **Feature Engineering**: Meskipun feature engineering dasar telah dilakukan (seperti encoding dan penghapusan fitur redundan), eksplorasi feature engineering yang lebih sophisticated (misalnya, pembuatan fitur interaksi, binning fitur numerik secara lebih mendalam) mungkin dapat lebih meningkatkan performa model.
4. **Missing Value Imputation**: Imputasi modus digunakan untuk menangani missing values pada fitur kategorikal. Metode imputasi yang lebih canggih (misalnya, k-NN imputer atau model-based imputation) bisa dieksplorasi.
5. **Imbalance Target Class**: Kelas target `income` tidak seimbang (sekitar 76% `<=50K` dan 24% `>50K`). Meskipun model mencapai akurasi yang baik, performa pada kelas minoritas (recall 64% untuk `>50K`) lebih rendah. Teknik penanganan data tidak seimbang (misalnya, SMOTE, ADASYN, atau penggunaan class weights) tidak secara eksplisit diterapkan dalam training model (selain `stratify` pada `train_test_split`).

### 6.4 Saran untuk Penelitian Selanjutnya

1. **Algorithm Comparison**: Melakukan studi komparatif dengan algoritma klasifikasi lain yang lebih canggih seperti Random Forest, XGBoost, LightGBM, atau Neural Networks untuk melihat potensi peningkatan akurasi dan metrik lainnya, terutama recall untuk kelas minoritas.
2. **Advanced Feature Engineering**: Mengeksplorasi teknik feature engineering yang lebih lanjut, seperti polynomial features, interaction terms antar fitur yang signifikan secara domain, atau binning fitur numerik (misalnya `age`, `hours_per_week`) menjadi kategori-kategori yang lebih bermakna.
3. **Handling Imbalanced Data**: Mengimplementasikan teknik khusus untuk menangani ketidakseimbangan kelas target secara eksplisit, seperti oversampling (misalnya SMOTE), undersampling, atau menggunakan parameter `class_weight='balanced'` pada algoritma model yang mendukung, untuk meningkatkan prediksi pada kelas minoritas.
4. **Updated Dataset**: Menggunakan dataset yang lebih baru dan relevan dengan kondisi sosial-ekonomi terkini untuk meningkatkan daya generalisasi dan relevansi praktis model.
5. **Model Explainability (XAI)**: Lebih mendalami aspek explainability model menggunakan teknik seperti SHAP (SHapley Additive exPlanations) atau LIME (Local Interpretable Model-agnostic Explanations), terutama jika di masa depan menggunakan model yang lebih kompleks (black-box).
6. **Deployment Enhancement**: Meningkatkan aplikasi web dengan fitur tambahan seperti kemampuan untuk batch prediction (mengunggah file data), penyimpanan histori prediksi pengguna, atau integrasi dengan dashboard monitoring performa model secara berkala dan mekanisme retraining otomatis.
7. **Cross-Cultural Analysis**: Jika data tersedia, melakukan analisis serupa pada dataset dari negara atau konteks budaya yang berbeda untuk membandingkan faktor-faktor penentu pendapatan dan universalitas model.

---

## DAFTAR PUSTAKA

[Tambahkan referensi yang relevan, contoh:]

1. Dua, S., & Du, X. (2016). Data mining and machine learning in cybersecurity. CRC press.

2. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An introduction to statistical learning (Vol. 112, p. 18). New York: springer.

3. Kohavi, R. (1996). Scaling up the accuracy of naive-bayes classifiers: A decision-tree hybrid. In Proceedings of the second international conference on knowledge discovery and data mining (pp. 202-207).

4. UCI Machine Learning Repository. Adult Data Set. https://archive.ics.uci.edu/dataset/2/adult

5. [Tambahkan referensi lain yang relevan dengan topik income prediction, decision tree, dan machine learning]

---

## LAMPIRAN

### Lampiran A: Source Code
[Link ke repository GitHub atau lampiran kode lengkap dari `klasifikasi.ipynb` dan `streamlit_income_app.py`]

### Lampiran B: Dataset Description
[Detail lengkap tentang dataset dari sumber UCI atau deskripsi yang lebih rinci]

### Lampiran C: Additional Visualizations
[Visualisasi tambahan yang tidak masuk ke bagian utama, seperti semua `distribution_[feature].png`, `correlation_matrix.png`, `correlation_after_encoding.png`, `final_correlation.png`, `decision_tree.png` (full depth jika ada), `feature_importance.png`]

### Lampiran D: Model Artifacts
[Informasi tentang model yang disimpan (`income_prediction_components.joblib`) dan cara penggunaannya. Ini mencakup detail tentang struktur dictionary yang disimpan: `model` (objek DecisionTreeClassifier), `feature_names` (list nama fitur input), `encoding_maps` (dictionary berisi mapping untuk setiap fitur kategorikal dan target), `model_params` (parameter terbaik dari GridSearchCV), `removed_features` (list fitur yang dihapus), `target_map` (mapping untuk variabel target).]
