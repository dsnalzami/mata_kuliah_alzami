"""
Streamlit App untuk Topic Modeling LDA Bahasa Indonesia

run: streamlit run lda_streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import re
import pickle
import base64
from io import BytesIO

# NLP processing
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
import streamlit.components.v1 as components

# Memastikan direktori output ada
if not os.path.exists('lda_output'):
    st.error("Direktori 'lda_output' tidak ditemukan. Pastikan model sudah dilatih.")
    st.stop()

# Fungsi untuk preproses teks - sama dengan yang di topic_azm.py
def preprocess_text(text):
    """
    Preprocessing teks:
    - Mengubah ke lowercase
    - Menghapus emoji dan karakter khusus
    - Menghapus angka
    - Menghapus URL
    """
    if isinstance(text, str):
        # Lowercase
        text = text.lower()
        
        # Hapus URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Hapus emoji dan karakter khusus
        text = re.sub(r'[^\w\s]', '', text)
        
        # Hapus angka
        text = re.sub(r'\d+', '', text)
        
        # Hapus multiple whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    return ""

# Daftar stopwords Bahasa Indonesia
def get_stopwords():
    """
    Daftar stopwords Bahasa Indonesia
    """
    stopwords = [
        'ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir', 
        'akhiri', 'akhirnya', 'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara',
        'antaranya', 'apa', 'apaan', 'apabila', 'apakah', 'apalagi', 'apatah', 'artinya', 'asal',
        'asalkan', 'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya', 'bagai', 'bagaikan',
        'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasanya',
        'baik', 'bakal', 'bakalan', 'balik', 'banyak', 'bapak', 'baru', 'bawah', 'beberapa', 'begini',
        'beginian', 'beginikah', 'beginilah', 'begitu', 'begitukah', 'begitulah', 'begitupun', 'bekerja',
        'belakang', 'belakangan', 'belum', 'belumlah', 'benar', 'benarkah', 'benarlah', 'berada',
        'berakhir', 'berakhirlah', 'berakhirnya', 'berapa', 'berapakah', 'berapalah', 'berapapun',
        'berarti', 'berawal', 'berbagai', 'berdatangan', 'beri', 'berikan', 'berikut', 'berikutnya',
        'berjumlah', 'berkali-kali', 'berkata', 'berkehendak', 'berkeinginan', 'berkenaan', 'berlainan',
        'berlalu', 'berlangsung', 'berlebihan', 'bermacam', 'bermacam-macam', 'bermaksud', 'bermula',
        'bersama', 'bersama-sama', 'bersiap', 'bersiap-siap', 'bertanya', 'bertanya-tanya', 'berturut',
        'berturut-turut', 'bertutur', 'berujar', 'berupa', 'besar', 'betul', 'betulkah', 'biasa',
        'biasanya', 'bila', 'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah', 'bolehlah', 'buat',
        'bukan', 'bukankah', 'bukanlah', 'bukannya', 'bulan', 'bung', 'cara', 'caranya', 'cukup',
        'cukupkah', 'cukuplah', 'cuma', 'dahulu', 'dalam', 'dan', 'dapat', 'dari', 'daripada',
        'datang', 'dekat', 'demi', 'demikian', 'demikianlah', 'dengan', 'depan', 'di', 'dia',
        'diakhiri', 'diakhirinya', 'dialah', 'diantara', 'diantaranya', 'diberi', 'diberikan',
        'diberikannya', 'dibuat', 'dibuatnya', 'didapat', 'didatangkan', 'digunakan', 'diibaratkan',
        'diibaratkannya', 'diingat', 'diingatkan', 'diinginkan', 'dijawab', 'dijelaskan', 'dijelaskannya',
        'dikarenakan', 'dikatakan', 'dikatakannya', 'dikerjakan', 'diketahui', 'diketahuinya',
        'dikira', 'dilakukan', 'dilalui', 'dilihat', 'dimaksud', 'dimaksudkan', 'dimaksudkannya',
        'dimaksudnya', 'diminta', 'dimintai', 'dimisalkan', 'dimulai', 'dimulailah', 'dimulainya',
        'dimungkinkan', 'dini', 'dipastikan', 'diperbuat', 'diperbuatnya', 'dipergunakan',
        'diperkirakan', 'diperlihatkan', 'diperlukan', 'diperlukannya', 'dipersoalkan',
        'dipertanyakan', 'dipunyai', 'diri', 'dirinya', 'disampaikan', 'disebut', 'disebutkan',
        'disebutkannya', 'disini', 'disinilah', 'ditambahkan', 'ditandaskan', 'ditanya', 'ditanyai',
        'ditanyakan', 'ditegaskan', 'ditujukan', 'ditunjuk', 'ditunjuki', 'ditunjukkan',
        'ditunjukkannya', 'ditunjuknya', 'dituturkan', 'dituturkannya', 'diucapkan', 'diucapkannya',
        'diungkapkan', 'dong', 'dua', 'dulu', 'empat', 'enggak', 'enggaknya', 'entah', 'entahlah',
        'guna', 'gunakan', 'hal', 'hampir', 'hanya', 'hanyalah', 'hari', 'harus', 'haruslah',
        'harusnya', 'hendak', 'hendaklah', 'hendaknya', 'hingga', 'ia', 'ialah', 'ibarat',
        'ibaratkan', 'ibaratnya', 'ibu', 'ikut', 'ingat', 'ingat-ingat', 'ingin', 'inginkah',
        'inginkan', 'ini', 'inikah', 'inilah', 'itu', 'itukah', 'itulah', 'jadi', 'jadilah',
        'jadinya', 'jangan', 'jangankan', 'janganlah', 'jauh', 'jawab', 'jawaban', 'jawabnya',
        'jelas', 'jelaskan', 'jelaslah', 'jelasnya', 'jika', 'jikalau', 'juga', 'jumlah',
        'jumlahnya', 'justru', 'kala', 'kalau', 'kalaulah', 'kalaupun', 'kalian', 'kami',
        'kamilah', 'kamu', 'kamulah', 'kan', 'kapan', 'kapankah', 'kapanpun', 'karena',
        'karenanya', 'kasus', 'kata', 'katakan', 'katakanlah', 'katanya', 'ke', 'keadaan',
        'kebetulan', 'kecil', 'kedua', 'keduanya', 'keinginan', 'kelamaan', 'kelihatan',
        'kelihatannya', 'kelima', 'keluar', 'kembali', 'kemudian', 'kemungkinan', 'kemungkinannya',
        'kenapa', 'kepada', 'kepadanya', 'kesamaannya', 'keseluruhan', 'keseluruhannya',
        'keterlaluan', 'ketika', 'khususnya', 'kini', 'kinilah', 'kira', 'kira-kira', 'kiranya',
        'kita', 'kitalah', 'kok', 'kurang', 'lagi', 'lagian', 'lah', 'lain', 'lainnya', 'lalu',
        'lama', 'lamanya', 'lanjut', 'lanjutnya', 'lebih', 'lewat', 'lima', 'luar', 'macam',
        'maka', 'makanya', 'makin', 'malah', 'malahan', 'mampu', 'mampukah', 'mana', 'manakala',
        'manalagi', 'masa', 'masalah', 'masalahnya', 'masih', 'masihkah', 'masing', 'masing-masing',
        'mau', 'maupun', 'melainkan', 'melakukan', 'melalui', 'melihat', 'melihatnya', 'memang',
        'memastikan', 'memberi', 'memberikan', 'membuat', 'memerlukan', 'memihak', 'meminta',
        'memintakan', 'memisalkan', 'memperbuat', 'mempergunakan', 'memperkirakan', 'memperlihatkan',
        'mempersiapkan', 'mempersoalkan', 'mempertanyakan', 'mempunyai', 'memulai', 'memungkinkan',
        'menaiki', 'menambahkan', 'menandaskan', 'menanti', 'menanti-nanti', 'menantikan',
        'menanya', 'menanyai', 'menanyakan', 'mendapat', 'mendapatkan', 'mendatang', 'mendatangi',
        'mendatangkan', 'menegaskan', 'mengakhiri', 'mengapa', 'mengatakan', 'mengatakannya',
        'mengenai', 'mengerjakan', 'mengetahui', 'menggunakan', 'menghendaki', 'mengibaratkan',
        'mengibaratkannya', 'mengingat', 'mengingatkan', 'menginginkan', 'mengira', 'mengucapkan',
        'mengucapkannya', 'mengungkapkan', 'menjadi', 'menjawab', 'menjelaskan', 'menuju',
        'menunjuk', 'menunjuki', 'menunjukkan', 'menunjuknya', 'menurut', 'menuturkan',
        'menyampaikan', 'menyangkut', 'menyatakan', 'menyebutkan', 'menyeluruh', 'menyiapkan',
        'merasa', 'mereka', 'merekalah', 'merupakan', 'meski', 'meskipun', 'meyakini', 'meyakinkan',
        'minta', 'mirip', 'misal', 'misalkan', 'misalnya', 'mula', 'mulai', 'mulailah', 'mulanya',
        'mungkin', 'mungkinkah', 'nah', 'naik', 'namun', 'nanti', 'nantinya', 'nyaris', 'nyatanya',
        'oleh', 'olehnya', 'pada', 'padahal', 'padanya', 'pak', 'paling', 'panjang', 'pantas',
        'para', 'pasti', 'pastilah', 'penting', 'pentingnya', 'per', 'percuma', 'perlu', 'perlukah',
        'perlunya', 'pernah', 'persoalan', 'pertama', 'pertama-tama', 'pertanyaan', 'pertanyakan',
        'pihak', 'pihaknya', 'pukul', 'pula', 'pun', 'punya', 'rasa', 'rasanya', 'rata', 'rupanya',
        'saat', 'saatnya', 'saja', 'sajalah', 'saling', 'sama', 'sama-sama', 'sambil', 'sampai',
        'sampai-sampai', 'sampaikan', 'sana', 'sangat', 'sangatlah', 'satu', 'saya', 'sayalah',
        'se', 'sebab', 'sebabnya', 'sebagai', 'sebagaimana', 'sebagainya', 'sebagian', 'sebaik',
        'sebaik-baiknya', 'sebaiknya', 'sebaliknya', 'sebanyak', 'sebegini', 'sebegitu', 'sebelum',
        'sebelumnya', 'sebenarnya', 'seberapa', 'sebesar', 'sebetulnya', 'sebisanya', 'sebuah',
        'sebut', 'sebutlah', 'sebutnya', 'secara', 'secukupnya', 'sedang', 'sedangkan', 'sedemikian',
        'sedikit', 'sedikitnya', 'seenaknya', 'segala', 'segalanya', 'segera', 'seharusnya',
        'sehingga', 'seingat', 'sejak', 'sejauh', 'sejenak', 'sejumlah', 'sekadar', 'sekadarnya',
        'sekali', 'sekali-kali', 'sekalian', 'sekaligus', 'sekalipun', 'sekarang', 'sekarang',
        'sekecil', 'seketika', 'sekiranya', 'sekitar', 'sekitarnya', 'sekurang-kurangnya',
        'sekurangnya', 'sela', 'selain', 'selaku', 'selalu', 'selama', 'selama-lamanya',
        'selamanya', 'selanjutnya', 'seluruh', 'seluruhnya', 'semacam', 'semakin', 'semampu',
        'semampunya', 'semasa', 'semasih', 'semata', 'semata-mata', 'semaunya', 'sementara',
        'semisal', 'semisalnya', 'sempat', 'semua', 'semuanya', 'semula', 'sendiri', 'sendirian',
        'sendirinya', 'seolah', 'seolah-olah', 'seorang', 'sepanjang', 'sepantasnya', 'sepantasnyalah',
        'seperlunya', 'seperti', 'sepertinya', 'sepihak', 'sering', 'seringnya', 'serta',
        'serupa', 'sesaat', 'sesama', 'sesampai', 'sesegera', 'sesekali', 'seseorang',
        'sesuatu', 'sesuatunya', 'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah',
        'seterusnya', 'setiap', 'setiba', 'setibanya', 'setidak-tidaknya', 'setidaknya',
        'setinggi', 'seusai', 'sewaktu', 'siap', 'siapa', 'siapakah', 'siapapun', 'sini',
        'sinilah', 'soal', 'soalnya', 'suatu', 'sudah', 'sudahkah', 'sudahlah', 'supaya',
        'tadi', 'tadinya', 'tahu', 'tahun', 'tak', 'tambah', 'tambahnya', 'tampak', 'tampaknya',
        'tandas', 'tandasnya', 'tanpa', 'tanya', 'tanyakan', 'tanyanya', 'tapi', 'tegas',
        'tegasnya', 'telah', 'tempat', 'tengah', 'tentang', 'tentu', 'tentulah', 'tentunya',
        'tepat', 'terakhir', 'terasa', 'terbanyak', 'terdahulu', 'terdapat', 'terdiri',
        'terhadap', 'terhadapnya', 'teringat', 'teringat-ingat', 'terjadi', 'terjadilah',
        'terjadinya', 'terkira', 'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata',
        'tersampaikan', 'tersebut', 'tersebutlah', 'tertentu', 'tertuju', 'terus', 'terutama',
        'tetap', 'tetapi', 'tiap', 'tiba', 'tiba-tiba', 'tidak', 'tidakkah', 'tidaklah',
        'tiga', 'tinggi', 'toh', 'tunjuk', 'turut', 'tutur', 'tuturnya', 'ucap', 'ucapnya',
        'ujar', 'ujarnya', 'umum', 'umumnya', 'ungkap', 'ungkapnya', 'untuk', 'usah',
        'usai', 'waduh', 'wah', 'wahai', 'waktu', 'waktunya', 'walau', 'walaupun', 'wong',
        'yaitu', 'yakin', 'yakni', 'yang', 'yg'
    ]
    
    # Tambahkan stopwords khusus untuk dataset (dari analisis awal)
    additional_stopwords = ['ya', 'nya', 'yg', 'aja', 'gw', 'ga', 'udah', 'kek', 'gak', 'nya',
                           'nih', 'sih', 'kalo', 'eh', 'aku', 'kau', 'lo', 'lu', 'kan', 'kok',
                           'juga', 'udh', 'eh', 'ah', 'kk', 'wkwk', 'wkwkwk', 'haha', 'hahaha', 
                           'bang', 'banget', 'emang']
    
    stopwords.extend(additional_stopwords)
    return set(stopwords)

# Stemming
def get_stemmer():
    """
    Mendapatkan stemmer bahasa Indonesia dari Sastrawi
    """
    factory = StemmerFactory()
    return factory.create_stemmer()

# Fungsi untuk melakukan preprocessing pada teks baru
def preprocess_new_text(text):
    """
    Preprocessing teks baru untuk diprediksi
    """
    # Preprocessing teks
    cleaned_text = preprocess_text(text)
    
    # Tokenisasi
    tokens = word_tokenize(cleaned_text)
    
    # Hapus stopwords
    stopwords = get_stopwords()
    tokens_without_stopwords = [word for word in tokens if word not in stopwords and len(word) > 2]
    
    # Stemming
    stemmer = get_stemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens_without_stopwords]
    
    return cleaned_text, stemmed_tokens

# Fungsi untuk memprediksi topik dari teks baru
def predict_topic(text, lda_model, id2word):
    """
    Memprediksi topik untuk teks baru
    """
    cleaned_text, stemmed_tokens = preprocess_new_text(text)
    
    # Buat bow untuk teks baru
    bow = id2word.doc2bow(stemmed_tokens)
    
    # Dapatkan distribusi topik
    topic_distribution = lda_model.get_document_topics(bow)
    
    # Urutkan berdasarkan probabilitas
    topic_distribution = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
    
    return topic_distribution, cleaned_text, stemmed_tokens

# Fungsi untuk menampilkan visualisasi LDA menggunakan pyLDAvis
def generate_pyldavis(lda_model, corpus, id2word):
    """
    Membuat visualisasi LDA menggunakan pyLDAvis
    """
    # Buat visualisasi
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds='mmds')
    
    # Simpan sebagai HTML
    html_string = pyLDAvis.prepared_data_to_html(vis_data)
    
    return html_string

# Fungsi untuk mencari dokumen berdasarkan topik
def search_documents_by_topic(topic_id, df_topic_docs, top_n=10):
    """
    Mencari dokumen berdasarkan topik tertentu
    """
    # Filter dokumen dengan topik dominan yang sesuai
    filtered_docs = df_topic_docs[df_topic_docs['Dominant_Topic'] == topic_id].sort_values(
        by='Topic_Perc_Contrib', ascending=False
    )
    
    # Ambil top N dokumen
    top_docs = filtered_docs.head(top_n)
    
    return top_docs

# Fungsi untuk plotting distribusi topik
def plot_topic_distribution(topic_counts):
    """
    Membuat plot distribusi topik
    """
    fig = px.bar(
        topic_counts, 
        x='Topic_Num', 
        y='Count',
        labels={'Topic_Num': 'Nomor Topik', 'Count': 'Jumlah Dokumen'},
        title='Distribusi Topik Dominan'
    )
    
    fig.update_layout(xaxis=dict(tickmode='linear'))
    
    return fig

# Fungsi download dataframe sebagai CSV
def download_csv(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Fungsi untuk memuat model dan data
@st.cache_resource
def load_model_and_data():
    """
    Memuat model dan data dari direktori output
    """
    # Muat model LDA
    lda_model = LdaModel.load('lda_output/lda_model')
    
    # Muat dictionary
    id2word = corpora.Dictionary.load('lda_output/id2word.dictionary')
    
    # Muat corpus
    corpus = corpora.MmCorpus('lda_output/corpus.mm')
    
    # Muat informasi topik
    if os.path.exists('lda_output/topics_keywords.csv'):
        topics_df = pd.read_csv('lda_output/topics_keywords.csv')
    else:
        # Jika file tidak ada, buat dari model
        topics = lda_model.print_topics()
        topics_df = pd.DataFrame({
            'topic_id': [i for i, _ in topics],
            'keywords': [keywords for _, keywords in topics]
        })
    
    # Muat distribusi topik
    if os.path.exists('lda_output/topic_distribution.csv'):
        topic_dist_df = pd.read_csv('lda_output/topic_distribution.csv')
    else:
        topic_dist_df = None
    
    # Muat dokumen dengan topik
    if os.path.exists('lda_output/document_topics.csv'):
        doc_topics_df = pd.read_csv('lda_output/document_topics.csv')
    else:
        doc_topics_df = None
    
    # Muat coherence scores
    if os.path.exists('lda_output/coherence_scores.csv'):
        coherence_df = pd.read_csv('lda_output/coherence_scores.csv')
    else:
        coherence_df = None
    
    return lda_model, id2word, corpus, topics_df, topic_dist_df, doc_topics_df, coherence_df

# Aplikasi Streamlit
def main():
    # Konfigurasi halaman
    st.set_page_config(
        page_title="LDA Topic Modeling - Bahasa Indonesia",
        page_icon="📊",
        layout="wide"
    )
    
    # Judul aplikasi
    st.title("📊 Topic Modeling LDA untuk Teks Bahasa Indonesia")
    st.write("Aplikasi ini menggunakan model LDA untuk menganalisis dan menemukan topik dalam teks bahasa Indonesia.")
    
    # Muat model dan data
    try:
        lda_model, id2word, corpus, topics_df, topic_dist_df, doc_topics_df, coherence_df = load_model_and_data()
        
        # Sidebar untuk menu navigasi
        menu = st.sidebar.radio(
            "Navigasi", 
            ["Dashboard", "Analisis Teks Baru", "Visualisasi LDA", "Cari Dokumen"]
        )
        
        # Halaman Dashboard
        if menu == "Dashboard":
            st.header("Dashboard Topik")
            
            # Tampilkan informasi model
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Informasi Model")
                st.write(f"• Jumlah Topik: {lda_model.num_topics}")
                st.write(f"• Jumlah Dokumen: {len(corpus)}")
                st.write(f"• Ukuran Kosakata: {len(id2word)}")
                
                if coherence_df is not None:
                    max_coherence = coherence_df['coherence_score'].max()
                    optimal_topics = coherence_df.loc[coherence_df['coherence_score'].idxmax(), 'num_topics']
                    st.write(f"• Coherence Score Terbaik: {max_coherence:.4f}")
                    st.write(f"• Jumlah Topik Optimal: {optimal_topics}")
            
            with col2:
                # Tampilkan distribusi topik
                if topic_dist_df is not None:
                    st.subheader("Distribusi Topik")
                    fig = plot_topic_distribution(topic_dist_df)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tampilkan kata kunci topik
            st.subheader("Kata Kunci Topik")
            
            if topics_df is not None:
                # Buat tampilan yang lebih menarik untuk kata kunci topik
                cols = st.columns(min(3, len(topics_df)))
                
                for i, (_, row) in enumerate(topics_df.iterrows()):
                    col_idx = i % len(cols)
                    with cols[col_idx]:
                        st.write(f"**Topik {row['topic_id']}**")
                        
                        # Format kata kunci agar lebih mudah dibaca
                        keywords_str = row['keywords']
                        # Ekstrak kata-kata dan bobot
                        try:
                            keywords_list = re.findall(r'(\d+\.\d+)\*"([^"]+)"', keywords_str)
                            if keywords_list:
                                for weight, word in keywords_list[:5]:  # Ambil 5 kata kunci teratas
                                    st.write(f"- {word} ({float(weight):.3f})")
                            else:
                                st.write(keywords_str)
                        except:
                            st.write(keywords_str)
            
            # Tampilkan beberapa contoh dokumen dengan topiknya
            if doc_topics_df is not None:
                st.subheader("Contoh Dokumen dan Topik")
                
                # Pilih 10 contoh dokumen secara acak
                sample_docs = doc_topics_df.sample(min(10, len(doc_topics_df)))
                
                for i, row in enumerate(sample_docs.iterrows()):
                    _, doc = row
                    with st.expander(f"Dokumen {i+1} (Topik {int(doc['Dominant_Topic'])})"):
                        st.write(f"**Teks**: {doc['Text']}")
                        st.write(f"**Kontribusi topik**: {doc['Topic_Perc_Contrib']:.4f}")
                        st.write(f"**Kata kunci**: {doc['Keywords']}")
        
        # Halaman Analisis Teks Baru
        elif menu == "Analisis Teks Baru":
            st.header("Analisis Teks Baru")
            
            # Input teks
            text_input = st.text_area(
                "Masukkan teks yang ingin dianalisis:",
                height=200
            )
            
            if st.button("Cari Topik"):
                if text_input:
                    # Prediksi topik
                    topic_distribution, cleaned_text, stemmed_tokens = predict_topic(
                        text_input,
                        lda_model,
                        id2word
                    )
                    
                    # Tampilkan hasil
                    st.subheader("Hasil Analisis")
                    
                    # Distribusi topik
                    st.write("**Distribusi Topik:**")
                    
                    # Buat data untuk visualisasi
                    topic_data = []
                    for topic_id, prob in topic_distribution:
                        topic_data.append({
                            'Topik': f"Topik {topic_id}",
                            'Probabilitas': prob
                        })
                    
                    topic_df = pd.DataFrame(topic_data)
                    
                    # Plot distribusi
                    if not topic_df.empty:
                        fig = px.bar(
                            topic_df, 
                            x='Topik', 
                            y='Probabilitas',
                            title='Distribusi Probabilitas Topik',
                            labels={'Topik': 'Topik', 'Probabilitas': 'Probabilitas'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Topik Dominan
                    if topic_distribution:
                        dominant_topic_id, dominant_topic_prob = topic_distribution[0]
                        
                        st.write(f"**Topik Dominan:** Topik {dominant_topic_id} (Probabilitas: {dominant_topic_prob:.4f})")
                        
                        # Kata kunci topik dominan
                        keywords = lda_model.show_topic(dominant_topic_id, topn=10)
                        
                        st.write("**Kata Kunci Topik Dominan:**")
                        keyword_data = []
                        for word, prob in keywords:
                            keyword_data.append({
                                'Kata': word,
                                'Bobot': prob
                            })
                        
                        keyword_df = pd.DataFrame(keyword_data)
                        
                        # Plot kata kunci
                        fig = px.bar(
                            keyword_df, 
                            x='Kata', 
                            y='Bobot',
                            title=f'Kata Kunci Topik {dominant_topic_id}',
                            labels={'Kata': 'Kata', 'Bobot': 'Bobot'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Tampilkan hasil preprocessing
                    with st.expander("Lihat Hasil Preprocessing"):
                        st.write("**Teks yang dibersihkan:**")
                        st.write(cleaned_text)
                        
                        st.write("**Token setelah stemming:**")
                        st.write(stemmed_tokens)
                else:
                    st.warning("Silakan masukkan teks terlebih dahulu.")
        
        # Halaman Visualisasi LDA
        elif menu == "Visualisasi LDA":
            st.header("Visualisasi LDA Interaktif")
            
            # Generate visualisasi LDA
            st.write("Visualisasi LDA interaktif untuk eksplorasi model:")
            try:
                # Generate HTML
                html_string = generate_pyldavis(lda_model, corpus, id2word)
                
                # Tampilkan visualisasi
                components.html(html_string, width=1300, height=800)
            except Exception as e:
                st.error(f"Gagal membuat visualisasi LDA: {str(e)}")
                st.info("Visualisasi alternatif: Distribusi kata per topik")
                
                # Tampilkan kata kunci untuk setiap topik
                for i in range(lda_model.num_topics):
                    st.subheader(f"Topik {i}")
                    
                    # Dapatkan kata kunci
                    keywords = lda_model.show_topic(i, topn=20)
                    
                    # Buat dataframe
                    keyword_df = pd.DataFrame(keywords, columns=['Kata', 'Bobot'])
                    
                    # Plot
                    fig = px.bar(
                        keyword_df, 
                        x='Kata', 
                        y='Bobot',
                        title=f'Kata Kunci Topik {i}',
                        labels={'Kata': 'Kata', 'Bobot': 'Bobot'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tampilkan coherence scores jika tersedia
            if coherence_df is not None:
                st.subheader("Coherence Scores")
                
                fig = px.line(
                    coherence_df, 
                    x='num_topics', 
                    y='coherence_score',
                    markers=True,
                    title='Coherence Score untuk Berbagai Jumlah Topik',
                    labels={'num_topics': 'Jumlah Topik', 'coherence_score': 'Coherence Score'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Halaman Cari Dokumen
        elif menu == "Cari Dokumen":
            st.header("Cari Dokumen berdasarkan Topik")
            
            if doc_topics_df is not None:
                # Pilih topik
                topic_options = list(range(lda_model.num_topics))
                selected_topic = st.selectbox(
                    "Pilih Topik",
                    options=topic_options,
                    format_func=lambda x: f"Topik {x}"
                )
                
                # Pilih jumlah dokumen
                num_docs = st.slider(
                    "Jumlah Dokumen",
                    min_value=1,
                    max_value=min(50, len(doc_topics_df)),
                    value=10
                )
                
                if st.button("Cari Dokumen"):
                    # Cari dokumen
                    results = search_documents_by_topic(selected_topic, doc_topics_df, num_docs)
                    
                    if not results.empty:
                        st.subheader(f"Dokumen Terkait Topik {selected_topic}")
                        
                        # Tampilkan hasil
                        for i, row in enumerate(results.iterrows()):
                            _, doc = row
                            with st.expander(f"Dokumen {int(doc['Document_No'])} (Kontribusi: {doc['Topic_Perc_Contrib']:.4f})"):
                                st.write(f"**Teks**: {doc['Text']}")
                                st.write(f"**Kata kunci**: {doc['Keywords']}")
                        
                        # Download hasil
                        st.markdown(
                            download_csv(results, f'dokumen_topik_{selected_topic}.csv'),
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning(f"Tidak ada dokumen yang termasuk dalam Topik {selected_topic}.")
            else:
                st.warning("Data dokumen tidak tersedia. Pastikan file document_topics.csv ada di direktori lda_output.")
    
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model dan data: {str(e)}")
        st.info("Pastikan model LDA sudah dilatih dan tersimpan di direktori 'lda_output'.")

if __name__ == "__main__":
    main()