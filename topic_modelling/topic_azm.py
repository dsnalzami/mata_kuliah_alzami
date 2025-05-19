"""
Topic Modelling menggunakan LDA (Latent Dirichlet Allocation) untuk Teks Bahasa Indonesia
"""

# 1. Import library yang diperlukan
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from pprint import pprint

# NLP processing
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models

# Scikit-learn
from sklearn.feature_extraction.text import CountVectorizer

# Memastikan paket NLTK yang diperlukan sudah terunduh
# nltk.download('punkt')

# 2. Load dataset
def load_data(file_path):
    """
    Memuat data dari file CSV
    """
    df = pd.read_csv(file_path, sep=';')
    return df

# 3. Preprocessing text
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

# 4. Stopwords Bahasa Indonesia
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

# 5. Stemming
def get_stemmer():
    """
    Mendapatkan stemmer bahasa Indonesia dari Sastrawi
    """
    factory = StemmerFactory()
    return factory.create_stemmer()

# 6. Preprocessing langkah-langkah
def preprocess_data(df, text_column):
    """
    Preprocessing data teks
    """
    # Preprocessing teks
    df['clean_text'] = df[text_column].apply(preprocess_text)
    
    # Tokenisasi
    df['tokens'] = df['clean_text'].apply(word_tokenize)
    
    # Hapus stopwords
    stopwords = get_stopwords()
    df['tokens_without_stopwords'] = df['tokens'].apply(
        lambda tokens: [word for word in tokens if word not in stopwords and len(word) > 2]
    )
    
    # Stemming
    stemmer = get_stemmer()
    df['stemmed_tokens'] = df['tokens_without_stopwords'].apply(
        lambda tokens: [stemmer.stem(word) for word in tokens]
    )
    
    return df

# 7. Membuat korpus untuk LDA
def create_lda_corpus(df, tokens_column='stemmed_tokens'):
    """
    Membuat korpus untuk LDA:
    - Dictionary
    - Bag of Words corpus
    """
    # Buat dictionary
    id2word = corpora.Dictionary(df[tokens_column])
    
    # Buat corpus
    corpus = [id2word.doc2bow(text) for text in df[tokens_column]]
    
    return corpus, id2word

# 8. Membuat model LDA
def create_lda_model(corpus, id2word, num_topics=5, passes=10):
    """
    Membuat model LDA
    """
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=42,
        passes=passes,
        alpha='auto',
        eta='auto'
    )
    
    return lda_model

# 9. Mengevaluasi model dengan coherence score
def compute_coherence_score(model, corpus, texts, dictionary):
    """
    Menghitung coherence score
    """
    coherence_model = CoherenceModel(
        model=model, 
        texts=texts, 
        dictionary=dictionary, 
        coherence='c_v'
    )
    
    return coherence_model.get_coherence()

# 10. Mencari jumlah topik optimal
def find_optimal_topics(corpus, id2word, texts, start=2, limit=10, step=1):
    """
    Mencari jumlah topik optimal berdasarkan coherence score
    """
    coherence_values = []
    model_list = []
    
    for num_topics in range(start, limit, step):
        model = create_lda_model(corpus=corpus, id2word=id2word, num_topics=num_topics)
        model_list.append(model)
        
        coherence_model = CoherenceModel(
            model=model, 
            texts=texts, 
            dictionary=id2word, 
            coherence='c_v'
        )
        
        coherence_values.append(coherence_model.get_coherence())
    
    # Plot coherence score
    plt.figure(figsize=(12, 6))
    plt.plot(range(start, limit, step), coherence_values)
    plt.xlabel("Jumlah Topik")
    plt.ylabel("Coherence Score")
    plt.title("Coherence Score untuk Berbagai Jumlah Topik")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('coherence_score.png')
    
    # Temukan indeks dengan coherence score tertinggi
    max_index = coherence_values.index(max(coherence_values))
    optimal_topics = range(start, limit, step)[max_index]
    
    return model_list[max_index], optimal_topics, coherence_values, 'coherence_score.png'

# 11. Visualisasi model LDA
def visualize_lda(lda_model, corpus, id2word):
    """
    Visualisasi model LDA dengan pyLDAvis
    """
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, 'lda_visualization.html')
    
    return 'lda_visualization.html'

# 12. Format hasil model
def format_topics_sentences(ldamodel, corpus, texts):
    """
    Format hasil topic modeling
    """
    # Init output - buat list untuk menampung data
    topics_data = []

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                # Simpan data dalam list
                topics_data.append((int(topic_num), round(prop_topic, 4), topic_keywords))
            else:
                break
    
    # Buat DataFrame sekaligus dari list data
    sent_topics_df = pd.DataFrame(topics_data, columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])

    # Tambahkan teks asli
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    
    return sent_topics_df

# 13. Fungsi untuk menyimpan dan memuat hasil processing
def save_results(results, output_dir='output'):
    """
    Menyimpan semua hasil processing ke dalam file CSV dan model
    """
    import os
    
    # Buat direktori output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Simpan model LDA, dictionary, dan corpus
    results['model'].save(os.path.join(output_dir, 'lda_model'))
    results['id2word'].save(os.path.join(output_dir, 'id2word.dictionary'))
    corpora.MmCorpus.serialize(os.path.join(output_dir, 'corpus.mm'), results['corpus'])
    
    # Simpan dataframe hasil ke CSV
    results['df_topic_keywords'].to_csv(os.path.join(output_dir, 'topic_keywords.csv'), index=False)
    
    # Simpan informasi coherence score
    coherence_df = pd.DataFrame({
        'num_topics': list(range(2, 2 + len(results['coherence_values']))),
        'coherence_score': results['coherence_values']
    })
    coherence_df.to_csv(os.path.join(output_dir, 'coherence_scores.csv'), index=False)
    
    # Simpan dataframe preprocessing jika tersedia
    if 'df_processed' in results:
        results['df_processed'].to_csv(os.path.join(output_dir, 'preprocessed_data.csv'), index=False)
    
    print(f"Semua hasil telah disimpan di direktori: {output_dir}")
    
    return output_dir

def load_results(output_dir='output'):
    """
    Memuat hasil yang telah disimpan sebelumnya
    """
    import os
    
    results = {}
    
    # Muat model LDA, dictionary, dan corpus
    if os.path.exists(os.path.join(output_dir, 'lda_model')):
        results['model'] = gensim.models.ldamodel.LdaModel.load(os.path.join(output_dir, 'lda_model'))
    
    if os.path.exists(os.path.join(output_dir, 'id2word.dictionary')):
        results['id2word'] = corpora.Dictionary.load(os.path.join(output_dir, 'id2word.dictionary'))
    
    if os.path.exists(os.path.join(output_dir, 'corpus.mm')):
        results['corpus'] = corpora.MmCorpus(os.path.join(output_dir, 'corpus.mm'))
    
    # Muat dataframe hasil
    if os.path.exists(os.path.join(output_dir, 'topic_keywords.csv')):
        results['df_topic_keywords'] = pd.read_csv(os.path.join(output_dir, 'topic_keywords.csv'))
    
    # Muat informasi coherence score
    if os.path.exists(os.path.join(output_dir, 'coherence_scores.csv')):
        coherence_df = pd.read_csv(os.path.join(output_dir, 'coherence_scores.csv'))
        results['coherence_values'] = coherence_df['coherence_score'].tolist()
        results['optimal_topics'] = coherence_df.loc[coherence_df['coherence_score'].idxmax(), 'num_topics']
    
    # Muat dataframe preprocessing
    if os.path.exists(os.path.join(output_dir, 'preprocessed_data.csv')):
        results['df_processed'] = pd.read_csv(os.path.join(output_dir, 'preprocessed_data.csv'))
        
        # Rekonstruksi kolom list dari string
        if 'tokens' in results['df_processed'].columns:
            results['df_processed']['tokens'] = results['df_processed']['tokens'].apply(
                lambda x: eval(x) if isinstance(x, str) else []
            )
        
        if 'tokens_without_stopwords' in results['df_processed'].columns:
            results['df_processed']['tokens_without_stopwords'] = results['df_processed']['tokens_without_stopwords'].apply(
                lambda x: eval(x) if isinstance(x, str) else []
            )
            
        if 'stemmed_tokens' in results['df_processed'].columns:
            results['df_processed']['stemmed_tokens'] = results['df_processed']['stemmed_tokens'].apply(
                lambda x: eval(x) if isinstance(x, str) else []
            )
    
    print(f"Hasil berhasil dimuat dari direktori: {output_dir}")
    
    return results
def run_topic_modeling(file_path, text_column='text', save_intermediates=True, output_dir='output'):
    """
    Menjalankan topic modeling
    """
    print("Loading data...")
    df = load_data(file_path)
    
    print("Preprocessing data...")
    df_processed = preprocess_data(df, text_column)
    
    # Simpan hasil preprocessing jika diminta
    if save_intermediates:
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Konversi kolom list menjadi string untuk disimpan ke CSV
        df_to_save = df_processed.copy()
        for col in ['tokens', 'tokens_without_stopwords', 'stemmed_tokens']:
            if col in df_to_save.columns:
                df_to_save[col] = df_to_save[col].apply(str)
        
        df_to_save.to_csv(os.path.join(output_dir, 'preprocessed_data.csv'), index=False)
        print(f"Data preprocessing disimpan di {os.path.join(output_dir, 'preprocessed_data.csv')}")
    
    print("Creating corpus...")
    corpus, id2word = create_lda_corpus(df_processed)
    
    # Simpan corpus dan dictionary jika diminta
    if save_intermediates:
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        id2word.save(os.path.join(output_dir, 'id2word.dictionary'))
        corpora.MmCorpus.serialize(os.path.join(output_dir, 'corpus.mm'), corpus)
        print(f"Corpus dan dictionary disimpan di direktori {output_dir}")
    
    print("Finding optimal number of topics...")
    best_model, optimal_topics, coherence_values, coherence_plot = find_optimal_topics(
        corpus=corpus, 
        id2word=id2word, 
        texts=df_processed['stemmed_tokens']
    )
    
    print(f"Optimal number of topics: {optimal_topics}")
    print(f"Best coherence score: {max(coherence_values)}")
    
    # Simpan model jika diminta
    if save_intermediates:
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        best_model.save(os.path.join(output_dir, 'lda_model'))
        
        # Simpan coherence scores
        coherence_df = pd.DataFrame({
            'num_topics': list(range(2, 2 + len(coherence_values))),
            'coherence_score': coherence_values
        })
        coherence_df.to_csv(os.path.join(output_dir, 'coherence_scores.csv'), index=False)
        print(f"Model LDA dan coherence scores disimpan di direktori {output_dir}")
    
    print("Visualizing LDA model...")
    vis_path = visualize_lda(best_model, corpus, id2word)
    
    print("Topics and their keywords:")
    topics = best_model.print_topics()
    pprint(topics)
    
    # Simpan topik dan kata kunci ke CSV
    if save_intermediates:
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        topics_df = pd.DataFrame({
            'topic_id': [i for i, _ in topics],
            'keywords': [keywords for _, keywords in topics]
        })
        topics_df.to_csv(os.path.join(output_dir, 'topics_keywords.csv'), index=False)
        print(f"Topik dan kata kunci disimpan di {os.path.join(output_dir, 'topics_keywords.csv')}")
    
    # Format hasil
    df_topic_sents_keywords = format_topics_sentences(
        ldamodel=best_model, 
        corpus=corpus, 
        texts=df_processed['clean_text']
    )
    
    # Dominant topic distribution
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    
    # Simpan hasil pemetaan dokumen ke topik
    if save_intermediates:
        df_dominant_topic.to_csv(os.path.join(output_dir, 'document_topics.csv'), index=False)
        print(f"Pemetaan dokumen ke topik disimpan di {os.path.join(output_dir, 'document_topics.csv')}")
    
    print("\nDominant Topic Distribution:")
    topic_counts = df_dominant_topic['Dominant_Topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic_Num', 'Count']
    print(topic_counts)
    
    # Simpan distribusi topik
    if save_intermediates:
        topic_counts.to_csv(os.path.join(output_dir, 'topic_distribution.csv'), index=False)
        print(f"Distribusi topik disimpan di {os.path.join(output_dir, 'topic_distribution.csv')}")
    
    # Plot distribusi topik
    plt.figure(figsize=(10, 6))
    plt.bar(topic_counts['Topic_Num'], topic_counts['Count'])
    plt.xlabel('Topic Number')
    plt.ylabel('Count')
    plt.title('Distribution of Dominant Topics')
    plt.xticks(topic_counts['Topic_Num'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_distribution.png'))
    
    results = {
        'model': best_model,
        'corpus': corpus,
        'id2word': id2word,
        'optimal_topics': optimal_topics,
        'coherence_values': coherence_values,
        'coherence_plot': coherence_plot,
        'visualization': vis_path,
        'topic_distribution': os.path.join(output_dir, 'topic_distribution.png'),
        'df_topic_keywords': df_dominant_topic,
        'df_processed': df_processed
    }
    
    print(f"\nSemua hasil telah disimpan di direktori: {output_dir}")
    return results

if __name__ == "__main__":
    # Ganti dengan path file CSV yang sesuai
    results = run_topic_modeling("review.csv", output_dir='lda_output')
    print("Topic modeling completed successfully!")
    
    # Contoh cara memuat kembali hasil yang sudah disimpan
    # loaded_results = load_results('lda_output')
    # print(f"Loaded model with {loaded_results['optimal_topics']} topics")