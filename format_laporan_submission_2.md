# Laporan Proyek Machine Learning - Sistem Rekomendasi Film

---

## **Project Overview**

Industri hiburan digital, khususnya film, terus berkembang pesat seiring dengan meningkatnya jumlah produksi film dan kemudahan akses melalui platform daring seperti Netflix, Hulu, dan Disney+. Masyarakat kini memiliki ribuan pilihan tontonan dalam berbagai genre, bahasa, dan gaya sinematik yang dapat diakses kapan saja. Namun, banyaknya pilihan justru menimbulkan fenomena yang disebut information overload, di mana pengguna merasa kewalahan saat harus memilih film yang sesuai dengan preferensinya [1].

Untuk mengatasi permasalahan tersebut, sistem rekomendasi menjadi teknologi penting dalam membantu pengguna menjelajahi konten secara efisien. Sistem ini digunakan secara luas oleh berbagai platform digital untuk meningkatkan pengalaman pengguna, memperpanjang waktu interaksi, dan meningkatkan loyalitas pengguna [2]. Salah satu pendekatan populer dalam sistem rekomendasi adalah Content-Based Filtering, yaitu metode yang merekomendasikan item kepada pengguna berdasarkan kesamaan fitur konten dengan item yang pernah disukai sebelumnya.

Dalam proyek ini, dikembangkan sebuah sistem rekomendasi film berbasis Content-Based Filtering menggunakan dataset MovieLens [3]. Dataset ini banyak digunakan dalam penelitian sistem rekomendasi karena menyediakan data interaksi pengguna (rating) serta metadata film seperti genre, sutradara, dan tahun rilis. Dengan menggunakan informasi tersebut, sistem dibangun untuk mengidentifikasi karakteristik film yang disukai oleh pengguna dan mencari film lain dengan profil konten yang serupa.

Pendekatan Content-Based Filtering memiliki beberapa keunggulan, antara lain kemampuannya untuk memberikan rekomendasi secara personal tanpa harus bergantung pada preferensi pengguna lain. Metode ini juga cocok untuk mengatasi masalah cold start pada pengguna baru, karena hanya memerlukan data interaksi individu tersebut [4]. Namun demikian, pendekatan ini juga memiliki keterbatasan, seperti kecenderungan memberikan rekomendasi yang homogen atau mirip dengan yang telah ditonton sebelumnya (kurangnya diversitas) dan kesulitan menangani konten dengan informasi metadata yang minim.

Sistem ini dibangun dengan menggunakan teknik text vectorization dan similarity measurement seperti TF-IDF dan cosine similarity untuk menghitung tingkat kemiripan antar film berdasarkan fitur-fitur yang tersedia. Hasil rekomendasi kemudian dievaluasi secara kualitatif dan kuantitatif untuk memastikan bahwa sistem mampu memberikan saran yang relevan dan sesuai dengan kebutuhan pengguna [5].

Melalui proyek ini, diharapkan sistem rekomendasi yang dibangun dapat menjadi dasar implementasi lebih lanjut, misalnya integrasi dengan sistem hybrid atau penggunaan teknik NLP lanjutan untuk memperkaya pemrosesan konten. Dengan pendekatan Content-Based Filtering, pengguna akan terbantu dalam menemukan film-film yang sesuai dengan selera mereka, sekaligus mengurangi beban kognitif dalam proses eksplorasi tontonan.

---

## **Business Understanding**

### **Problem Statements**

- Pengguna sering mengalami kesulitan dalam memilih film yang relevan dari ribuan judul yang tersedia di platform digital.
- Rekomendasi film yang bersifat umum atau konvensional cenderung tidak sesuai dengan preferensi pribadi masing-masing pengguna.
- Kurangnya sistem rekomendasi yang mampu memahami karakteristik dan minat spesifik pengguna secara otomatis.

### **Goals**

- Mengembangkan sistem rekomendasi film yang mampu memberikan saran secara personal menggunakan pendekatan Content-Based Filtering.
- Meningkatkan kepuasan pengguna melalui rekomendasi yang relevan berdasarkan histori interaksi dan kesamaan konten film.
- Menyediakan rekomendasi Top-N film yang paling mirip dengan film-film yang disukai pengguna.

### **Solution Approach**

- **Content-Based Filtering**: Sistem akan menganalisis fitur-fitur film seperti genre dan tag untuk membangun profil preferensi pengguna. Rekomendasi kemudian diberikan berdasarkan kemiripan antara film-film yang telah disukai pengguna dengan film lain yang tersedia.
- Proses dilakukan menggunakan teknik vectorisasi teks (seperti TF-IDF) untuk representasi fitur, serta cosine similarity untuk mengukur kemiripan antar film.

---

## **Data Understanding**

Dataset yang digunakan adalah [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/). Data ini terdiri dari beberapa file:

- `movies.csv`: Informasi judul dan genre film
- `ratings.csv`: Rating yang diberikan pengguna terhadap film
- `tags.csv`: Tag yang diberikan pengguna terhadap film
- `links.csv`: ID film dari berbagai sumber (tidak digunakan dalam modeling)

### 1. movies.csv

#### Struktur Kolom:

- `movieId` (int): ID unik untuk setiap film.
- `title` (string): Judul film lengkap dengan tahun rilis, misalnya `Toy Story (1995)`.
- `genres` (string): Daftar genre film, dipisahkan dengan tanda `|` (pipe).

#### Karakteristik:

- Jumlah film unik: **87.585**.
- Sebuah film bisa memiliki lebih dari satu genre.
- Terdapat **20 genre unik**, termasuk `(no genres listed)` untuk film tanpa genre tercatat.
- Tahun rilis dapat diekstrak dari judul.
- Genre paling umum: `Drama`, `Comedy`, `Action`, dan `Thriller`.

### 2. ratings.csv

#### Struktur Kolom:

- `userId` (int): ID pengguna.
- `movieId` (int): ID film yang diberi rating.
- `rating` (float): Nilai rating antara 0.5 hingga 5.0 dalam kelipatan 0.5.
- `timestamp` (int): Waktu rating dalam Unix timestamp.

#### Karakteristik:

- Total rating: **32.000.204**.
- Jumlah pengguna unik: **283.228**.
- Rating paling sering: **4.0**, diikuti oleh 3.5, 5.0, dan 3.0.
- Rating menunjukkan distribusi condong ke atas (banyak rating positif).
- Data dapat digunakan untuk menganalisis pola preferensi pengguna atau membuat sistem rekomendasi.
- Timestamp dapat dianalisis untuk melihat tren waktu.

### 3. tags.csv

#### Struktur Kolom:

- `userId` (int): ID pengguna.
- `movieId` (int): ID film yang diberi tag.
- `tag` (string): Kata kunci atau label bebas dari pengguna.
- `timestamp` (int): Waktu pemberian tag dalam Unix timestamp.

#### Karakteristik:

- Jumlah total tag: **2.000.072**.
- Tag bersifat bebas dan tidak distandarkan (contoh: `superhero`, `super hero`).
- Sekitar 23% pengguna memberikan tag.
- Tag memberikan konteks tambahan terhadap film, seperti suasana, tema, atau gaya.
- Cocok untuk sistem rekomendasi berbasis konten atau analisis semantik.

### 4. links.csv

#### Struktur Kolom:

- `movieId` (int): ID film dari MovieLens.
- `imdbId` (int): ID film pada IMDb.
- `tmdbId` (int): ID film pada TMDb (The Movie Database).

#### Karakteristik:

- Jumlah total link: **87.585**.
- Menghubungkan film ke sumber data eksternal.
- Sebagian besar film memiliki data lengkap (hanya 21 film tanpa link).
- Berguna untuk pengayaan metadata seperti sinopsis, gambar, sutradara, aktor, dan lainnya.
- Cocok digunakan untuk membuat sistem rekomendasi berbasis metadata.

### Missing Value Analysis

Berikut adalah jumlah nilai hilang (missing values) pada masing-masing dataset:

#### Movies:

- `movieId`: 0
- `title`: 0
- `genres`: 0

**Semua data lengkap.** Tidak ada nilai yang hilang dalam dataset `movies.csv`.

#### Ratings:

- `userId`: 0
- `movieId`: 0
- `rating`: 0
- `timestamp`: 0

**Data utuh.** Tidak ada nilai yang hilang dalam dataset `ratings.csv`.

#### Tags:

- `userId`: 0
- `movieId`: 0
- `tag`: 17
- `timestamp`: 0

**Terdapat 17 nilai kosong pada kolom `tag`**, kemungkinan karena pengguna tidak mengisi teks tag.  
Penanganan yang bisa dilakukan:

- Menghapus entri kosong.
- Mengisi dengan label seperti `unknown` jika ingin mempertahankan entri.

#### Links:

- `movieId`: 0
- `imdbId`: 0
- `tmdbId`: 124

**Sebanyak 124 nilai hilang pada kolom `tmdbId`**. Hal ini menunjukkan 124 film tidak memiliki referensi ke The Movie Database.  
Penanganan yang mungkin:

- Jika tidak digunakan, bisa diabaikan.
- Jika dibutuhkan, bisa dilengkapi secara eksternal atau ditandai sebagai "tanpa data TMDb".

---

## **Exploratory Data Analysis**

### **1. Distribusi Rating**

1. **Distribusi Tidak Merata**:

   - Data rating tidak tersebar secara merata di seluruh rentang nilai.
   - Rating cenderung **terkonsentrasi di angka-angka tertentu**.

2. **Puncak Distribusi**:

   - Terdapat puncak (peak) paling tinggi di sekitar **rating 4**, yang mengindikasikan bahwa **rating 4 adalah yang paling umum** diberikan oleh pengguna.
   - Selain itu, rating **3 dan 5** juga cukup sering muncul.

3. **Rating Rendah Jarang Diberikan**:

   - Rating antara **0 hingga 2** memiliki frekuensi yang jauh lebih rendah, artinya hanya sedikit pengguna yang memberikan nilai rendah.

4. **Distribusi Multimodal**:

   - Terdapat beberapa puncak dalam kurva KDE, yang menunjukkan **banyaknya nilai rating yang sering diberikan secara spesifik** (misalnya: 3.0, 4.0, 5.0).
   - Ini mengindikasikan kecenderungan pengguna untuk memilih angka bulat.

5. **KDE Terlihat Tajam dan Berfluktuasi**:
   - Kurva KDE sangat fluktuatif (tajam dan bergerigi), yang kemungkinan besar disebabkan oleh:
     - **Jumlah data yang sangat besar**
     - **Bandwidth KDE yang terlalu kecil**

#### Kesimpulan

Grafik distribusi rating menunjukkan bahwa sebagian besar pengguna memberikan rating tinggi (khususnya 4), sementara hanya sebagian kecil yang memberikan rating rendah. Ini bisa menjadi indikator positif terhadap kualitas produk atau layanan yang dinilai.

### **2. Film dengan Rating Terbanyak**

1. **Film dengan Rating Terbanyak**:

   - **The Shawshank Redemption (1994)** menempati posisi pertama dengan jumlah rating tertinggi, sedikit di atas 100.000 rating.
   - Film ini sering dianggap sebagai salah satu film terbaik sepanjang masa, yang sejalan dengan tingginya jumlah rating.

2. **Film Lain dengan Jumlah Rating Tinggi**:

   - **Forrest Gump (1994)** dan **Pulp Fiction (1994)** menyusul di posisi kedua dan ketiga, dengan jumlah rating yang sangat dekat.
   - Film-film ini juga tergolong klasik dan populer secara global.

3. **Dominasi Film Tahun 1990-an**:

   - Mayoritas film dalam daftar dirilis pada **dekade 1990-an**, menunjukkan bahwa film-film dari periode tersebut masih sangat populer dan relevan bagi banyak penonton hingga kini.
   - Hanya satu film dari dekade 2000-an, yaitu **The Lord of the Rings: The Fellowship of the Ring (2001)**, yang masuk dalam daftar.

4. **Keberagaman Genre**:
   - Daftar ini mencakup berbagai genre seperti drama (**Shawshank Redemption**), aksi (**Matrix**, **Fight Club**), sci-fi (**Star Wars**), hingga thriller psikologis (**Silence of the Lambs**), menunjukkan bahwa popularitas tidak terbatas pada satu genre tertentu.

#### Kesimpulan

Grafik ini mengungkap bahwa film klasik dan berkualitas dari tahun 1990-an mendominasi dalam hal jumlah rating yang diberikan pengguna. Jumlah rating yang tinggi ini bisa menjadi indikator bahwa film-film tersebut banyak ditonton, dikenang, dan dinilai ulang oleh generasi penonton yang berbeda.

### **3. Genre Terbanyak**

1. **Genre Terpopuler**:

   - **Drama** merupakan genre dengan jumlah film terbanyak, yaitu mendekati **34.000 film**.
   - Diikuti oleh **Comedy** (~24.000 film) dan **Thriller** (~22.000 film).
   - Ini menunjukkan bahwa genre drama, komedi, dan thriller mendominasi produksi film secara umum.

2. **Genre Menengah**:

   - Genre seperti **Romance**, **Action**, **Documentary**, dan **Horror** memiliki jumlah film yang cukup besar, berkisar antara **8.000–11.000** film.
   - Menunjukkan bahwa tema cinta, aksi, dan dokumenter juga cukup banyak diminati dan diproduksi.

3. **Genre Kurang Umum**:

   - Genre seperti **Western**, **Musical**, **Film-Noir**, dan **IMAX** memiliki jumlah film yang jauh lebih sedikit dibanding genre lainnya.
   - **IMAX** menjadi genre dengan jumlah film paling sedikit, menunjukkan bahwa film dengan format khusus ini lebih jarang diproduksi.

4. **Keberagaman Genre**:
   - Terdapat lebih dari 15 genre yang berbeda, mencerminkan bahwa industri film menawarkan pilihan yang sangat beragam untuk berbagai selera dan usia.

#### Kesimpulan

Distribusi ini menunjukkan bahwa **Drama** adalah genre paling dominan dalam industri film, diikuti oleh **Comedy** dan **Thriller**. Sementara itu, genre-genre seperti **IMAX** dan **Film-Noir** relatif jarang. Informasi ini dapat digunakan untuk memahami tren preferensi produksi film dan potensi pasar dalam industri perfilman.

### **4. Tag yang Paling Sering Digunakan**

1. **Tag Terpopuler**:

   - Tag **"sci-fi"** menjadi yang paling sering digunakan, dengan jumlah melebihi **11.000 kemunculan**, menunjukkan bahwa genre fiksi ilmiah sangat populer di kalangan penonton atau pencatat metadata.

2. **Tag Umum Lainnya**:

   - Tag **"atmospheric"**, **"action"**, dan **"comedy"** juga termasuk yang paling banyak digunakan, menunjukkan bahwa elemen suasana, aksi, dan komedi adalah daya tarik utama dalam film.

3. **Unsur Cerita & Visual**:

   - Tag seperti **"twist ending"**, **"visually appealing"**, dan **"based on a book"** menunjukkan bahwa **alur cerita yang mengejutkan**, **visual yang menarik**, dan **adaptasi dari buku** juga merupakan faktor yang diapresiasi penonton.

4. **Kecenderungan Genre & Nuansa**:
   - Adanya tag seperti **"dark comedy"**, **"surreal"**, dan **"funny"** menunjukkan minat penonton terhadap film dengan nuansa tertentu, baik itu **humor gelap** maupun **pengalaman sinematik yang tidak biasa (surealis)**.

#### Kesimpulan

Tag-tag paling populer mencerminkan **preferensi audiens terhadap genre dan elemen naratif tertentu**, dengan fiksi ilmiah, atmosfer kuat, aksi, dan komedi menjadi daya tarik utama. Selain itu, aspek visual dan struktur cerita yang unik juga memainkan peran penting dalam persepsi dan klasifikasi film.

---

## **Data Preparation**

### 1. Pisahkan Genre

```python
movies['genre_list'] = movies['genres'].apply(lambda x: x.split('|') if x != '(no genres listed)' else [])
```

Kode ini melakukan transformasi data pada DataFrame `movies` dengan:

1. **Membuat kolom baru** bernama `'genre_list'`
2. **Mengisi kolom tersebut** dengan:
   - Daftar genre yang dipisahkan dari string asli (kolom `'genres'`) menggunakan `split('|')`
   - Jika genre adalah `'(no genres listed)'`, diisi dengan list kosong `[]`

**Contoh Transformasi**:

- Input: `"Action|Adventure|Sci-Fi"`  
  Output: `["Action", "Adventure", "Sci-Fi"]`
- Input: `"(no genres listed)"`  
  Output: `[]`

**Fungsi Utama**:

- Mengubah format genre dari string tunggal menjadi list terpisah
- Mempermudah analisis/operasi berbasis genre individual

### 2. Gabungan Data Tag per Film

```python
tags['tag'] = tags['tag'].astype(str).str.lower()
tags_clean = tags.dropna(subset=['tag'])
```

Kode berikut digunakan untuk membersihkan kolom `tag` pada dataset `tags.csv` dengan cara:

1. **Ubah menjadi lowercase**
   Tujuan:

   - astype(str) – Mengonversi nilai menjadi string agar bisa diproses secara tekstual
   - .str.lower() – Mengubah semua karakter menjadi huruf kecil untuk standarisasi Contoh: Superhero → superhero

2. **Hapus nilai kosong (NaN)**
   Tujuan:
   Menghapus baris yang tidak memiliki nilai tag agar hasil analisis hanya mencakup tag yang valid dan bermakna

3. **Hasil akhir**
   - Dataset baru bernama `tags_clean`
   - Berisi kolom `tag` yang telah:
     - Diubah ke huruf kecil
     - Bersih dari nilai kosong
   - Siap digunakan untuk analisis teks atau sistem rekomendasi berbasis konten

```python
tags_grouped = tags_clean.groupby('movieId')['tag'].apply(lambda x: ' '.join(set(x))).reset_index()
```

Kode ini melakukan penggabungan (aggregasi) tag untuk setiap film dengan cara:

1. **Group by movieId**

   - Mengelompokkan data berdasarkan `movieId` (ID film unik)

2. **Gabungkan tag**

   - Menggunakan `lambda x: ' '.join(set(x))` untuk:  
     a. `set(x)` - Menghapus duplikat tag  
     b. `' '.join()` - Menggabungkan tag unik menjadi satu string dipisahkan spasi

3. **Hasil akhir**
   - Dataframe baru `tags_grouped` dengan kolom:
     - `movieId` (ID film)
     - `tags_combined` (string gabungan semua tag unik untuk film tersebut)

### 2. Gabungkan `movies.csv` dengan Tag

```python
movies_merged = pd.merge(movies, tags_grouped, on='movieId', how='left')
movies_merged['tags_combined'] = movies_merged['tags_combined'].fillna('')
```

Kode ini melakukan penggabungan (merge) dua DataFrame dengan langkah:

1. **Merge Data**

   - Menggabungkan `movies` dan `tags_grouped` berdasarkan kolom `movieId`
   - `how='left'` artinya semua data dari `movies` dipertahankan (left join)
   - Tag hanya akan ditambahkan jika `movieId` ada di kedua tabel

2. **Handle Missing Values**
   - `fillna('')` mengubah nilai kosong/NULL di kolom `tags_combined` menjadi string kosong
   - Memastikan tidak ada nilai NaN yang mengganggu operasi berikutnya

### 3. Pembuatan Fitur Konten

```python
movies_merged['genre_str'] = movies_merged['genre_list'].apply(lambda x: ' '.join(x))
movies_merged['combined_features'] = (
    movies_merged['title'].str.lower() + ' ' +
    movies_merged['genre_str'].str.lower() + ' ' +
    movies_merged['tags_combined']
)
```

Kode ini menggabungkan beberapa fitur teks dari dataset film menjadi satu kolom:

1. **Membuat string genre**  
   `movies_merged['genre_str'] = movies_merged['genre_list'].apply(lambda x: ' '.join(x))`

   - Mengubah list genre menjadi string tunggal dipisahkan spasi

2. **Menggabungkan semua fitur**  
   `movies_merged['combined_features']` dibuat dengan menggabungkan:
   - Judul film (dikonversi lowercase)
   - String genre (lowercase)
   - Tag yang sudah digabungkan
   - Dipisahkan oleh spasi

### 4. Gabung dengan Rata-Rata Rating Film

Kode berikut digunakan untuk menghitung rata-rata rating setiap film dan menggabungkannya ke dataset film

```python
rating_mean = ratings.groupby('movieId')['rating'].mean().reset_index()
```

Tujuan:

- Mengelompokkan data rating berdasarkan `movieId`
- Menghitung rata-rata nilai rating untuk setiap film
- `reset_index()` digunakan untuk mengubah hasil agregasi menjadi DataFrame yang rapi

```python
rating_mean.columns = ['movieId', 'average_rating']
```

Tujuan:
Memberi nama yang lebih deskriptif pada kolom hasil agregasi: - `movieId` → ID film - `average_rating` → nilai rata-rata rating

```python
movies_final = pd.merge(movies_merged, rating_mean, on='movieId', how='left')
```

Tujuan:

- Menggabungkan dataset `movies_merged` dengan `rating_mean` berdasarkan `movieId`
- `how='left'` memastikan semua film tetap ada, meskipun tidak memiliki rating
- Hasilnya adalah dataset movies_final yang berisi informasi film ditambah kolom `average_rating`

**Hasil Akhir**:

- DataFrame baru bernama `movies_final` dengan kolom-kolom seperti:
  - `movieId`, `title`, `genres`, metadata lain (dari `movies_merged`)
  - `average_rating` – nilai rata-rata rating untuk setiap film
- Siap digunakan untuk analisis lebih lanjut, misalnya:
  - Menyaring film dengan rating tertinggi
  - Menyusun rekomendasi berdasarkan rating

### 5.TF-IDF Vectorization

Mengubah teks gabungan (`combined_features`) menjadi vektor numerik:

```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_final['combined_features'])
print(f"Bentuk TF-IDF matrix: {tfidf_matrix.shape}")
```

Contoh Output:

```
Bentuk TF-IDF matrix: (87585, 71417)
```

---

## **Modeling and Result**

Bagian ini menjelaskan pendekatan utama dalam sistem rekomendasi film yang dikembangkan: **Content-Based Filtering**, dengan tambahan fitur rekomendasi berdasarkan suasana hati (_mood_).

### **Content-Based Filtering**

Pendekatan ini merekomendasikan film berdasarkan **kemiripan konten** seperti genre, judul, dan tag. Metode utama yang digunakan adalah **KNN (K-Nearest Neighbors)** berbasis **Cosine Similarity**.

#### Langkah-langkah Pemodelan

**1. KNN (Cosine Similarity)**

Mengukur kemiripan antar film:

```python
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(tfidf_matrix)
```

**2. Mapping Judul dan `movieId` ke Index**
Memudahkan pencarian film:

```python
indices = pd.Series(movies_final.index, index=movies_final['title'].str.lower()).drop_duplicates()
movieId_to_index = pd.Series(movies_final.index, index=movies_final['movieId'])
```

### Fungsi Rekomendasi Berdasarkan Judul

```python
def recommend_movies(title, num_recommendations=10):
    ...
```

**Contoh Pemanggilan:**

```python
recommend_movies("Toy Story (1995)")
```

**Contoh Output (Top 10):**

```
1. Toy Story 2 (1999)
2. Toy Story 4 (2019)
3. Toy Story 3 (2010)
4. Toy Story That Time Forgot (2014)
5. Toy Story Toons: Small Fry (2011)
6. Toy Story Toons: Partysaurus Rex (2012)
7. Toy Story Toons: Hawaiian Vacation (2011)
8. Toy Masters (2014)
9. Small Soldiers (1998)
10. Lamp Life (2020)
```

---

### **Rekomendasi Berdasarkan Mood (Advanced)**

Pengguna dapat menyaring rekomendasi berdasarkan **mood** tertentu:

```python
def recommend_movies_advanced(title, num_recommendations=10, mood_filter=None):
    ...
```

Contoh `mood_filter` yang tersedia:

- `bahagia`
- `romantis`
- `sedih`
- `tegang`
- `ceria`
- `imajinatif`
- `nyata`
- `klasik`

**Contoh Penggunaan:**

```python
recommend_movies_advanced("Jumanji (1995)", num_recommendations=5, mood_filter="imajinatif")
```

**Contoh Output:**

```
1. Spider-Man: Into the Spider-Verse (2018)
2. Back to the Future (1985)
3. Terminator 2: Judgment Day (1991)
4. The Terminator (1984)
5. Deadpool (2016)
```

---

## **Evaluasi Sistem Rekomendasi: Precision@10**

### Tujuan Evaluasi:

Mengevaluasi seberapa **relevan rekomendasi film** berdasarkan film yang disukai oleh pengguna.  
Metode yang digunakan adalah **Precision@10**, yaitu proporsi film yang benar-benar disukai (berdasarkan rating pengguna) di antara 10 film yang direkomendasikan.

---

### Langkah Evaluasi:

1. **Filter User Aktif**

   - Hanya pengguna yang memberi **≥ 20 rating** diikutkan dalam evaluasi.
   - Hal ini dilakukan agar hasil evaluasi tidak bias akibat data yang terlalu sedikit.

2. **Fungsi `evaluate_precision_at_k`**

   - Untuk setiap pengguna:
     - Ambil semua film yang dia beri **rating ≥ 4.0** (film yang disukai).
     - Untuk setiap film yang disukai, cari **10 film mirip** menggunakan sistem rekomendasi (TF-IDF + KNN).
     - Hitung berapa banyak dari film yang direkomendasikan juga termasuk film yang disukai oleh user tersebut (selain film asal).
     - Hitung Precision@10:
       \[
       \text{Precision@10} = \frac{\text{Jumlah hits}}{10 \times \text{jumlah film disukai}}
       \]

3. **Evaluasi 30 Pengguna**
   - Precision@10 dihitung untuk **30 user aktif pertama**.

---

### Hasil Akhir:

```python
Rata-rata Precision@10: 0.1408
```

- Artinya, **sekitar 14.08%** dari film yang direkomendasikan untuk setiap pengguna benar-benar termasuk dalam daftar film yang mereka sukai.
- Ini merupakan hasil awal yang **lumayan baik untuk sistem content-based sederhana**, namun tentu masih dapat ditingkatkan (misalnya dengan personalisasi, hybrid model, atau collaborative filtering).

---

### Kesimpulan:

- **Content-based filtering** seperti ini **masih terbatas** dalam memahami selera pengguna secara menyeluruh karena hanya melihat kesamaan antar film.
- Precision@10 sebesar **0.1408** menunjukkan bahwa dari 10 film yang direkomendasikan, rata-rata **hanya 1–2 film yang benar-benar disukai** oleh user.
- Sistem ini bisa dijadikan dasar yang kuat, tetapi untuk **peningkatan kualitas rekomendasi**, perlu integrasi teknik tambahan seperti:
  - Collaborative Filtering
  - Rekomendasi berbasis histori user
  - Feedback loop dari pengguna

### Interpretasi dan Implikasi Bisnis:

- Sistem rekomendasi yang dibangun telah **berhasil mengurangi kebingungan pengguna** dalam memilih film, dengan memberikan **daftar film yang memiliki kesamaan konten** terhadap film yang mereka sukai sebelumnya.
- Meskipun nilai **Precision@10 sebesar 0.1408** belum sangat tinggi, ini merupakan **indikator awal yang positif**, terutama untuk sistem yang belum menggunakan data eksplisit dari preferensi pengguna secara mendalam (misalnya perilaku menonton).

- Hasil ini menunjukkan bahwa sistem **sudah memiliki kemampuan dasar dalam memahami kesamaan konten**, sehingga dapat ditingkatkan lebih lanjut untuk:
  - Mengintegrasikan perilaku pengguna (collaborative filtering)
  - Menyesuaikan rekomendasi berdasarkan suasana hati (mood)
  - Menyediakan rekomendasi yang lebih personal dan kontekstual

---

### Alignment dengan Solution Approach:

- Sistem telah menggunakan **TF-IDF vectorization** untuk menangkap informasi fitur film (judul, genre, tags).
- **Model KNN** berbasis **cosine similarity** telah digunakan untuk mengukur kedekatan antar film dan merekomendasikan film yang paling mirip.
- Evaluasi dengan metrik **Precision@10** menunjukkan bahwa model bekerja secara logis dan dapat dikembangkan lebih jauh untuk meningkatkan kepuasan pengguna.

---

## **Referensi**

[1] Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer.

[2] Gómez-Uribe, C. A., & Hunt, N. (2015). The Netflix Recommender System: Algorithms, Business Value, and Innovation. ACM Transactions on Management Information Systems (TMIS), 6(4), 13.

[3] Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), 1–19.

[4] Aggarwal, C. C. (2016). Recommender Systems: The Textbook. Springer.

[5] Lops, P., De Gemmis, M., & Semeraro, G. (2011). Content-based Recommender Systems: State of the Art and Trends. In Recommender Systems Handbook (pp. 73–105). Springer.
