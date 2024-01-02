# Laporan Proyek Machine Learning
### Nama : Deni Andriansyah
### Nim : 211351040
### Kelas : Pagi B

## Domain Proyek
Proyek ini ditujukan untuk melakukan analisis kangker paru - paru, penyakit ini telah menjadi fokus utama penelitian dalam dunia kedokteran dan kesehatan karena dampaknya yang sangat merusak bagi individu, keluarga, dan bahkan masyarakat secara keseluruhan. Analisis disini menggunakan data yang mencakup informasi gejala- gejala yang dialami pasien untuk menentukan tingkat keparahan kangker paru - paru pasien, hasil dari proyek ini agar bisa membantu dalam pencegahan atau perawatan pasien kanker paru - paru.

## Business Understanding
Memahami tentang penyakit kanker paru - paru dengan melibatkan gejala - gejala yang dialami yang menjadi penyebab kanker paru - paru itu muncul pada pasien seberapa parah kanker yang dialaminya dan diharapakan bisa menjadi alat yang dapat mengetahui seberapa parah kanker yang dialami pasien.

### Problem Statements
Agar dapat mengetahui seberapa tingkat keparahan pasien kanker paru - paru untuk segera mendapatkan perawatan yang efektif

### Goals
Untuk mengetahui gejala gejala yang dialami pasien serta tingkat keparahan kanker paru - paru pasien agar mambatu langkah perawatan selnajutnya yang efektif untuk pasien kanker paru - paru

### Solution statements
Pengembangan model prediksi untuk membantu memprediksi seberapa parah kanker paru - paru yang dialami pasien menggunakan algoritma K-Nearst Neighbors
## Data Understanding
Dataset yang saya gunakan saya mengambilnya dari Kaggle yang merupakan dasar analisis terkait prediksi kanker paru - paru<br>
[Lung Cancer](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link).

### Variabel-variabel pada Fastfood Nutrition adalah sebagai berikut:
- Patient Id = Identitas Pasien = object
- Age = Usia pasien = int64
- Gender = Jenis kelamin pasien = int64
- Air_Pollution = Tingkat paparan polusi udara pasien = int64
- Alcohol_use = Tingkat penggunaan alkohol pasien = int64
- Dust_Allergy = Tingkat alergi debu pasien = int64
- Genetic_Risk = Tingkat penyakit paru kronis pasien = int64
- chronic_Lung_Disease = Tingkat penyakit paru kronis pasien = int64
- Balanced_Diet = Tingkat diet seimbang pasien = int64
- Obesity = Tingkat obesitas pasien = int64
- Smoking = Tingkat merokok pasien = int64
- Passive_Smoker = Tingkat perokok pasif pasien = int64
- Chest_Pain = Tingkat nyeri dada pasien = int64
- Coughing_of_Blood = Tingkat batuk darah pasien = int64
- Fatigue = Tingkat kelelahan pasien = int64
- Weight_Loss = Tingkat penurunan berat badan pasien = int64
- Shortness_of_Breath = Tingkat sesak nafas pasien = int64
- Wheezing = Tingkat mengi pasien = int64
- Swallowing_Difficulty = Tingkat kesulitan menelan pasien = int64
- Clubbing_of_Finger_Nails = Tingkat clubbing kuku jari pasien = int64
- Frequent_Cold = Sering Pilek = int64
- Dry_Cough = Batuk kering = int64
- Snoring = Keruh = int64
- Level = Level kanker paru - paru = object
  
## Data Preparation
Dataset yang saya gunakan yaitu mengambil dari Kaggle

Pertama import library yang akan digunakan
``` bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
```
Selanjutnya agar bisa mendownload dataset dari Kaggle melalui  google colab  dengan Kaggle buat token di Kaggle lalu download
dan unggah token yang sudah di download  pada script di bawah ini
```bash
from google.colab import files
files.upload()
```
Setelah mengupload tokennya, bisa di lanjut dengan membuat sebuah folder untuk menyimpan file kaggle.json yang sudah diupload 
File kaggle.json berisi kunci API Anda yang akan digunakan untuk otentikasi saat menggunakan API Kaggle 
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
Selanjutnya mendwonload dataset dari Kaggle
```bash
!kaggle datasets download -d ulrikthygepedersen/fastfood-nutrition
```
Setelah terdownload extract file yang telah terdownload tadi
```bash
!mkdir fastfood-nutrition
!unzip fastfood-nutrition.zip -d fastfood-nutrition
!ls fastfood-nutrition
```
Lanjut dengan membaca file csv yang telah di extract sebelumnya
```bash
df = pd.read_csv('/content/fastfood-nutrition/fastfood.csv')
```
Lalu melihat 5 baris pertama pada datasetnya untuk memeriksa data apakah sudah benar
```bash
df.head()
```
Dan juga melihat 5 baris terakhir pada dataset untuk memeriksa data apakah sudah benar juga
```bash
df.tail()
```
Dikarenkan ada nilai yang hilang maka disini akan menghapus nilai tersebut
```bash
df.dropna(inplace=True)
```
Selesai menghapus data yang hilang, agar dapat melihat mengenai type data maka
```bash
df.info()
```
Selanjutnya disini akan memeriksa apakah sudah aman atau masih terdapat nialai yang hilang
```bash
sns.heatmap(df.isnull())
```
![Alt text](hm.png) <br>
Bisa dilihat aman<br>
Selanjutnya agar mengetahui detail informasi dari dataset
```bash
df.describe()
```
Lalu selanjutnya agar mengetahui jumlah masing-masing jenis restoran yang terdaftar 
```bash
df['restaurant'].value_counts()
```
Dan juga disini agar mengetahui jumlah masing-masing jenis item 
```bash
df['item'].value_counts()
```
Selanjutnya 
## Visualisasi Data
Untuk melihat visualisasi dan memahami hubungan antara berbagai fitur dalam dataset dengan warna yang menggambarkan tingkat dan arah korelasi antar fitur
```bash
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
```
![Alt text](htmp.png) <br>
mari kita lihat penyebaran kalori per itemnya
```bash
chick_fila = df[df['restaurant'] == 'Chick Fil-A']
plt.figure(figsize=(18, 10))
plt.title('Chick Fil-A Calorie Per Item')
sns.barplot(y=chick_fila['item'], x=df['calories'])
```
![download](download.png)
Tahap selanjutnya
## Modeling
Karena library yang akan digunakan sudah diawal maka selanjutnya<br>
Untuk melakukan modeling  memakai algoritma regresi linear dimana harus memisahkan atribut yang akan dijadikan sebagai fitur(x) dan atribut mana yang dijadikan label(y).
```bash
features = ['fiber', 'total_carb', 'sodium', 'cal_fat', 'total_fat', 'sat_fat', 'protein', 'sugar']
x=df[features]
y=df['calories']
x.shape, y.shape
```
Setelah itu lakukan split data, memisahkan data training dan data testing 
```bash
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=70)
y_test.shape
```
Selanjutnya masukan data training dan testing ke dalam model regresi linier
```bash
lr = LinearRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
```
Setelah itu untuk mengecek akurasi
```bash
score = lr.score(x_test, y_test)
print('akurasi model regresi linier =', score)
```
```bash
akurasi model regresi linier = 0.991859361417439
```
Hasil akurasinya yaitu 99.19% bahwa hasil itu adalah hasil data yang akurat<br>
Selanjutnya melakukan test menggunakan sebuah array value
```bash
# fiber =3.0 total_carb =44 sodium =1110 cal_fat=60 total_fat =7 sat_fat =2.0 protein =37.0 sugar =11
input_data = np.array([[3.0,44,1110,60,7,2.0,37.0,11]])

prediction = lr.predict(input_data)
print('Estimasi kalori pada makanan cepat saji :', prediction)
```
```bash
Estimasi kalori pada makanan cepat saji : [402.15587709]
```
Berhasil membuat model dapat diketahui estimasi kalori pada makanan cepat saji, selanjutnya save model sebagai sav agar dapat digunakan pada streamlit
```bash
import pickle

filename = 'estimasi_kalori.sav'
pickle.dump(lr,open(filename,'wb'))
```
## Evaluation
Evaluasi ini merupakan seberapa cocok model dengan data yang dipakai.
Untuk metrik evaluasi yang digunakan yaitu R-squared<br>
R-squared yaitu koefisien determinasi yang merupakan ukuran seberapa baik model regresi linear cocok dengan data yang diamati

Selanjutnya untuk evaluasi seberapa baik model cocok dengan data dihitung dengan rumus:<br>
![Alt text](rms.png) <br>
```bash
lr.fit(x_train, y_train)
y_train_prediction = lr.predict(x_train)
```
```bash
r_squared = r2_score(y_train, y_train_prediction)
print(f"R-squared : {r_squared}")
```
```bash
R-squared : 0.9525584382039234
```
Dan hasil yang saya dapatkan adalah 0.9525584382039234 atau 95.26% model regresi secara umum cukup cocok dengan data, karena memiliki kemampuan yang baik untuk menjelaskan variasi dalam target.
## Deployment
[My Estimation App](https://estimasi-kalori-f7tfpsa8flrfnuehxcjzlx.streamlit.app/).

![Alt text](tm.png)

