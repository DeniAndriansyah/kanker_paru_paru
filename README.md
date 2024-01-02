# Laporan Proyek Machine Learning
### Nama : Deni Andriansyah
### Nim : 211351040
### Kelas : Pagi B

## Domain Proyek

Estimasi makanan cepat saji yang tinggi akan kalori, jika tidak menyeimbangkannya dengan aktivitas fisik yang cukup dapat menyebabkan kelebihan berat badan dan obesitas. Kondisi ini meningkatkan risiko terkena berbagai penyakit, seperti diabetes, penyakit jantung, dan beberapa jenis kanker. Untuk itu saya berupaya membuat perhitungan kalori dengan 8 parameter yang sudah ditentukan agar mencegah kalian terkena berbagai resiko penyakit.

## Business Understanding

Memudahkan orang-orang dalam mengecek makanan yang jumlah kalorinya rendah untuk mencegah terkena bergbagi resiko penyakit 

Bagian laporan ini mencakup:

### Problem Statements

Ketidaktahuan orang- orang pada kandungan makanan yang rendah kalori pada makanan cepat saji.

### Goals

Agar kalian waspada dalam membeli makanan cepat saji dan menjaga pola hidup sehat agar terhindar dari berbagai resiko penyakit.

### Solution statements
- Pengembangan estimasi kalori berbasis web yang menghitung penjumlahan kalori pada makanan cepat saji untuk memudahkan pengguna mengecek jumlah kalori pada makanan cepat saji terlebih dahulu dengan menggunakan model algoritma regresi linear.
## Data Understanding
Dataset yang saya gunakan saya mengambilnya dari Kaggle yang berisi informasi tentang kalori, lemak, karbohidrat, protein, dan nutrisi penting lainnya, kumpulan data ini memberikan sumber daya berharga bagi ahli gizi, peneliti, dan individu yang sadar kesehatan. Dengan menganalisis kumpulan data ini, kita dapat memperoleh pemahaman yang lebih baik tentang dampak nutrisi dari konsumsi makanan cepat saji dan berupaya menciptakan pilihan makanan yang lebih sehat.<br>
[Fastfood Nutrition](https://www.kaggle.com/datasets/ulrikthygepedersen/fastfood-nutrition).

### Variabel-variabel pada Fastfood Nutrition adalah sebagai berikut:
- restaurant : Merupakan toko pada makanan cepat saji.(object)
- item : Merupakan makanan cepat saji.(object)
- calories : Merupakan kalori pada makanan dan minuman.(int)
- fiber : Merupakan serat pangan.(float)
- total_carb : Merupakan jumlah total kabohidrat.(int)
- sodium : Merupakan penyedap dari bahan alami.(int)
- cal_fat : Merupakan kalori dari lemak.(int)
- total_fat : Merupakan total lemak.(int)
- sat_fat : Merupakan lemak jenuh.(float)
- protein : Merupakan protein pada makanan.(float)
- sugar : Merupakan rasa manis pada makanan.(int)
- trans_fat : Merupakan kemak tak jenuh pada makanan(float)
- vit_a : Merupakan Vitamin pada makanan(float)
- vit_c : Merupakan Vitamin pada makanan(float)
- calcium : Merupakan Kalsium pada makanan(float)
- cholesterol : Merupakan kolesterol pada makanan(int)
- salad : Merupakan jenis makanan sayuran(object)

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

