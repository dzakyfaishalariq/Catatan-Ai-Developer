
> Penggunana ini diwajibkan ke dalam mesin learning statistical dengan memanfaatkan penggunan [sklearn](https://scikit-learn.org/stable/) harap melakukan instalasi ini terlebih dahulu
# Klasifikasi
Merupakan teknik untuk mengelompokan data berdasarkan class yang sudah di sediakan

---
### Logistic Regression
pengertian : Sebuah metode statistik yang digunakan untuk memprediksi probabilitas dari peristiwa biner berdasarkan satu atau beberapa variabel prediktor.
contoh code:
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```

### Support Vector Machines (SVM):
pengertian : Metode pembelajaran yang digunakan untuk klasifikasi atau regresi, yang membagi ruang fitur dengan cara mencari hyperplane yang mengoptimalkan pemisahan antar kelas.
contoh code :
```python
from sklearn.svm import SVC
model = SVC()
```

### K-Nearest Neighbors (KNN):
pengertian : Algoritma yang beroperasi berdasarkan asumsi bahwa data dengan atribut yang serupa cenderung berada dalam grup atau kelas yang sama.
contoh code:
```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
```

### Naive Bayes Classifier:
pengertian : Metode klasifikasi yang berdasarkan pada teorema Bayes dengan asumsi independensi antara fitur-fitur.
contoh code :
```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
```
### Decision Trees:
pengertian: Metode pembelajaran yang menghasilkan model keputusan dalam bentuk struktur pohon. Setiap simpul dalam pohon tersebut mewakili keputusan berdasarkan fitur-fitur input.
contoh code :
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
```

### Random Forest:
pengertian : Sebuah ensemble learning method yang menggunakan beberapa pohon keputusan secara acak untuk melakukan klasifikasi atau regresi.
contoh code :
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
```

### Neural Networks (Multilayer Perceptron):
pengertian : Model pembelajaran mesin yang terinspirasi oleh struktur jaringan saraf biologis, terdiri dari beberapa lapisan neuron.
contoh code :
``` python
from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
```

# Regresi:
Merupakan teknik untuk mengetahui keterhubungan dari fitur - fitur yang ada di dalam colom dalam garis lurus.

---
### Linear Regression:
pengertian : Metode statistik yang digunakan untuk memodelkan hubungan antara variabel dependen dengan satu atau lebih variabel independen dengan asumsi hubungan tersebut bersifat linier.
contoh code :
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

### Polynomial Regression:
pengertian : Variasi dari regresi linear di mana hubungan antara variabel independen dan dependen dimodelkan sebagai derajat polinomial.
contoh code:
``` python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression()
```

### Ridge Regression:
pengertian : Metode regresi yang menambahkan penalitas L2 regularization ke dalam fungsi kerugian untuk mengurangi overfitting.
contoh code :
```python
from sklearn.linear_model import Ridge
model = Ridge()
```

### Lasso Regression:
pengertian : Metode regresi yang menambahkan penalitas L1 regularization ke dalam fungsi kerugian untuk memperoleh model yang lebih sederhana dan sparse.
contoh code :
```python
from sklearn.linear_model import Lasso
model = Lasso()
```

### Support Vector Regression (SVR):
pengertian: Versi regresi dari SVM yang mencari hyperplane yang menghasilkan margin terbesar untuk menyesuaikan data dengan baik.
contoh code :
```python
from sklearn.svm import SVR
model = SVR()
```

### Decision Tree Regression:
pengertian : Versi regresi dari decision tree yang menggunakan struktur pohon untuk memprediksi nilai numerik.
contoh code :
```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
```

### Random Forest Regression:
pengertian: Versi regresi dari random forest yang menggunakan ensemble dari beberapa pohon keputusan untuk memprediksi nilai kontinu.
contoh code :
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
```

# Clustering:
Merupakan teknik untuk mengelompokan gambar sesuai identitas atau cirihas data tersebut di setiap baris tampa menggunakan label class yang di tetapkan.

---
### K-Means Clustering:
pengertian : Metode klastering yang membagi data menjadi k kelompok dengan cara mencari pusat klaster yang mewakili data dengan jarak terkecil.
contoh code :
```python
from sklearn.cluster import KMeans
model = KMeans()
```

### Hierarchical Clustering (Agglomerative):
pengertian : Metode klastering yang membangun hirarki klaster dengan menggabungkan klaster secara berurutan berdasarkan jarak antara mereka.
contoh code :
```python
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering()
```

### DBSCAN:
pengertian : Metode klastering yang membagi data berdasarkan kerapatan, dengan mengidentifikasi klaster sebagai wilayah data yang memiliki kerapatan yang cukup tinggi.
contoh code :
```python
from sklearn.cluster import DBSCAN
model = DBSCAN()
```

### Mean Shift Clustering:
pengertian : Metode klastering non-parametrik yang mencari pusat klaster dengan memaksimalkan kerapatan data di sekitarnya.
contoh code : 
```python
from sklearn.cluster import MeanShift
model = MeanShift()
```

### Gaussian Mixture Models (GMM):
pengertian : Model yang mengasumsikan bahwa data terdiri dari sejumlah klaster Gaussian yang berbeda, dan mencoba untuk menemukan distribusi Gauss yang paling baik mewakili data.
contoh code :
```python
from sklearn.mixture import GaussianMixture
model = GaussianMixture()
```

### Spectral Clustering:
pengertian : Metode klastering yang menggunakan representasi grafik data untuk mengidentifikasi klaster berdasarkan struktur spektral.
contoh code :
```python
from sklearn.cluster import SpectralClustering
model = SpectralClustering()
```

### Self-Organizing Maps (SOM):
pengertian : Metode klastering dan pemetaan non-linier yang menggunakan jaringan saraf untuk memetakan data ke dalam struktur spasial yang diorganisasi.
keterangan : harus menginstal pihak ketiga seperti [MiniSom](https://pypi.org/project/MiniSom/).

# Peramalan:
dalam sistem ini model ditugaskan untuk memprediksi perkiraan hasil kedepan berdasarkan urutan waktu *time seris*

---

> Dalam bagian ini kita harus menginstal aplikasi yaitu [statsmodels](https://www.statsmodels.org/stable/gettingstarted.html) silahkan menginstal aplikasi tersebut dan jalankan di kode python

### ARIMA:
pengertian : Model statistik yang digunakan untuk meramalkan nilai-nilai waktu seri berdasarkan informasi historis dan pola tren, musiman, serta residual.
contoh code :
```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(data, order=(p, d, q))
```

### Exponential Smoothing Methods (Holt-Winters):
pengertian: Metode peramalan yang memperkirakan nilai-nilai masa depan dengan memberikan bobot yang berbeda pada pengamatan masa lalu.
contoh code :
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
```

### Seasonal Decomposition of Time Series (STL):
pengertian: Metode yang memecah seri waktu menjadi komponen-komponen musiman, trend, dan residu.
contoh code:
```python
from statsmodels.tsa.seasonal import STL
model = STL(data, seasonal=13)  # Contoh untuk musiman dengan periode 13
```

### Long Short-Term Memory (LSTM) Networks:
pengertian : Jenis arsitektur jaringan saraf rekuren yang memiliki kemampuan untuk mengingat informasi dalam jangka waktu yang panjang.
keterangan : harus menginstall pihak ketiga seperti [Keras](https://keras.io/) [TensorFlow](https://www.tensorflow.org/) 

### Prophet:
pengertian : Alat peramalan yang dikembangkan oleh Facebook untuk meramalkan data deret waktu dengan mudah dan cepat.
contoh code :
```python
from fbprophet import Prophet
model = Prophet()
```
keterangan : harus menginstall [fbprophet](https://pypi.org/project/fbprophet/)

### Gaussian Process Regression (GPR):
pengertian : Metode regresi non-parametrik yang menggunakan proses Gaussian untuk menghasilkan distribusi probabilitas atas fungsi regresi.
contoh code :
``` python
from sklearn.gaussian_process import GaussianProcessRegressor
model = GaussianProcessRegressor()
```

### Recurrent Neural Networks (RNNs) for Time Series Prediction:
pengertian : Jaringan saraf rekuren yang dirancang untuk memodelkan hubungan antara data deret waktu, memungkinkan informasi dari masa lalu untuk dipertahankan dalam model.
keterangan :  harus menginstall pihak ketiga seperti [Keras](https://keras.io/) [TensorFlow](https://www.tensorflow.org/) 
[[Pipline]]

#model #mesin_learning #kecerdasan_buatan 