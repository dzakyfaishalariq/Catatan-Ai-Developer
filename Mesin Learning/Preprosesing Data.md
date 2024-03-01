## Data Numerik

### Mising Value
1) Menghapus colom yang memiliki nilai yang mis atau kosong di salah satu nilai baris nya
	Kode yang digunakan dalam python untuk menghapus kolom tersebut:
		`x_train.drop("nama_Colom", axis=1)`
2) Mengisi nilai yang hilang yang ada didalam salah satu kolom:
	Kode yang digunakan dalam python menggunakan sebuah library:
```python
form sklearn.impute import SampelImputer
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
```

Dan juga ada beberapa penggunaan `SampelImputer` yang sering digunakan dalam mesin learning:
```python
from sklearn.impute import SimpleImputer

# Membuat objek imputer dengan menggunakan median
imputer_median = SimpleImputer(strategy='median')

# Membuat objek imputer dengan menggunakan mode
imputer_mode = SimpleImputer(strategy='most_frequent')

# Membuat objek imputer dengan menggunakan kuartil pertama (Q1)
imputer_q1 = SimpleImputer(strategy='constant', fill_value=data.quantile(0.25))

# Membuat objek imputer dengan menggunakan interpolasi linear
imputer_linear = SimpleImputer(strategy='linear')

# Membuat objek imputer dengan menggunakan konstanta
imputer_constant = SimpleImputer(strategy='constant', fill_value=0)

```
## Data Categorical
Pada saat mengolah data yang besifat kategorikal maka kita akan mengubah nya ke dalam bentuk sebuah bilangan integer atau binary
1) Mengubah menjadi bilangan integer
contoh code:
``` python
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])
```
Kita juga dapat melakukan costum di `OrdinalEncoder`
```python
from sklearn.preprocessing import OrdinalEncoder

# Membuat objek OrdinalEncoder dengan pengaturan kustom
ordinal_encoder = OrdinalEncoder(
    categories=[['S', 'M', 'L'], ['red', 'green', 'blue']],
    dtype=int,
    handle_unknown='use_encoded_value',
    unknown_value=-1
)
```
2) Mengubah menjadi bilangan Binary
contoh code
```python
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
```

[[Pipline]]

#kecerdasan_buatan #mesin_learning 