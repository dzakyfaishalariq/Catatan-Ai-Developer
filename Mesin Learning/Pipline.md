Penggunaan Papilne berugan untuk mempersingkat dan memudahkan mendevelob sebuah preprosesing hingga ke dalam training dalam mesing learning.

berikut contoh code nya:
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
# Preprocessing untuk data numerical
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing untuk data kategorical
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Menggabugkan tahapan sesuai data yang kita punya
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

```
hal ini juga bisa untuk menggabungkan dengan tahapan modelnya dengang kode seperti berikut ini:
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# inisialisasi model
model = RandomForestRegressor(n_estimators=100, random_state=0)
# Gabungkan proses preprosesing dengan model sebagai tahapan di pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# lakukan fit atau pembelajaran
my_pipeline.fit(X_train, y_train)```

