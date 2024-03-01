# Library math
Penggunaan Library math di dalam python terdiri dari
1. Factorial referensi:[klik](https://en.wikipedia.org/wiki/Factorial)
Contoh code nya:
``` python
import math
n=5
fact = math.factorial(n)
```
# Library ZipFile
Digunakan untuk melakukan unzipfile atau sebaliknya dari data unduhan zip
ZipFile
Reverensi:[klik](https://www.geeksforgeeks.org/working-zip-files-python/)
contoh code nya :
```python
# extract the downloaded dataset to a local directory: /tmp/fcnn
import zipfile

local_zip = '/tmp/fcnn-dataset.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/tmp/fcnn')

zip_ref.close()
```

#kecerdasan_buatan #mesin_learning 