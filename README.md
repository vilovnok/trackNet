# TrackNet_Inference

Моделирование треки частиц через детектор и обработка координатов попадания (хитов) в файл.

## Libtorch
Get libtorch from https://pytorch.org/get-started/locally/ and unzip it into ```include``` directory.
```bash
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcu118.zip
```

## Тестовые данные
To get test data:
```bash
python spdsim_v3.py 1000
```
Будем генерировать output.tsv файл с 1000 событиями.

## Создаем Docker образ
```bash
docker-compose up -d
```
Connect to terminal:
```bash
docker-compose exec app bash
```

## Создаем С cmake
```bash
cmake -S . -B build
cmake --build build
```
Then execute:
 ```bash
 ./main file.tsv n_events
 ```
