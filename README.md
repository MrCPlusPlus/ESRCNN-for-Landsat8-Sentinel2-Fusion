# ESRCNN-for-Landsat8-Sentinel2-Fusion

## To run this code you need the following:

Python3
Pytorch 0.3.0

## usage:
1. Get into generate_data folder to run codes inside, which will help you generate training samples. e.g. 
```bash
python3 generate_data_S2self.py
```
2. perform training. e.g.
```bash
python3 train_S2self.py
```
3. perform testing. e.g.
```bash
python3 test_S2self.py
```

Notice: you can change parameters which are stored in .json file. (options/test or options/train)
