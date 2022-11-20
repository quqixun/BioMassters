import rasterio
# import matplotlib.pyplot as plt
from rasterio.plot import show


fp = r'./fff05995_S1_04.tif'
src = rasterio.open(fp)
array = src.read()
print(array.shape)
# mention band no. in read() method starting from 1 not 0
show(src.read(4))
