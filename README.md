# Floor Complexity Data Records
<img src="https://img.shields.io/badge/Google Colab-F9ABOO?style=for-the-badge&logo=Google Colab&logoColor=white" link='https://colab.google/'> <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">  
We introduce an indicator called 'Floor complexity', based on the floor plan images in Busan, Daegu, Daejeon, and Gwangju.

To drive the floor complexity index, we collect floor plan images, refine images, calculate Delentropy, and implement spatial interpolation. 
<p align="center">
  <img src = "README_image/flowchart.png" width = "30%"> <br>
  Figure 1. Steps to obtain floor complexity index
</p>

This four-step process is necessary to effectively compute the green index, and sample data was stored in the 'DATA' folder to replicate this calculation.

Data in this repository consists of Excel and CSV files:

## Image Preprocessing
Since Delentropy is calculated by changes of gray-scale level between pixels, image preprocessing is necessary. The raw data of floor plan images is in *'RAW DATA'* folder.
To refine the images for accurate calculation, we detect edge image, edges of floor plan images, create closed form images and blur outside of closed form images.

<p align="center">
  <img src = "/README_image/spatial interpolation.png" width = "60%"> <br>
  Figure 3. Graphical description of spatial interpolation.
</p>   

## Calculation of Delentropy

```python
import os
import pandas as pd
from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def grayscale(image_path):
  image = Image.open(image_path)
  # Converted to grayscale
  grayscale_image = image.convert('L')
  grayscale_array = np.array(grayscale_image)
  return grayscale_array

def delentropy_2d(image):
  image_with_nan = np.where(image == 255, np.nan, image)

  gradient_x = np.array([[1,-1], [1,-1]])/2
  gradient_y = np.array([[1,1], [-1, -1]])/2

  fx = convolve2d(image_with_nan, gradient_x, mode='valid', boundary='fill', fillvalue=np.nan)
  fy = convolve2d(image_with_nan, gradient_y, mode='valid', boundary='fill', fillvalue=np.nan)

  histogram, _, _ = np.histogram2d(fx[~np.isnan(fx)], fy[~np.isnan(fy)], bins=[511, 511], range=[[-255, 255], [-255, 255]])

  probability = histogram / np.sum(histogram)
  delentropy = -np.sum(probability * np.log2(probability + np.finfo(float).eps))
  return delentropy

area = ['bs', 'dg', 'dj', 'gw']

for i in range(0,len(area)):
    name = area[i]
    index = os.listdir(f"image_{name}")
    df_index = [name.rstrip('.jpg') for name in index]
    image_file = f"image_{name}\\"

    busan = pd.DataFrame()
    busan['index'] = ''
    busan['delentropy'] = ''

    del_value = []
    del_index = []
    a = 0

    for i in range(0, len(os.listdir(f"image_{name}"))):
        image_path = str(image_file) + str(df_index[i]) + '.jpg'
        gray_image = grayscale(image_path)
        del_image = delentropy_2d(gray_image)
        del_index.append(df_index[i])
        del_value.append(del_image)
        print(a, '/', len(df_index))
        a+=1

    busan['index'] = del_index
    busan['delentropy'] = del_value
    busan.to_csv(f'Delentropy\del_{name}.csv', index=False)

```


## Spatial Interpolation
Spatial interpolation step can be utilized to remedy the uneven spatial distribution of GSV images.   
To implement the spatial interpolation method, refer to the sample data file named *'Data.csv'* and *Street Greenness.csv*.    
The columns required to effectively manage the green index are as follows:   

*Data.csv*
- x: Longitude in the Cartesian coordinate system of the transaction point
- y: Latitude in the Cartesian coordinate system of the transaction point
   
*Street Greenness.csv*
- Longitude: Longitude of GSV image
- Latitude: Latitude of GSV image
- Green Index: Calculated street greenness

Spatial interpolation requires the distance between two objects based on longitude and latitude. It can be obtained by using haversine formula as follows:

$$d_{\text{haversine}} = 2 \times R \times \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \text{lat}}{2}\right) + \cos(\text{lat}_p) \cos(\text{lat}_g) \sin^2\left(\frac{\Delta \text{lng}}{2}\right)}\right)$$
   
<p align="center">
  <img src = "/README_image/spatial interpolation.png" width = "60%"> <br>
  Figure 3. Graphical description of spatial interpolation.
</p>   

The following code uses above mathematical form and aggregates the green index with 50 images closest to the transaction point. The final result file is in *Green Index_Spatial Interpolation_bs.csv*.
```python
import pandas as pd
import pandas as pd
from haversine import haversine

area = ['bs', 'dg', 'dj', 'gw']

for i in range(0, len(area)):
    name = area[i]

    df = pd.read_excel(f'Delentropy\df_{name}.xlsx')
    df['Delentropy'] = ''
    delentropy = pd.read_csv(f'Delentropy\del_{name}.csv')

    df['index'] = df['index'].astype(str)
    delentropy['index'] = delentropy['index'].astype(str)

    del_df = pd.merge(df, delentropy, on=['index'], how ='left')
    del_df.drop(columns = ['Delentropy'], inplace=True)
    del_df.to_excel(f'Delentropy\df_{name}_del.xlsx', index=False)

    ## Spatial Interpolation
    del_df_1 = del_df[del_df['delentropy'].isna()].reset_index()
    dff = del_df[['위도', '경도', 'delentropy']].copy()
    dff = dff[dff['delentropy'].notna()].drop_duplicates().reset_index(drop=True)

    Aggregated_Entropy = []
    Aggregated_Entropy_Distance = []
    del_df['delentropy_d'] = ''

    a = 0

    for y, x, ind in zip(del_df_1['위도'], del_df_1['경도'], del_df_1.index):
        distance = []

        for en_y, en_x, hgvi in zip(dff['위도'], dff['경도'], dff['delentropy']):
            dis = haversine([y,x], [en_y, en_x], unit='km')
            distance.append([x,y,en_x,en_y,dis,hgvi])
        dis_df = pd.DataFrame(distance)
        dis_df.columns = ['x','y','en_x','en_y','distance','HGVI']
        dis_df = dis_df.sort_values('distance', ascending=True)

        # Extract the 100 nearest green indices
        dis_df_100 = dis_df.iloc[:100]

        mean_hgvi_100 = dis_df_100['HGVI'].mean()
        mean_dis_100 = dis_df_100['distance'].mean()

        Aggregated_Entropy.append(mean_hgvi_100)
        Aggregated_Entropy_Distance.append(mean_dis_100)

        a += 1

        print(a, '/', len(del_df_1))

    del_df_1['delntropy'] = Aggregated_Entropy
    del_df_1['delentropy_d'] = Aggregated_Entropy_Distance

    for i in range(0,len(del_df_1)):
        del_df['delentropy'][del_df_1['level_0'][i]] = Aggregated_Entropy[i]  # i번째 결측치의 원래 index에 entropy값 넣기
        del_df['delentropy_d'][del_df_1['level_0'][i]] = Aggregated_Entropy_Distance[i]

    del_df.to_csv(f'Delentropy\spatial_interpolation_{name}.csv',index=False,encoding='utf-8-sig')
```
Through this process, we can get the green index for all points of transaction and all information of hedonic variables including green index is in *Hedonic Dataset.xlsx*.
