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
from haversine import haversine

entropy_bs = pd.read_excel('Write your path\df_bs_del.xlsx')
entropy_bs_1 = entropy_bs[entropy_bs['delentropy'].isna()].reset_index()
df_bs = entropy_bs[['위도', '경도', 'delentropy']].copy()
df_bs = df_bs[df_bs['delentropy'].notna()].drop_duplicates().reset_index(drop=True)

Aggregated_Entropy = []
Aggregated_Entropy_Distance = []
entropy_bs['delentropy_d'] = ''

a = 0

for y, x, ind in zip(entropy_bs_1['위도'], entropy_bs_1['경도'], entropy_bs_1.index):
  distance = []

  for en_y, en_x, hgvi in zip(df_bs['위도'], df_bs['경도'], df_bs['delentropy']):
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

  print(a, '/', len(entropy_bs_1))

entropy_bs_1['delntropy'] = Aggregated_Entropy
entropy_bs_1['delentropy_d'] = Aggregated_Entropy_Distance

# Filling missing values
for i in range(0,len(entropy_bs_1)):
  entropy_bs['delentropy'][entropy_bs_1['level_0'][i]] = Aggregated_Entropy[i]
  entropy_bs['delentropy_d'][entropy_bs_1['level_0'][i]] = Aggregated_Entropy_Distance[i]

entropy_bs.to_csv('Write your path\spatial_interpolation_bs.csv',index=False,encoding='utf-8-sig')
```
Through this process, we can get the green index for all points of transaction and all information of hedonic variables including green index is in *Hedonic Dataset.xlsx*.
