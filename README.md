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

The following code uses above mathematical form and aggregates the green index with 50 images closest to the transaction point. The final result file is in *Green Index_Spatial Interpolation.csv*.
```python
import pandas as pd
from haversine import haversine

data_df = pd.read_('Write your path\Data.csv')
green_df = pd.read_csv('Write your path\Street Greenness.csv')

Aggregated_Green_Index = []
Aggregated_Green_Index_Distance = []

num = 1
for y, x, ind in zip(data_df['y'], data_df['x'], data_df.index):
  distance = []

  for gr_y, gr_x, hgvi in zip(green_df['Latitude'], green_df['Longitude'], green_df['Green Index']):
    dis = haversine([y,x], [gr_y, gr_x], unit='km')
    distance.append([x,y,gr_x,gr_y,dis,hgvi])
  dis_df = pd.DataFrame(distance)
  dis_df.columns = ['x','y','gr_x','gr_y','distance','HGVI']
  dis_df = dis_df.sort_values('distance', ascending=True)

  # Extract the 50 nearest green indices
  dis_df_50 = dis_df.iloc[:50]

  mean_hgvi_50 = dis_df_50['HGVI'].mean()
  mean_dis_50 = dis_df_50['distance'].mean()

  Aggregated_Green_Index.append(mean_hgvi_50)
  Aggregated_Green_Index_Distance.append(mean_dis_50)

data_df['Green Index'] = Aggregated_Green_Index
data_df['Green Index_d'] = Aggregated_Green_Index_Distance
data_df.to_csv('Green Index_Spatial Interpolation.csv',index=False,encoding='utf-8-sig')
```
Through this process, we can get the green index for all points of transaction and all information of hedonic variables including green index is in *Hedonic Dataset.xlsx*.
