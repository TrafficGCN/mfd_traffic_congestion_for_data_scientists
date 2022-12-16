'''
 # You may use, distribute and modify this code freely
 # Contributors Thomas A. Fink
 # Color Schema https://clrs.cc/
'''
### Import Libraries
import os
from os import remove
from tkinter import font
from numpy import append, size
import pandas as pd
import geopandas
#import matplotlib.pyplot as plt
import plotly.express as px
from rasterio.plot import show as rioshow
import plotly.graph_objects as go


OS_PATH = os.path.dirname(os.path.realpath('__file__'))


### Fetch the data from eth_bolton_sensors_utd19.csv

# Data Import Path
SENSORS_CSV = OS_PATH +'/data'+'/eth_bolton_sensors_utd19.csv';

# Data Import
df1 = pd.read_csv(SENSORS_CSV)

# Keep only relevant columns
df = df1.loc[:, ("ID", "LENGTH", "POSITION", "FCLASS", "ROAD", "LIMIT", "CITYCODE", "LINKID", "LANES", "LONGITUDE","LATITUDE")]


# Remove sensors of missing geocoordinates
df = df[df['LONGITUDE'].notna()]
df = df[df['LATITUDE'].notna()]

#### Data Plotting
figBolton = px.scatter_mapbox(
    df, 
    lat="LATITUDE", 
    lon="LONGITUDE", 
    color="CITYCODE", 
    hover_name='ID', 
    hover_data=['ROAD'], 
    zoom=14.13,
    color_discrete_sequence=["rgb(255, 220, 0)", "rgb(255, 133, 27)", "rgb(255, 65, 54)", "rgb(133, 20, 75)", "rgb(240, 18, 190)", "rgb(177, 13, 201)", "rgb(57, 204, 204)", "rgb(127, 219, 255)", "rgb(0, 116, 217)", "rgb(0, 31, 63)", "rgb(61, 153, 112)", "rgb(46, 204, 64)", "rgb(1, 255, 112)", "rgb(170, 170, 170)", "rgb(221, 221, 221)", "rgb(255, 255, 255)", "rgb(128, 0, 0)", "rgb(255, 255, 0)","rgb(255, 0, 0)","rgb(128, 128, 0)","rgb(1, 255, 112)","rgb(0, 128, 128)","rgb(0, 116, 217)","rgb(127, 219, 255)","rgb(17, 17, 17)"],
    title="<span style='font-size: 32px;'><b>Bolton Sensor Map</b></span>",
    opacity=.9,
    width=2000,
    height=1000,
    center=go.layout.mapbox.Center(
            lon=-2.43,
            lat=53.58
        ),
    size_max=1
    )

# Now using Mapbox
figBolton.update_layout(mapbox_style="light", 
                  mapbox_accesstoken="###############################################################################################################################",
                  legend=dict(yanchor="top", y=1, xanchor="left", x=0.9),
                  title=dict(yanchor="top", y=.85, xanchor="left", x=0.075),
                  font_family="Times New Roman",
                  font_color="#333333",
                  title_font_size = 32,
                  font_size = 18)

# Save map in output folder
print("Saving image to output folder...");
#figBolton.write_image(OS_PATH + '/output/bolton_sensor_map.jpg',format='jpg',engine='kaleido', scale=5)

# Show map in web browser
print("Generating map in browser...");
figBolton.show()