'''
 # You may use, distribute and modify this code freely
 # Contributors Thomas A. Fink
 # Color Schema https://clrs.cc/
 # Data Provided by ETH Zurich https://utd19.ethz.ch/
 # An example of the macroscopic fundamental diagram for the city of Bolton, England
'''
import os
import numpy as np; np.random.seed(0)
from matplotlib import pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
from numpy import *
import matplotlib as mpl


## ------------------ Speed Density Diagram
def speed_density_diagram(df):

    df_speed_density = df.loc[
                     (df.SPEED< 100) &
                     (df.SPEED > 0) &
                     (df.DENSITY< 250) &
                     (df.DENSITY> 0), :]

    ax = df_speed_density.plot.scatter(x="DENSITY", y="SPEED", figsize=(8, 8), marker='.', alpha=0.5,  color='#333333', s=3)

    # Set axis minimums to 0
    plt.axis([0, None, 0, None])

    
    Uf = df_speed_density["SPEED"].max(); # Uf
    Kjam = df_speed_density["DENSITY"].max(); # Kjam

    # Calculate the midpoint of the diagonal
    mid_x = (0 + Kjam) / 2
    mid_y = (Uf + 0) / 2

    # Draw the line from origin to the max half dashed from midway point
    plt.plot([0,mid_x], [Uf, mid_y], 'k-', lw=1.5, color= '#ffd700',)
    plt.plot([mid_x, Kjam], [mid_y, 0], 'k--', lw=1.5, color= '#ffd700',)


    # Draw the line to y maximum from y axis
    plt.hlines(y = mid_y,  xmin=0, xmax=mid_x, color= '#ffd700', linestyles="--")
    # Line to y maximum from x axis
    plt.vlines(x = mid_x,  ymin=0, ymax=mid_y, color= '#ffd700', linestyles="--")


    # Add line label to Qcap
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(mid_x+0.3, 1, "$\it{Q}$$_\mathbf{cap}$", color="#ffd700", ha="left")

    # Add line label to Uf
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0.3, Uf+1, "$\it{U}$$_\mathbf{f}$", color="#ffd700", ha="left")

    # Add line label to Ucap
    plt.text(mid_x,mid_y+5,"(" + str(round(mid_x)) + ", " + str(round(mid_y)) + ")", color="#ffd700", weight="bold")

    # Add line label to Kcap
    trans = transforms.blended_transform_factory(ax.get_xticklabels()[0].get_transform(), ax.transData)
    ax.text(0.3, mid_y+1, "$\it{K}$$_\mathbf{cap}$", color="#ffd700", ha="left", fontname = 'Times New Roman' )

    # Add line label to Kjam
    trans = transforms.blended_transform_factory(ax.get_xticklabels()[0].get_transform(), ax.transData)
    ax.text(Kjam+0.3, 1, "$\it{K}$$_\mathbf{j}$", color="#ffd700", ha="left", fontname = 'Times New Roman' )

    # Label the axes
    ax.set_xlabel('DENSITY (vehicles/kilometer/lane)',fontweight="bold")
    ax.set_ylabel('SPEED (kilometers/minutes)',fontweight="bold")
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20

    # Add a title
    ax.title.set_color('#333333')
    plt.title("Speed Density Diagram", fontweight="bold", fontdict={'fontsize':14}, pad=20)

    plt.tight_layout()

    # Save the diagram
    print("Saving diagram...")
    plt.savefig(OS_PATH + "/output/mfd_speed_density.jpg", dpi=1000)

    # Show the diagram
    print("Generating diagram...")
    plt.show()


## ------------------  Speed Flow Diagram
def speed_flow_diagram(df):

    df_speed_flow = df.loc[
                        (df.FLOW< 2000) &
                        (df.FLOW > 0) &
                        (df.SPEED< 250) &
                        (df.SPEED> 0), :]

    ax = df_speed_flow.plot.scatter(x="FLOW", y="SPEED", figsize=(8, 8), marker='.', alpha=0.5,  color='#333333', s=3)


    # Find the x value to the y maximum change to numpy arrays
    x_data = np.array([])
    for row in df_speed_flow["FLOW"]:
        x_data = np.append(x_data, row)
        
    y_data = np.array([])
    for row in df_speed_flow["SPEED"]:
        y_data= np.append(y_data, row)
    
    # Resort the arrays
    order = x_data.argsort()
    y_data = y_data[order]
    x_data = x_data[order]

    max_x = df_speed_flow["FLOW"].max();
    max_y= np.interp(max_x, x_data, y_data,  left=None, right=None, period=None)



    # Line to x maximum from y axis
    plt.plot([0, max_x], [0, max_y], 'k--', lw=1.5, color= '#ffd700',)
    # Line to y maximum from y axis
    plt.hlines(y = max_y,  xmin=0, xmax=max_x, color= '#ffd700', linestyles="--")
    # Line to y maximum from x axis
    plt.vlines(x = max_x,  ymin=0, ymax=max_y, color= '#ffd700', linestyles="--")


    # Add line label to Qcap
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(max_x+9, 3, "$\it{Q}$$_\mathbf{cap}$", color="#ffd700", ha="left")

    # Add line label to Kcap
    plt.text(max_x+12,max_y+3,"$\it{K}$$_\mathbf{cap}$ (" + str(round(max_x,2)) + ", " + str(round(max_y)) + ")", color="#ffd700", weight="bold")

    # Add line label to Ucap
    trans = transforms.blended_transform_factory(ax.get_xticklabels()[0].get_transform(), ax.transData)
    ax.text(9, max_y+3, "$\it{U}$$_\mathbf{cap}$", color="#ffd700", ha="left", fontname = 'Times New Roman' )

    # Set axes limits
    plt.axis([0, max_x +50, 0, max_y +50])


    # Label the axes
    ax.set_xlabel('FLOW (vehicles/15 minutes/lane))', fontweight="bold")
    ax.set_ylabel('SPEED (kilometers/15 minutes)', fontweight="bold")
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20

    # Add a title
    ax.title.set_color('#333333')
    plt.title("Speed Flow Diagram", fontweight="bold", fontdict={'fontsize':14}, pad=20)

    plt.tight_layout()

    # Save the diagram
    print("Saving diagram...")
    plt.savefig(OS_PATH + "/output/mfd_speed_flow.jpg", dpi=1000)

    # Show the diagram
    print("Generating diagram...")
    plt.show()





## ------------------ Flow Density Diagram
def flow_density_diagram(df):

    # Remove outliers
    df_flow_density = df.loc[
            (df.FLOW< 2000) &
            (df.FLOW > 0) &
            (df.DENSITY< 50) &
            (df.DENSITY> 0), :]

    Kjam = df_flow_density["DENSITY"].max()

    # Plot the scatter diagram
    ax = df_flow_density.plot.scatter(x="DENSITY", y="FLOW", figsize=(8, 8), marker='.', alpha=0.5,  color='#333333', s=3)

    # Find the x value to the y maximum change to numpy arrays
    x_data = np.array([])
    for row in df_flow_density["DENSITY"]:
        x_data = np.append(x_data, row)
        
    y_data = np.array([])
    for row in df_flow_density["FLOW"]:
        y_data= np.append(y_data, row)
        
    # resort the arrays
    order = y_data.argsort()
    y_data = y_data[order]
    x_data = x_data[order]

    max_y = df_flow_density["FLOW"].max()
    max_x = np.interp(max_y, y_data, x_data,  left=None, right=None, period=None)


    # line from origin to the max
    plt.plot([0, max_x], [0, max_y], 'k--', lw=1.5, color= '#ffd700',)
    # line to y maximum from y axis
    plt.hlines(y = max_y,  xmin=0, xmax=max_x, color= '#ffd700', linestyles="--")
    # line to y maximum from x axis
    plt.vlines(x = max_x,  ymin=0, ymax=max_y, color= '#ffd700', linestyles="--")


    # Add line label to Qcap
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0.3, max_y+10, "$\it{Q}$$_\mathbf{cap}$", color="#ffd700", ha="left")

    # Add line label to Ucap
    plt.text(max_x+0.7,max_y+10,"$\it{U}$$_\mathbf{cap}$ (" + str(round(max_x,2)) + ", " + str(round(max_y)) + ")", color="#ffd700", weight="bold")

    # Add line label to Kcap
    trans = transforms.blended_transform_factory(ax.get_xticklabels()[0].get_transform(), ax.transData)
    ax.text(max_x+0.3, 10, "$\it{K}$$_\mathbf{cap}$", color="#ffd700", ha="left", fontname = 'Times New Roman' )

    # Add line label to Kjam
    trans = transforms.blended_transform_factory(ax.get_xticklabels()[0].get_transform(), ax.transData)
    ax.text(Kjam+0.3, 10, "$\it{K}$$_\mathbf{jam}$", color="#ffd700", ha="left", fontname = 'Times New Roman' )

    # Prep polynomial function
    x =  df_flow_density["DENSITY"]
    fo = lambda x: -3*x**2+ 1.*x +20. 
    f = lambda x: fo(x) + (np.random.normal(size=len(x))-0.5)*4
    y = f(x)

    # Polynomial fit
    def fit(ax, x,y, sort=True):
        z = np.polyfit(x, y, 1)
        fit = np.poly1d(z)
        print(fit)
        if sort:
            x = np.sort(x)
        ax.plot(x, fit(x), label="Polynomial f(x)^1: " '''+ str(fit)''', lw=1.5, color= '#ffd700', alpha=1)  
        ax.legend()

    # Plot the Polynomial fit
    fit(ax, df_flow_density["DENSITY"],df_flow_density["FLOW"], sort=True) 
    
    # Set axis limits
    plt.axis([0, max_x+50, 0, max_y+50])

    # Label the axes
    ax.set_xlabel('DENSITY (vehicles/kilometer/lane)', fontweight="bold")
    ax.set_ylabel('FLOW (vehicles/15 minutes/lane)', fontweight="bold")
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20


    # Add a title
    ax.title.set_color('#333333')
    plt.title("Density Flow Diagram", fontweight="bold", fontdict={'fontsize':14}, pad=20)

    plt.tight_layout()

    # Save the diagram
    print("Saving diagram...")
    plt.savefig(OS_PATH + "/output/mfd_density_flow.jpg", dpi=1000)

    # Show the diagram
    print("Generating diagram...")
    plt.show()


### Fetch the data from eth_bolton_sensor_data_utd19.csv

OS_PATH = os.path.dirname(os.path.realpath('__file__'))

# Data Import Path
MST_SENSOR_DATA_CSV   = OS_PATH + '/data/eth_bolton_sensor_data_utd19.csv'

# Data Import
df = pd.read_csv(MST_SENSOR_DATA_CSV)

# Keep only relevant columns
df = df.loc[:, ("DATE", "INTERVAL", "ID", "FLOW" ,"DENSITY","SPEED", "CITY")]

# Convert to dataframe
df = pd.DataFrame(df)

# Drop all rows with any NaN and NaT values
df = df[df['FLOW'].notna()]
df = df[df['SPEED'].notna()]
df = df[df['DENSITY'].notna()]

df =  df.loc[(df!=0).any(axis=1)]

# Error Removal
df = df.drop_duplicates(subset='DENSITY', keep="last")

# Query
#df = df[(df['DATE'].str.contains("2017-11-13")==True)]


# Set default chart font
plt.rcParams["font.family"] = "Times New Roman"

# Set data junk size for polynomial trends
mpl.rcParams['agg.path.chunksize'] = 10000

# Set defaults for all charts
params = {        
        'lines.linewidth': 2,
        'figure.titlesize': 14,
        'figure.titleweight': 'bold',
        'font.family':  'Times New Roman',
        'font.weight':  'bold',
        'text.color': '#333333',
        'axes.titlesize': 12,
        'axes.titlecolor': '#333333',
        'axes.titleweight': 'bold',
        'axes.edgecolor': '#333333',
        'axes.labelweight': 'bold',
        'axes.labelsize': 10,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'legend.loc' : 'upper right',
        'figure.constrained_layout.use': True,
        }
plt.rcParams.update(params)


speed_density_diagram(df)
speed_flow_diagram(df)
flow_density_diagram(df)