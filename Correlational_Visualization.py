import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

#uncomment Lines 8 and 14 to change size of output

n = 20

style.use('ggplot')

def visualize_data():
    df = pd.read_csv('Data_Acquisition/joined_close_corr.csv')
    #df = df.iloc[0:n,0:n ]
    print(df.shape)

    df.set_index("Unnamed: 0", inplace=True)
    data1 = df.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df.columns
    row_labels = df.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    ax1.tick_params(axis='both', which='major', labelsize=5)
    ax1.tick_params(axis='both', which='minor', labelsize=5)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()


visualize_data()