############################################################################################################
#----------------------------------------------------------------------------------------------------------#
#------------------- Oesophagus Epithelium P-MyosinII & Vinculin Stainings Data Analysis ------------------#
#----------------------------------------------------------------------------------------------------------#
#---------------------------------------------- Adrien Hallou ---------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------- University of Cambridge ----------------------------------------#
#----------------------------------------------------------------------------------------------------------#                         
#------------------------------------------------ 2018-2021 -----------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
############################################################################################################

############################################# Python libraries #############################################

# Import python libraries
import numpy as np
import scipy as sp
import matplotlib as mpl
import seaborn as sns
import pandas as pd

# Import python sub-libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import built-in functions from matplotlib
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Use Latex for font rendering in figures
rc('font',**{'family':'sans-serif','sans-serif':['cm']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']


############################################## Data Path ###############################################

"""
Provide local paths to input data and output data  
"""

input_path='/home/adrien/Desktop/Image_Analysis_McGinn_Nat-Cell-Biol_2021/Oesophagus_Epithelium_P-MyosinII_Vinculin_Stainings/Data/Data_'

output_path='/home/adrien/Desktop/Image_Analysis_McGinn_Nat-Cell-Biol_2021/Oesophagus_Epithelium_P-MyosinII_Vinculin_Stainings/Plots/'

########################################## Data Structure #############################################

"""
~~~~ Data structure dictionaries ~~~~

"""

# Dictonary of stainings mean fluorescence intensities  
stn_dic={'P-Myosin':1, 'Vinculin':11}

# Dictonary of time points
tpts_dic={'P7':0,'P28':1,'P70':2}

"""
~~~~ Data structure list ~~~~

"""
# List of stainings
l_stn=['P-Myosin','Vinculin']
# Number of stainings
N_stn=len(l_stn)

# List of time points
t_pts=['P7','P28','P70']
# Number of time points
N_tp=len(t_pts)

# List of biological replicates
b_rs=['BR1','BR2','BR3','BR4']
# Number of biological replicates
N_br=len(b_rs)

# List of technical replicates
t_rs=['TR1']
# Number of biological replicates
N_tr=len(t_rs)

"""
~~~~ Scale ~~~~

"""
# Image size (microns)
L=129.42
    
################################################## Plots #####################################################  

#################### Figure 5 ####################  
    
# Fig.5 - Boxplots of stainings mean fluoresence intensities for a given tissue region 
def Fig5_stain_intensity_bxplts(b_rs,t_rs,t_pts,stn,tiss_reg):
    # Create an array to store the final data
    data_c=np.zeros((len(t_pts)), dtype=object)
    for i in range (0, len(t_pts)):
        l_d=[]
        for j in range (0, len(b_rs)):
            for k in range (0, len(t_rs)):
                data=np.genfromtxt(input_path+'Stainings_'+b_rs[j]+'_'+t_rs[k]+'_'+t_pts[i]+'_'+tiss_reg+'.csv', delimiter=",", skip_header=1) # Load data file 
                Int= data[stn_dic[stn]]
                l_d.append(Int)
        data_c[i]=l_d
    # Save data as csv file with appropriate header
    b = list(data_c)
    df = pd.DataFrame(b)
    dfs=df.T
    dfs.to_csv(output_path +'Fig5_Stain_Int_Boxplots_'+stn+'_'+tiss_reg+'_Data.csv', header=t_pts, sep = ",")
    # Import saved data for plotting
    data_p=np.genfromtxt(output_path + 'Fig5_Stain_Int_Boxplots_'+stn+'_'+tiss_reg+'_Data.csv', delimiter=",", skip_header=1)
    print(data_p[:,1:])
    # Plot 
    fig,axes = plt.subplots(1,1,figsize=(4.4,4.0))
    axes.plot
    axes= sns.swarmplot(data=list(data_p[:,1:].T), size=4.0, color='0.25', alpha=0.75, zorder=1)
    axes=sns.boxplot(data=list(data_p[:,1:].T), color="white", width=0.6, whis=1.5, showmeans=True, meanline=True, showfliers=False, whiskerprops=dict(linewidth=1.5, color='black', alpha=0.7), capprops=dict(linewidth=1.5, color='black', alpha=0.7), boxprops=dict(linewidth=1.5, edgecolor='black', alpha=0.7), medianprops=dict(linestyle='-', linewidth=1.5, color='black',alpha=0.7), meanprops=dict(linestyle='--', linewidth=1.5, color='red', alpha=0.7), zorder=1)
#    axes.set_ylim([0.5,10.0]) # 8 / 10 / 110
    axes.set_ylabel(str(stn) + r'\, intensity (a.u)', size=20)
    axes.set_xticklabels(t_pts, fontsize=22) 
    axes.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    # Save Fig
    fig.savefig(output_path+'Fig5_Stain_Int_Boxplots_'+stn+'_'+tiss_reg+'.pdf', dpi=400)
    fig.savefig(output_path+'Fig5_Stain_Int_Boxplots_'+stn+'_'+tiss_reg+'.png', dpi=400)
    return()  
    
################################################## Plot Commands ######################################################  

#f5a=Fig5_stain_intensity_bxplts(b_rs,t_rs,t_pts,'Vinculin','Stroma')
    
    
    
    
    