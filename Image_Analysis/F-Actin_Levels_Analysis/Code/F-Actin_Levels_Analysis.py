############################################################################################################
#----------------------------------------------------------------------------------------------------------#
#------------------------------------------ F-Actin Levels Analysis ---------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#---------------------------------------------- Adrien Hallou ---------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------- University of Cambridge ----------------------------------------#
#----------------------------------------------------------------------------------------------------------#                         
#------------------------------------------------ 2018-2021 -----------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
############################################################################################################

############################################# Python libraries #############################################

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

# Use Latex for font rendering in figures
rc('font',**{'family':'sans-serif','sans-serif':['cm']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']

################################################ Data Path ###############################################

"""
Provide local paths to input data and output data  
"""

input_path= '/home/adrien/Desktop/Image_Analysis/F-Actin_Levels_Analysis/Data/Data_'

output_path='/home/adrien/Desktop/Image_Analysis/F-Actin_Levels_Analysis/Plots/'

########################################### Data Structure ###############################################

"""
~~~~ Raw data structure (Cells) ~~~~

c0=index 

c1= surface area
c2= centroid x position
c3= centroid y position 
c4= perimeter
c5= bounding box centre x position
c6= bounding box centre y position
c7= bounding box width
c8= bounding box height
c9= ellipse major axis
c10= ellipse minor axis 
c11= ellipse major axis angle (with x axis)
c12= circularity
c13= Feret diameter
c14= Feret x start position 
c15= Feret y start position
c16= Feret angle (with x axis)
c17= minimum Feret diameter
c18= Aspect ratio
c19= Roundness
c20= Solidity

c21= Mean Intensity DAPI
c22= Std Intensity DAPI
c23= Mode Intensity DAPI
c24= Min Intensity DAPI
c25= Max Intensity DAPI
c26= Xm Intensity DAPI
c27= Ym Intensity DAPI
c28= Integrated Intensity DAPI
c29= Median Intensity DAPI
c30= Raw Integrated Intensity DAPI

c31= Mean Intensity F-Actin
c32= Std Intensity F-Actin
c33= Mode Intensity F-Actin
c34= Min Intensity F-Actin
c35= Max Intensity F-Actin
c36= Xm Intensity F-Actin
c37= Ym Intensity F-Actin
c38= Integrated Intensity F-Actin
c39= Median Intensity F-Actin
c40= Raw Integrated Intensity F-Actin
"""

"""
~~~~ Data structure dictionaries ~~~~

"""

# Dictonary of stainings mean intensities
stn_dic={'DAPI':21,'F-Actin':31}

# Dictonary of stainings integrated intensities
stn_dic_int={'DAPI':28,'F-Actin':38}

# Dictonary of stainings raw intensities
stn_dic_raw={'DAPI':30,'F-Actin':40}

# Dictonary of membrane mean stainings 
stn_dic_mb={'F-Actin':1}
#stn_dic_mb={'F-Actin':9}

# Dictonary of geometrical and morphological descriptors
gm_dic={'area':1,'perimeter':4,'l_axis':9,'s_axis':10,'angle':11,'circularity':12,'AR':18,'roundness':19,'solidity':20}

# Dictonary of time points
tpts_dic={'P7':0,'P28':1,'P70':2}

"""
~~~~ Data structure list ~~~~

"""
# List of stainings
l_stn=['DAPI','F-Actin']
# Number of stainings
N_stn=len(l_stn)

# List of membrane stainings
l_stn_mb=['F-Actin']
# Number of stainings
N_stn_mb=len(l_stn_mb)

# List of time points
t_pts=['P7','P28','P70']
# Number of time points
N_tp=len(t_pts)

# List of biological replicates
b_rs=['BR1','BR2','BR3']
# Number of biological replicates
N_br=len(b_rs)

# List of technical replicates
t_rs=['TR1','TR2','TR3']
# Number of biological replicates
N_tr=len(t_rs)

"""
~~~~ Scale ~~~~

"""
# Image size (microns)
L=129.42

################################################# Data analysis functions ###############################################

def remove_values_from_list(the_list, val):
    """
    Function to remove multiple occurences of a given element in a list.
    
    """
    return ([value for value in the_list if value != val])
    
def f_max_int_tot(b_rs,t_rs,t_pts,stn):
    """
    Find maximum intensity of a given staining amongst all biological replicate and time points.
    
    Inputs
    ----------
    b_rs: list of biological replicate | t_rs: list of technical replicate | t_pts: list of time points  
    stn: staining
    
    Ouputs
    ----------
    max_int: maximum intensity value, float
    """
    max_int_list=[]
    for i in range (0, len(b_rs)):
        for j in range (0, len(t_rs)):
            for k in range (0, len(t_pts)):
                D=np.genfromtxt(input_path+b_rs[i]+'_'+t_rs[j]+'_'+t_pts[k]+'.csv', delimiter=",", skip_header=1) # Load data file
                if stn=='DAPI' or stn=='Sox2':
                    d=np.divide(D[:,stn_dic[stn]],np.mean(D[:,stn_dic[stn]])) # Extract relevent data and normalise them by mean
                else:
                    d=np.divide(D[:,stn_dic[stn]],np.min(D[:,stn_dic[stn]])) # Extract relevent data and normalise them by mean
                max_int_list.append(np.max(d))
    max_int_ar=np.asarray(max_int_list)
    max_int=np.max(max_int_ar)
    return(max_int)

######################################################## Plots ##########################################################  

#################### Figure 5 ####################  
 
# Fig.5 - Boxplots of F-Actin Membrane Intensity 
def Fig5_memb_intensity_bxplts(b_rs,t_rs,t_pts,stn):
    # Create an array to store the final data
    data_c=np.zeros((len(t_pts)), dtype=object)
    for i in range (0, len(t_pts)):
        l_d=[]
        for j in range (0, len(b_rs)):
            for k in range (0, len(t_rs)):
                data=np.genfromtxt(input_path+'Membrane_'+b_rs[j]+'_'+t_rs[k]+'_'+t_pts[i]+'.csv', delimiter=",", skip_header=1) # Load data file 
                Int= data[stn_dic_mb[stn]]
                l_d.append(Int)
        data_c[i]=l_d
    # Save data as csv file with appropriate header
    b = list(data_c)
    df = pd.DataFrame(b)
    dfs=df.T
    dfs.to_csv(output_path +'Fig5_Memb_Int_Boxplots_'+stn+'_Data.csv', header=t_pts, sep = ",")
    # Import saved data for plotting
    data_p=np.genfromtxt(output_path + 'Fig5_Memb_Int_Boxplots_'+stn+'_Data.csv', delimiter=",", skip_header=1)
    # Plot 
    fig1,axes = plt.subplots(1,1,figsize=(4.4,4.0))
    axes.plot
    axes= sns.swarmplot(data=list(data_p[:,1:].T), size=4.0, color='0.25', alpha=0.75, zorder=1)
    axes=sns.boxplot(data=list(data_p[:,1:].T), color="white", width=0.6, whis=1.5, showmeans=True, meanline=True, showfliers=False, whiskerprops=dict(linewidth=1.5, color='black', alpha=0.7), capprops=dict(linewidth=1.5, color='black', alpha=0.7), boxprops=dict(linewidth=1.5, edgecolor='black', alpha=0.7), medianprops=dict(linestyle='-', linewidth=1.5, color='black',alpha=0.7), meanprops=dict(linestyle='--', linewidth=1.5, color='red', alpha=0.7), zorder=1)
    axes.set_ylim([0.0,100.0]) # 8 / 10 / 110
    axes.set_ylabel(str(stn) + r'\, intensity (a.u)', size=22)
    axes.set_xticklabels(t_pts, fontsize=22) 
    axes.tick_params(axis='both', which='major', labelsize=20)
    fig1.tight_layout()
    # Save Fig
    fig1.savefig(output_path+'Fig5_Memb_Int_Boxplots_'+stn+'.pdf', dpi=400)
    fig1.savefig(output_path+'Fig5_Memb_Int_Boxplots_'+stn+'.png', dpi=400)
    return()  
    
################################################## Plots Commands ######################################################  

f5a=Fig5_memb_intensity_bxplts(b_rs,t_rs,t_pts,'F-Actin')

