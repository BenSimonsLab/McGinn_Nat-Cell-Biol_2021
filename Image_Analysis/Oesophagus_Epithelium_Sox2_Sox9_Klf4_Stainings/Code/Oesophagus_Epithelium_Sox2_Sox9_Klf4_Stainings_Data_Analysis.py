############################################################################################################
#----------------------------------------------------------------------------------------------------------#
#--------------------- Oesophagus Epithelium Sox2 Sox9 Klf4 Stainings Data Analysis -----------------------#
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
import matplotlib.gridspec as gridspec 

# Import built-in functions from matplotlib
from matplotlib import rc
from matplotlib.patches import Polygon, Ellipse
from matplotlib.collections import PatchCollection, PolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import built-in functions from Scipy
from scipy.spatial import Delaunay, Voronoi, ConvexHull, KDTree

# Import built-in functions from Skimage
from skimage.measure import regionprops
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import label, closing, square

# Use Latex for font rendering in figures 
rc('font',**{'family':'sans-serif','sans-serif':['cm']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']

############################################# Data Path ################################################

"""
Provide local paths to input data and output data  
"""

input_path = '/home/adrien/Desktop/Image_Analysis_McGinn_Nat-Cell-Biol_2021/Oesophagus_Epithelium_Sox2_Sox9_Klf4_Stainings/Data/Data_'

output_path = '/home/adrien/Desktop/Image_Analysis_McGinn_Nat-Cell-Biol_2021/Oesophagus_Epithelium_Sox2_Sox9_Klf4_Stainings/Plots/'

########################################## Data Structure #############################################

"""
~~~~ Raw data (CSV files) structure ~~~~

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

c31= Mean Intensity Sox2
c32= Std Intensity Sox2
c33= Mode Intensity Sox2
c34= Min Intensity Sox2
c35= Max Intensity Sox2
c36= Xm Intensity Sox2
c37= Ym Intensity Sox2
c38= Integrated Intensity Sox2
c39= Median Intensity Sox2
c40= Raw Integrated Intensity Sox2

c41= Mean Intensity Klf4 
c42= Std Intensity Klf4 
c43= Mode Intensity Klf4 
c44= Min Intensity Klf4 
c45= Max Intensity Klf4 
c46= Xm Intensity Klf4 
c47= Ym Intensity Klf4 
c48= Integrated Intensity Klf4 
c49= Median Intensity Klf4 
c50= Raw Integrated Intensity Klf4 

c51= Mean Intensity Sox9
c52= Std Intensity Sox9
c53= Mode Intensity Sox9
c54= Min Intensity Sox9
c55= Max Intensity Sox9
c56= Xm Intensity Sox9
c57= Ym Intensity Sox9
c58= Integrated Intensity Sox9
c59= Median Intensity Sox9
c60= Raw Integrated Intensity Sox9

"""

# Dictonary of stainings - Mean 
stn_dic={'DAPI':21,'Sox2':31,'Klf4':41,'Sox9':51}

# Dictonary of stainings - Integrated
stn_dic_int={'DAPI':28,'Sox2':38,'Klf4':48,'Sox9':58}

# Dictonary of stainings - Raw
stn_dic_raw={'DAPI':30,'Sox2':40,'Klf4':50,'Sox9':60}

# Dictonary of stainings colour maps
cmap_dic={'DAPI':'Blues','Sox2':'BuPu','Klf4':'RdPu','Sox9':'BuGn'}

# Dictonary of geometrical and morphological descriptors
gm_dic={'area':1,'perimeter':4,'l_axis':9,'s_axis':10,'angle':11,'circularity':12,'AR':18,'roundness':19,'solidity':20}

# Dictonary  of time points
tpts_dic={'P2':0,'P7':1,'P14':2,'P28':3,'P49':4,'P70':5}

# List of stainings
stn_lt=['DAPI','Sox2','Klf4','Sox9']
 # Number of AB stainings
N_stn_lt=len(stn_lt)

# List of geometrical and morphological descriptors
gm_lt=['area','perimeter','l_axis','s_axis','angle','circularity','AR','roundness','solidity']
 # Number of geometrical and morphological descriptors
N_gm_lt=len(gm_lt)

# List of time points
t_pts=['P2','P7','P14','P28','P49','P70']
t_pts_p=['P7','P14','P28','P49','P70']
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

# Image size (microns)
L=129.29

################################################# Data analysis functions ###############################################

def remove_values_from_list(the_list, val):
    """
    Function to remove multiple occurences of a given element in a list.
    
    """
    return [value for value in the_list if value != val]


def f_max_int(b_r,t_rs,t_pts,stn):
    """
    Find maximum intensity of a given staining for a given biological replicate and amongst all time points.
    
    Inputs
    ----------
    b_r: biological replicate | t_rs: list of technical replicate | t_pts: list of time points  
    stn: staining
    
    Ouputs
    ----------
    max_int: maximum intensity value, float
    """
    max_int_list=[]
    for i in range (0, len(t_rs)):
        for j in range (0, len(t_pts)):
            D=np.genfromtxt(input_path+b_r+'_'+t_rs[i]+'_'+t_pts[j]+'.csv', delimiter=",", skip_header=1) # Load data file
            if stn=='DAPI' or stn=='Sox2':
                d=np.divide(D[:,stn_dic[stn]],np.mean(D[:,stn_dic[stn]])) # Extract relevent data and normalise them by mean
            else:
                d=np.divide(D[:,stn_dic[stn]],np.min(D[:,stn_dic[stn]])) # Extract relevent data and normalise them by mean
            max_int_list.append(np.max(d))
    max_int_ar=np.asarray(max_int_list)
    max_int=np.max(max_int_ar)
    return(max_int)
    
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
  
def f_max_val(b_r,t_rs,t_pts,gm):
    """
    Find maximum value of a given geometrical/morphological descriptor for a given biological replicate and amongst all time points.
    
    Inputs
    ----------
    b_r: biological replicate | t_rs: list of technical replicate | t_pts: list of time points  
    gm: geometrical/morphological descriptor 
    
    Ouputs
    ----------
    max_val: maximum descriptor value, float
    """
    # Temporary fix to solve the issue that there is only 2 BR for P84
    if b_r=='BR3':
        t_pts=['P2','P28']
    max_val_list=[]
    for i in range (0, len(t_rs)):
        for j in range (0, len(t_pts)):
            D=np.genfromtxt(input_path+b_r+'_'+t_rs[i]+'_'+t_pts[j]+'.csv', delimiter=",", skip_header=1) # Load data file
            if gm=='area'or  gm=='perimeter' or gm=='l_axis' or gm=='s_axis' or gm=='angle' or gm=='circularity'or gm=='AR'or gm=='roundness' or gm=='solidity':
                d=D[:,gm_dic[gm]]
            else: 
                d=np.divide(np.sqrt(np.divide(np.multiply(4.0,D[:,1]), np.pi)),D[:,9]) # case of compactness 
            max_val_list.append(np.max(d))
    max_val_ar=np.asarray(max_val_list)
    max_val=np.max(max_val_ar)
    return(max_val)

def percentile_analysis(b_r,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L):
    """
    Derive p & q percentile values for stn1 & stn2 from arbitrary selected percentile thresholds p_stn1 & p_stn2.
    
    Save associated normalised intensities for stn1 & stn2.
    
    Inputs
    ----------
    b_r: biological replicate | t_rs: list of technical replicate | t_pts: list of time points  
    stn1: staining 1 | stn2: staining 2 
    p_stn1: percentile threshold for staining 1 |  p_stn2: percentile threshold for staining 2
    
    Ouputs
    ----------
    p,q: computed percentile thresholds for stn1 & stn2, float.
    data_s1, data_s2: normalised intensities for stn1 & stn2, np.array.
    """      
    # Import raw intensity data, normalise them according to their type and concatenate together the different technical replicate along a single column
    data_s1=np.zeros((len(t_pts)), dtype=object)
    data_s2=np.zeros((len(t_pts)), dtype=object)   
    for i in range (0, len(t_pts)):
        l_d1=[]
        l_d2=[]
        for j in range (0, len(t_rs)):
            # Extract relevent data and normalise them by minimum (Sox9 & Klf4) and by mean (DAPI & Sox2)
            data=np.genfromtxt(input_path+b_r+'_'+t_rs[j]+'_'+t_pts[i]+'.csv', delimiter=",", skip_header=1) # Load data file         
            if stn1=='DAPI' or stn1=='Sox2':
                d1=np.divide(data[:,stn_dic[stn1]],np.mean(data[:,stn_dic[stn1]]))     
            elif stn1=='Sox9' or stn1=='Klf4':
                d1=np.divide(data[:,stn_dic[stn1]],np.min(data[:,stn_dic[stn1]]))   
            if stn2=='DAPI' or stn2=='Sox2':
                d2=np.divide(data[:,stn_dic[stn2]],np.mean(data[:,stn_dic[stn2]]))
            elif stn2=='Sox9' or stn2=='Klf4':
                d2=np.divide(data[:,stn_dic[stn2]],np.min(data[:,stn_dic[stn2]]))
            # Find cells on the boundary of the image  
            b_c=f_bd_pts((data[:,2],data[:,3]),d)
            # Sort out cells which are not on the boundary of the image       
            l_da=[]
            l_db=[]
            k=0
            for j in range(0,len(d1)):
                if any(b_c==j):
                    k=k+1
                else:
                    l_da.append(d1[j])
                    l_db.append(d2[j])
            l_d1.append(l_da)
            l_d2.append(l_db)
        data_s1[i]=np.concatenate(l_d1)
        data_s2[i]=np.concatenate(l_d2)
    # Calculate p percentile of Stn1 at t_p_stn1 and q percentile of Stn2 at t_p_stn2
    p=np.percentile(data_s1[tpts_dic[t_p_stn1]],p_stn1) 
    q=np.percentile(data_s2[tpts_dic[t_p_stn2]],p_stn2) 
    return(p,q,data_s1,data_s2) 
 
def percentile_analysis_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L):
    """
    Derive p & q percentile values for stn1 & stn2 from arbitrary selected percentile thresholds p_stn1 & p_stn2.
    
    Sort associated normalised intensities for stn1 & stn2.
    
    Inputs
    ----------
    b_rs: list of biological replicate | t_rs: list of technical replicate | t_pts: list of time points  
    stn1: staining 1 | stn2: staining 2 
    p_stn1: percentile threshold for staining 1 |  p_stn2: percentile threshold for staining 2
    
    Ouputs
    ----------
    p,q: computed percentile thresholds for stn1 & stn2, float.
    data_s1, data_s2: normalised intensities for stn1 & stn2, np.array.
    """      
    # Import raw intensity data, normalise them according to their type and concatenate together the different technical replicate along a single column
    data_s1=np.zeros((len(t_pts)), dtype=object)
    data_s2=np.zeros((len(t_pts)), dtype=object)   
    for i in range (0, len(t_pts)):
        l_d1=[]
        l_d2=[]
        for j in range (0, len(b_rs)):
            for k in range (0, len(t_rs)):
                # Extract relevent data and normalise them by minimum (Sox9 & Klf4) and by the mean (DAPI & Sox2)
                data=np.genfromtxt(input_path+b_rs[j]+'_'+t_rs[k]+'_'+t_pts[i]+'.csv', delimiter=",", skip_header=1) # Load data file
                if stn1=='DAPI' or stn1=='Sox2':
                    d1=np.divide(data[:,stn_dic[stn1]],np.mean(data[:,stn_dic[stn1]]))     
                elif stn1=='Sox9' or stn1=='Klf4':
                    d1=np.divide(data[:,stn_dic[stn1]],np.min(data[:,stn_dic[stn1]]))   
                if stn2=='DAPI' or stn2=='Sox2':
                    d2=np.divide(data[:,stn_dic[stn2]],np.mean(data[:,stn_dic[stn2]]))
                elif stn2=='Sox9' or stn2=='Klf4':
                    d2=np.divide(data[:,stn_dic[stn2]],np.min(data[:,stn_dic[stn2]]))
                # Find cells on the boundary of the image  
                b_c=f_bd_pts((data[:,2],data[:,3]),d)
                # Sort out cells which are not on the boundary of the image       
                l_da=[]
                l_db=[]
                k_i=0
                for l in range(0,len(d1)):
                    if any(b_c==l):
                        k_i=k_i+1
                    else:
                        l_da.append(d1[l])
                        l_db.append(d2[l])
                l_d1.append(l_da)
                l_d2.append(l_db)
        data_s1[i]=np.concatenate(l_d1)
        data_s2[i]=np.concatenate(l_d2)
    # Calculate p percentile of Stn1 at t_p_stn1 and q percentile of Stn2 at t_p_stn2
    p=np.percentile(data_s1[tpts_dic[t_p_stn1]],p_stn1) 
    q=np.percentile(data_s2[tpts_dic[t_p_stn2]],p_stn2) 
    return(p,q,data_s1,data_s2) 
          
def percentile_sort_data(b_r,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L):
    """
    Sort normalised intensity data for 2 different stainings according to 2 percentile thresholds fixed at given time points. 
    
    Inputs
    ----------
    b_r: biological replicate | t_rs: list of technical replicate | t_pts: list of time points  
    stn1: staining 1 | stn2: staining 2 
    p_stn1: percentile threshold for staining 1 |  p_stn2: percentile threshold for staining 2
    
    Ouputs
    ----------
    p,q: computed percentile thresholds for stn1 & stn2, float
    Normalised intensities of cells for stn1 & stn2 resp. such as:
    data_p_stn1 / data_p_stn2: cells above p-percentile threshold for stn1 & below q-percentile threshold stn2, np.array 
    data_q_stn1 / data_q_stn2: cells below p-percentile threshold for stn1 & above q-percentile threshold stn2, np.array 
    data_pq_stn1 / data_pq_stn2: cells above p-percentile threshold for stn1 & above q-percentile threshold stn2, np.array
    data_n_stn1 / data_n_stn2: cells below p-percentile threshold for stn1 & below q-percentile threshold stn2, np.array
    """ 
    # Import percentile threshold
    (p,q,data_s1,data_s2)=percentile_analysis(b_r,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)
    # Sort the data according this p & q percentiles thresholds 
    data_p_stn1=np.zeros((len(t_pts)), dtype=object)
    data_p_stn2=np.zeros((len(t_pts)), dtype=object)
    data_q_stn1=np.zeros((len(t_pts)), dtype=object)
    data_q_stn2=np.zeros((len(t_pts)), dtype=object)
    data_pq_stn1=np.zeros((len(t_pts)), dtype=object)
    data_pq_stn2=np.zeros((len(t_pts)), dtype=object)
    data_n_stn1=np.zeros((len(t_pts)), dtype=object)
    data_n_stn2=np.zeros((len(t_pts)), dtype=object)
    for i in range (0,len(t_pts)):
        l_p_stn1=[]
        l_p_stn2=[]
        l_q_stn1=[]
        l_q_stn2=[]
        l_pq_stn1=[]
        l_pq_stn2=[]
        l_n_stn1=[]
        l_n_stn2=[]
        for j in range (0, len(data_s1[i])):
            if data_s1[i][j]>p and data_s2[i][j]<q:
                l_p_stn1.append(data_s1[i][j])
                l_p_stn2.append(data_s2[i][j])
            if data_s1[i][j]<p and data_s2[i][j]>q:
                l_q_stn1.append(data_s1[i][j])
                l_q_stn2.append(data_s2[i][j])
            if data_s1[i][j]>p and data_s2[i][j]>q:
                l_pq_stn1.append(data_s1[i][j])
                l_pq_stn2.append(data_s2[i][j])
            else:
                l_n_stn1.append(data_s1[i][j])
                l_n_stn2.append(data_s2[i][j])              
        data_p_stn1[i]=l_p_stn1
        data_p_stn2[i]=l_p_stn2
        data_q_stn1[i]=l_q_stn1
        data_q_stn2[i]=l_q_stn2
        data_pq_stn1[i]=l_pq_stn1
        data_pq_stn2[i]=l_pq_stn2
        data_n_stn1[i]=l_n_stn1
        data_n_stn2[i]=l_n_stn2              
    return(p,q,data_p_stn1,data_p_stn2,data_q_stn1,data_q_stn2,data_pq_stn1,data_pq_stn2,data_n_stn1,data_n_stn2)    

def percentile_sort_data_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L):
    """
    Sort intensity data for 2 different stainings according to 2 percentile thresholds fixed at given time points 
    
    Inputs
    ----------
    b_r: biological replicates | t_rs: list of technical replicates | t_pts: list of time points  
    stn1: staining 1 | stn2: staining 2 
    p_stn1: percentile threshold for staining 1 |  p_stn2: percentile threshold for staining 2
    
    Ouputs
    ----------
    p,q: computed percentile thresholds for stn1 & stn2, float
    Normalised intensities of cells for stn1 & stn2 resp. such as:
    data_p_stn1 / data_p_stn2: cells above p-percentile threshold for stn1 & below q-percentile threshold stn2, np.array 
    data_q_stn1 / data_q_stn2: cells below p-percentile threshold for stn1 & above q-percentile threshold stn2, np.array 
    data_pq_stn1 / data_pq_stn2: cells above p-percentile threshold for stn1 & above q-percentile threshold stn2, np.array
    data_n_stn1 / data_n_stn2: cells below p-percentile threshold for stn1 & below q-percentile threshold stn2, np.array
    """ 
    # Import percentile threshold
    (p,q,data_s1,data_s2)=percentile_analysis_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)
    # Sort the data according this p & q percentiles thresholds 
    data_p_stn1=np.zeros((len(t_pts)), dtype=object)
    data_p_stn2=np.zeros((len(t_pts)), dtype=object)
    data_q_stn1=np.zeros((len(t_pts)), dtype=object)
    data_q_stn2=np.zeros((len(t_pts)), dtype=object)
    data_pq_stn1=np.zeros((len(t_pts)), dtype=object)
    data_pq_stn2=np.zeros((len(t_pts)), dtype=object)
    data_n_stn1=np.zeros((len(t_pts)), dtype=object)
    data_n_stn2=np.zeros((len(t_pts)), dtype=object)
    for i in range (0,len(t_pts)):
        l_p_stn1=[]
        l_p_stn2=[]
        l_q_stn1=[]
        l_q_stn2=[]
        l_pq_stn1=[]
        l_pq_stn2=[]
        l_n_stn1=[]
        l_n_stn2=[]
        for j in range (0, len(data_s1[i])):
            if data_s1[i][j]>p and data_s2[i][j]<q:
                l_p_stn1.append(data_s1[i][j])
                l_p_stn2.append(data_s2[i][j])
            if data_s1[i][j]<p and data_s2[i][j]>q:
                l_q_stn1.append(data_s1[i][j])
                l_q_stn2.append(data_s2[i][j])
            if data_s1[i][j]>p and data_s2[i][j]>q:
                l_pq_stn1.append(data_s1[i][j])
                l_pq_stn2.append(data_s2[i][j])
            else:
                l_n_stn1.append(data_s1[i][j])
                l_n_stn2.append(data_s2[i][j])              
        data_p_stn1[i]=l_p_stn1
        data_p_stn2[i]=l_p_stn2
        data_q_stn1[i]=l_q_stn1
        data_q_stn2[i]=l_q_stn2
        data_pq_stn1[i]=l_pq_stn1
        data_pq_stn2[i]=l_pq_stn2
        data_n_stn1[i]=l_n_stn1
        data_n_stn2[i]=l_n_stn2              
    return(p,q,data_p_stn1,data_p_stn2,data_q_stn1,data_q_stn2,data_pq_stn1,data_pq_stn2,data_n_stn1,data_n_stn2) 
    
def percentile_discret_maps(b_r,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L):
    """
    Discretised intensities for 2 different stainings according to 2 percentile thresholds fixed at given time points.
    
    Inputs
    ----------
    b_r: biological replicate | t_rs: list of technical replicate | t_pts: list of time points  
    stn1: staining 1 | stn2: staining 2 
    p_stn1: percentile threshold for staining 1 |  p_stn2: percentile threshold for staining 2
    L: Image size
    
    Ouputs
    ----------
    p,q: computed percentile thresholds for stn1 & stn2, float.
    data_s: discretised intensity value for each cell according to the p & q thresholds,int.
    """
    # Import percentile threshold 
    (p,q,data_s1,data_s2)=percentile_analysis(b_r,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)
    # Arays for each technical replicate and time-point whith thresholded intensity data 
    data_s=np.zeros((len(t_pts),len(t_rs)), dtype=object)
    for i in range (0, len(t_pts)):
        for j in range (0, len(t_rs)):
            # Extract relevent data and normalise them by minimum for Sox9 & Klf4 and by the mean for DAPI & Sox2
            data=np.genfromtxt(input_path+b_r+'_'+t_rs[j]+'_'+t_pts[i]+'.csv', delimiter=",", skip_header=1) # Load data file         
            if stn1=='DAPI' or stn1=='Sox2':
                d1=np.divide(data[:,stn_dic[stn1]],np.mean(data[:,stn_dic[stn1]]))     
            elif stn1=='Sox9' or stn1=='Klf4':
                d1=np.divide(data[:,stn_dic[stn1]],np.min(data[:,stn_dic[stn1]]))   
            if stn2=='DAPI' or stn2=='Sox2':
                d2=np.divide(data[:,stn_dic[stn2]],np.mean(data[:,stn_dic[stn2]]))
            elif stn2=='Sox9' or stn2=='Klf4':
                d2=np.divide(data[:,stn_dic[stn2]],np.min(data[:,stn_dic[stn2]]))
            # Sort intensity data into discretized maps according to calculated p and q percentiles thresholds 
            d_b=np.zeros((len(d1)), dtype=float) 
            for k in range (0,len(d1)):
                if d1[k]>p and d2[k]<q:    # p positve & q negative cells
                    d_b[k]=3.0 
                elif d1[k]<p and d2[k]>q: # p negative & q positive cells
                    d_b[k]=2.0 
                elif d1[k]>p and d2[k]>q: # p positve & q positive cells
                    d_b[k]=1.0
                else:                    # p negative & q negative cells
                    d_b[k]=0.0                   
            data_s[i,j]=d_b                           
    return(p,q,data_s)
    
def percentile_discret_maps_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L):
    """
    Discretised intensities for 2 different stainings according to 2 percentile thresholds fixed at given time points.
    
    Inputs
    ----------
    b_rs: biological replicates | t_rs: list of technical replicates | t_pts: list of time points  
    stn1: staining 1 | stn2: staining 2 
    p_stn1: percentile threshold for staining 1 |  p_stn2: percentile threshold for staining 2
    L: Image size
    
    Ouputs
    ----------
    p,q: computed percentile thresholds for stn1 & stn2, float.
    data_s: discretised intensity value  for each cell according to the p & q thresholds, int.
    """
    # Import percentile threshold 
    (p,q,data_s1,data_s2)=percentile_analysis_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)
    # Arays for each technical replicate and time-point whith thresholded intensity data 
    data_s=np.zeros((len(t_pts),len(b_rs),len(t_rs)), dtype=object)
    for i in range (0, len(t_pts)):
         for j in range (0, len(b_rs)):
             for k in range (0, len(t_rs)):
                 # Extract relevent data and normalise them by minimum for Sox9 & Klf4 and by the mean for DAPI & Sox2
                 data=np.genfromtxt(input_path+b_rs[j]+'_'+t_rs[k]+'_'+t_pts[i]+'.csv', delimiter=",", skip_header=1) # Load data file         
                 if stn1=='DAPI' or stn1=='Sox2':
                     d1=np.divide(data[:,stn_dic[stn1]],np.mean(data[:,stn_dic[stn1]]))     
                 elif stn1=='Sox9' or stn1=='Klf4':
                     d1=np.divide(data[:,stn_dic[stn1]],np.min(data[:,stn_dic[stn1]]))   
                 if stn2=='DAPI' or stn2=='Sox2':
                     d2=np.divide(data[:,stn_dic[stn2]],np.mean(data[:,stn_dic[stn2]]))
                 elif stn2=='Sox9' or stn2=='Klf4':
                     d2=np.divide(data[:,stn_dic[stn2]],np.min(data[:,stn_dic[stn2]]))
                # Sort intensity data into discretized maps according to calculated p and q percentiles thresholds 
                 d_b=np.zeros((len(d1)), dtype=float) 
                 for l in range (0,len(d1)):
                     if d1[l]>p and d2[l]<q:
                         d_b[l]=3.0
                     elif d1[l]<p and d2[l]>q:
                         d_b[l]=2.0 
                     elif d1[l]>p and d2[l]>q:
                         d_b[l]=1.0
                     else:
                        d_b[l]=0.0                   
                 data_s[i,j,k]=d_b                           
    return(p,q,data_s)

def f_bd_pts(c_xy, d):    
    """
    Find indices of cell centroids within an arbitrary distance d of domain boundaries. 

    Inputs
    ----------
    (x,y) : (x,y) coordinates of input points, np.array.
    d: arbitrary distance to domain boundaries, float.

    Outputs
    -------
    index_bd_pts=indices of points within an arbitrary distance d of domain boundaries, np.array.
    """
    x=c_xy[0]
    y=c_xy[1]
    i_bd_pts=[]
    k=0
    for i in range(0, len(x)):
        if (d < x[i] < L-d) and (d < y[i] < L-d):
            k+1
        else:
            i_bd_pts.append(i)
    index_bd_pts=np.array(i_bd_pts)
    return(index_bd_pts)
   
def f_bd_pts_coord(c_xy, d):    
    """
    Find indices and coordinates of cells centroids within an arbitrary distance d of domain boundaries. 

    Inputs
    ----------
    (x,y) : (x,y) coordinates of input points, np.array.
    d: arbitrary distance to domain boundaries, float.

    Outputs
    -------
    index_bd_pts=indices of points within an arbitrary distance d of domain boundaries, np.array.
    (x_b,y_b)=coordinates of points within an arbitrary distance d of domain boundaries, np.array.
    index_c_pts=indices of points outside an arbitrary distance d of domain boundaries, np.array.
    (x_c,y_c)=coordinates of points within an arbitrary distance d of domain boundaries, np.array.
    
    """
    x=c_xy[0]
    y=c_xy[1]
    i_bd_pts=[]
    x_b=[]
    y_b=[]
    i_c_pts=[]
    x_c=[]
    y_c=[]
    for i in range(0, len(x)):
        if (d < x[i] < L-d) and (d < y[i] < L-d):
            i_c_pts.append(i)
            x_c.append(x[i])
            y_c.append(y[i])   
        else:
            i_bd_pts.append(i)
            x_b.append(x[i])
            y_b.append(y[i])            
    index_bd_pts=np.array(i_bd_pts)
    x_b=np.array(x_b)
    y_b=np.array(y_b)
    index_c_pts=np.array(i_c_pts)
    x_c=np.array(x_c)
    y_c=np.array(y_c)
    return(index_bd_pts,(x_b,y_b),index_c_pts,(x_c,y_c))

def conv_angle(angles):    
    """
    Convert angles in the [0;180] degree range to the [-90;90] range 

    Inputs
    ----------
    (angles) : angles in degrees, np.array.

    Outputs
    -------
    angles_90=angles in degree in the [-90;90] range, np.array.
    """
    ang=[]
    for i in range(0, len(angles)):
        if angles[i] < 90.0:
            ang.append(angles[i])
        else:
            ang.append(-(180.0-angles[i]))
    angles_90=np.array(ang)
    return(angles_90)
  
def structure_factor(c_xy,N,L):    
    """
    Compute tissue structure factor S(k) for a range of wavenumber k using cells centroid positions as an input.

    Inputs
    ----------
    (x,y) : (x,y) coordinates of input points, np.array.
    N: Maximum eigenvalue of the wave number k. 
    L: Image size, float.
    
    Output
    -------
    k: Computed wavenumber values, np.array.
    S: Computed structure factor values, np.array.
    """
    x=c_xy[0]
    y=c_xy[1]
    rho=np.divide(1.0,len(x))
    n=np.arange(1,N,1) 
    k=np.divide((2.0*np.pi*n),L) # Compute all possible wavenumber k (considering periodic boundary conditions)
    S=np.zeros(len(k),dtype=float) # Create array to store structure factor data
    # Compute structure factor for all values of k
    for i in range (0, len(k)):
        l_s1=[]
        for j in range (0, len(x)):
            p=(k[i]*x[j])+(k[i]*y[j])
            l_s1.append(np.exp(1j*p))
        ls1=np.array(l_s1)
        S[i]=np.multiply(np.square(np.abs(np.sum(ls1))),rho)    
    return(rho,k,S)

def structure_factor_2D(c_xy,N,L):    
    """
    Compute 2D tissue structure factor S(k) for a range of wavenumber k using cells centroid positions as an input.

    Inputs
    ----------
    (x,y) : (x,y) coordinates of input points, np.array.
    N: Maximum eigenvalue of the wave number k. 
    L: Image size, float.
    
    Output
    -------
    k: Computed wavenumber values, np.array.
    S: Computed structure factor values, np.array.
    """
    x=c_xy[0]
    y=c_xy[1]
    rho=np.divide(1.0,len(x))
    n=np.arange(1,N,1) 
    m=np.divide((2.0*np.pi*n),L)
    k_x=np.concatenate((np.multiply(-1.0,m[::-1]),m)) # Compute all possible wavenumber k_x (considering periodic boundary conditions)
    k_y=np.concatenate((np.multiply(-1.0,m[::-1]),m)) # Compute all possible wavenumber k_y (considering periodic boundary conditions)
    S=np.zeros((len(k_x),len(k_y)),dtype=float) # Create array to store structure factor data
    # Compute structure factor for all values of k
    for i in range (0, len(k_x)):
        for j in range (0,len(k_y)):
            l_s=[]
            for l in range (0, len(x)):
                p=(k_x[i]*x[l])+(k_y[j]*y[l])
                l_s.append(np.exp(1j*p))
            l_sa=np.array(l_s)
            S[i,j]=np.multiply(np.square(np.abs(np.sum(l_sa))),rho)    
    return(rho,m,k_x,k_y,S)

def local_cell_density(c_xy,rad,n_r):    
    """
    Compute local cell density as the number of cell cells centroids within an arbitrary radius of each input cell.

    Inputs
    ----------
    (x,y) : (x,y) coordinates of input points, np.array.
    rad: radius of the disk in which to look for nearest neighbours of input points, float.
    n_rc: multiplicative factor of the characteristic neighbour distance. Used if rad=0.0, float.

    Output
    -------
    loc_cell_dens: value (cell/length unit^2) of the local cell density, np.array.
    loc_cell_dens_norm: value (cell/length unit^2) of the local cell density normalised by the mean local cell density, np.array.
    """
    x=c_xy[0]
    y=c_xy[1]
    rho=np.divide(len(x),L**2) # Compute the global cell density
    r_c=np.divide(1.0,np.sqrt(rho)) # Compute the characteristic neighbour distance
    if rad==0:
        r=np.multiply(n_r,r_c)
    else:
        r=rad 
    print(r)
    c=[]
    for i in range(0, len(x)):
       c.append([x[i],y[i]])
    kd_tree=KDTree(np.array(c)) # Compute the kd-tree of all the input points
    nghs=kd_tree.query_ball_tree(kd_tree, r) 
    N_nghs=np.array(map(len, nghs))
#    print(N_nghs)
    loc_cell_dens=np.divide(N_nghs,r**2)
    lcd_mean=np.mean(loc_cell_dens)
#    print(lcd_mean)
    loc_cell_dens_norm= np.divide(loc_cell_dens,lcd_mean)
    return(loc_cell_dens,loc_cell_dens_norm)

def delaunay(c_xy):
    """
    Perform a 2D delaunay triangulation using Scipy 'Delaunay' function

    Inputs
    ----------
    (x,y) : (x,y) coordinates of input points, np.array.

    Outputs
    -------
    tri: delaunay triangulation - Output diagram
    """
    n_xy=np.vstack((c_xy)).T
    tri=Delaunay(n_xy)
    return(tri)

def f_nghs_del(index,tri):
    """
    Compute number and indices of input point nearest neighbours from a 2D delaunay triangulation.

    Inputs
    ----------
    index : index of the input point, integer.
    tri: delaunay triangulation - Input diagram
    
    Outputs
    -------
    N_nghs: number of nearest neighbours of the input point- integer.
    I_nghs: indices of nearest neighbours of the input point - tuple of integers.
    """
    I_nghs=tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][index]:tri.vertex_neighbor_vertices[0][index+1]]
    N_nghs=len(I_nghs)
    return(N_nghs, I_nghs)
    
def voronoi(c_xy):
    """
    Construct a 2D voronoi tesselation using Scipy 'Voronoi' function & eliminates infinite cells

    Inputs
    ----------
    (x,y) : (x,y) coordinates of input points, np.array.

    Outputs
    -------
    vor: voronoi  - Output diagram
    """
    n_xy=np.vstack((c_xy)).T
    vor=Voronoi(n_xy)
    # Filter regions which have a vertex outside the voronoi diagram i.e the infinite cells
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
        if region != [] and flag:
            regions.append(region)
    vor.filtered_points = n_xy
    vor.filtered_regions = regions
    return(vor)

def vor_cell_centroid(polygon):
    """
    Compute the centroid of a 2D voronoi cell using python Qhull library
    
    Inputs
    ----------
    polygon : coordinates of vertices that bound a 2d voronoi cell.

    Outputs
    -------
    centroid: (x,y) coordinates of the voronoi cell centroid, float
    """
    hull=ConvexHull(polygon)
    centroid=np.mean(polygon[hull.vertices, :], axis=0)
    return (centroid)

def vor_cell_N_vert(polygon):
    """
    Compute the number of vertices of finite size voronoi cells using python Qhull library
    
    Inputs
    ----------
    polygon : coordinates of vertices that bound a 2d voronoi cell.
    
    Ouputs
    ----------
    N_vertices: number of vertices, integer
    
    """
    hull=ConvexHull(polygon)
    N_vertices=len(hull.vertices)
    return(N_vertices)
    
def vor_cell_perimeter(polygon):
    """
    Compute the perimeter of a 2D voronoi cell using python Qhull library.
    
    Inputs
    ----------
    polygon : coordinates of vertices that bound a 2d voronoi cell.
    
    Ouputs
    ----------
    perimeter: perimeter, float
    
    """
    hull=ConvexHull(polygon)
    perimeter=hull.area
    return(perimeter)

def vor_cell_surface(polygon):
    """
    Compute surface area of a 2D voronoi cells using python Qhull library.
    
    Inputs
    ----------
    polygon : coordinates of vertices that bound a 2d voronoi cell.
    
    Ouputs
    ----------
    area: surface area, float
    
    """
    hull=ConvexHull(polygon)
    area=hull.volume
    return(area)    
    
def voronoi_finite_cells(vor,radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite regions.

    Inputs
    ----------
    vor : Input diagram - Voronoi
    radius : Distance to 'points at infinity' - float, optional 

    Outputs
    -------
    regions : Indices of vertices in each revised Voronoi regions - list of tuples
    vertices : Coordinates for revised Voronoi vertices. Same as coordinates of input vertices, with 'points at infinity' appended to the end - list of tuples.

    """
    # Test if the input voronoi diagram is 2D
    if vor.points.shape[1] != 2: 
        raise ValueError("Requires 2D input")
    new_regions = [] # Empty list to store new voronoi cells 
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2
    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all([v >= 0 for v in vertices]):
            # For finite region
            new_regions.append(vertices)
            continue
        # Reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # Sort regions counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())
    new_vertices=np.asarray(new_vertices)
    return (new_regions, new_vertices)

def voronoi_neighbours(b_r,t_r,t_pt,d,L):      
    # Load data file     
    data=np.genfromtxt(input_path+b_r+'_'+t_r+'_'+t_pt+'.csv', delimiter=",", skip_header=1)
    # Find cells on the boundary of the image  
    b_c=f_bd_pts((data[:,2],data[:,3]), d)
    # Voronoi diagram for all nulcei centroids
    vor=voronoi((data[:,2],data[:,3])) 
    regions,vertices=voronoi_finite_cells(vor) # Finite cells voronoi diagram 
    # Compute for each voronoi cell its polygon class & sort it accordingly 
    P_b=[] # Border cell / excluded from analysis
    P_3=[] # Triangle
    P_4=[] # Square
    P_5=[] # Pentagon
    P_6=[] # Hexagon
    P_7=[] # Heptagon
    P_8=[] # Octogon
    P_9=[] # Nonagon
    k=-1
    for region in regions:
        k=k+1
        polygon=vertices[region]
        if any(b_c==k):
            P_b.append(polygon)
        else:
            if vor_cell_N_vert(polygon)==3:
                P_3.append(polygon)
            elif vor_cell_N_vert(polygon)==4:
                P_4.append(polygon)
            elif vor_cell_N_vert(polygon)==5:
                P_5.append(polygon)
            elif vor_cell_N_vert(polygon)==6:
                P_6.append(polygon)
            elif vor_cell_N_vert(polygon)==7:
                P_7.append(polygon)
            elif vor_cell_N_vert(polygon)==8:
                P_8.append(polygon)
            elif vor_cell_N_vert(polygon)==9:
                P_9.append(polygon)
    # Compute stats on the number of neighbours
    n_ngh=np.array([len(P_3),len(P_4),len(P_5),len(P_6),len(P_7),len(P_8),len(P_9)],dtype=float) # Number of cells of a given polygon class
    f_ngh=np.multiply(np.divide(n_ngh,np.sum(n_ngh)),100.0) # Frequency (in %) of cells of a given polygon class
    return(n_ngh,f_ngh)

#################################################### Plots ######################################################
    
#################### Figure 2 #################### 
    
# Fig.2 - Histogram of stainings intensity 
def Fig2_intensity_hist(b_r,t_rs,t_pt,stn,L):
    # Create an array to store the final data
    data_c=np.zeros(1, dtype=object)
    # Import raw intensity data, normalise them according to their type and concatenate together the different technical replicate along a single column
    l_d=[]
    for j in range (0, len(t_rs)):
        data=np.genfromtxt(input_path+b_r+'_'+t_rs[j]+'_'+t_pt+'.csv', delimiter=",", skip_header=1) # Load data file 
        # Extract relevent data and normalise them by minimum for Sox9 & Klf4 and by the mean for DAPI & Sox2
        if stn=='DAPI' or stn=='Sox2':
            d=np.divide(data[:,stn_dic[stn]],np.mean(data[:,stn_dic[stn]]))
        else:
            d=np.divide(data[:,stn_dic[stn]],np.min(data[:,stn_dic[stn]]))
        l_d.append(d)
    data_c=np.concatenate(l_d)
    w_data_c=(np.ones_like(data_c)/np.float(len(data_c)))*100.0
    # Plot histogram 
    fig1,axes = plt.subplots(1,1,figsize=(4.0,4.0))
#    axes.hist(data_c, bins=16, weights=weights, color='lightgrey', edgecolor='black', alpha=0.5)
#    sns.distplot(data_c, kde=True, hist=False, kde_kws={"shade":True,"clip":(0.0,100.0),"bw":4.0}, color='lightgrey', ax=axes)
    p_h=sns.distplot(data_c, bins=16, kde=False, hist=True, norm_hist=False, hist_kws={'weights':w_data_c,"range":(0.0,100.0), "edgecolor":'black', "linewidth":0.5} , color='lightgrey', ax=axes)
    bar=[h.get_height() for h in p_h.patches] # Values contained in the histogram bars
    ax_b = axes.twinx()
    p_k=sns.distplot(data_c, kde=True, hist=False, kde_kws={"shade":False,"clip":(0.0,100.0),"bw":4.0}, color='lightgrey', ax=ax_b)
    x_k,y_k=p_k.get_lines()[0].get_data() # x,y values of the KDE curve
    ax_b.yaxis.set_ticks([])
    axes.annotate((t_pt), xy=(0.8, 0.88), xycoords="axes fraction", bbox=dict(facecolor='none', edgecolor='black'), fontsize=18) 
    axes.set_xlim([-10.0,110.0])
    y_max=100.0
    scl=(y_max/np.max(bar)) # Scaling factor on y axis between the hist and kde plot
    axes.set_ylim([0.0,y_max])
    ax_b.set_ylim([0.0,np.max(y_k)*scl])
    axes.set_xlabel(r'\,'+stn+'\, intensity (a.u)', fontsize=22)
    axes.set_ylabel(r'Cell counts ($\%$)', fontsize=22)
    axes.tick_params(axis='both', which='major', labelsize=20)
    fig1.tight_layout()
    # Save Fig
    fig1.savefig(output_path+'Fig2_Int_Histogram_'+b_r+'_'+stn+'_'+t_pt+'.pdf', dpi=400)
    fig1.savefig(output_path+'Fig2_Int_Histogram_'+b_r+'_'+stn+'_'+t_pt+'.png', dpi=400)
    # Save data as csv file with appropriate header
    b = list(data_c)
    df = pd.DataFrame(b)
    df.to_csv(output_path +'Fig2_intensity_hist.csv', header=t_pt, sep = ",")
    return()
    
# Fig.2 - Percentile histogram of staining stn1 highlighting values above an arbitrary p percentile
def Fig2_percentile_intensity_hist(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L):
    # Import discretised intensity maps according p & q percentile thresholds      
    (p,q,data_s)=percentile_discret_maps_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)  
    # Create an array to store the final data
    data_m1=np.zeros((len(t_pts)), dtype=float)
    data_e1=np.zeros((len(t_pts)), dtype=float)
    for j in range (0,len(t_pts)): 
        l_X_p=[]
        for l in range(0,len(b_rs)):
            for i in range (0,len(t_rs)):
                data=np.genfromtxt(input_path+b_rs[l]+'_'+t_rs[i]+'_'+t_pts[j]+'.csv', delimiter=",", skip_header=1) # Load data file 
                # Find cells on the boundary of the image  
                b_c=f_bd_pts((data[:,2],data[:,3]),d)
                # Assign to each voronoi cell a colour on the basis of p & q percentile threshold and sort them accordingly
                P_b=[] # Border cells
                H_Stn1=[] 
                H_Stn2=[]
                H_both=[]
                L_both=[]
                for k in range(0,len(data[:,2])):
                    if any(b_c==k):
                        P_b.append(k)
                    else:
                        if data_s[j,l,i][k]==1.0:
                            H_both.append(k)
                        elif data_s[j,l,i][k]==2.0:
                            H_Stn2.append(k)
                        elif data_s[j,l,i][k]==3.0:
                            H_Stn1.append(k)
                        else:
                            L_both.append(k)   
                # Total number of cell - number of border cells
                N_tot=np.float(np.subtract(len(data[:,2]),len(P_b)))
                # Number of cells above the p percentile for stn1
                N_p=np.float(np.add(len(H_Stn1),len(H_both)))
                # Fraction of cells above the p percentile for stn1
                X_p=np.multiply(np.divide(N_p,N_tot),100.0)
                l_X_p.append(X_p)
        data_m1[j]=np.round(np.mean(l_X_p),2) #Mean
        data_e1[j]=np.round(np.divide(np.std(l_X_p),np.sqrt(len(b_rs)*len(t_rs))),2) # SEM
    print(data_m1)
    print(data_e1) 
    # Import intensity data sorted accorded p & q percentile thresholds
    (p,q,data_p_stn1,data_p_stn2,data_q_stn1,data_q_stn2,data_pq_stn1,data_pq_stn2,data_n_stn1,data_n_stn2)=percentile_sort_data_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)
    # Plot the data   
    for j in range (0, len(t_pts)):
        stn1_all=np.concatenate((data_p_stn1[j],data_q_stn1[j],data_pq_stn1[j], data_n_stn1[j]),axis=None) 
        fig1,axes = plt.subplots(1,1,figsize=(4.0,4.0))
        #Compute probability weights for percentage histograms
        w_data_c=(np.ones_like(stn1_all)/np.float(len(stn1_all)))*100.0
        axes.plot
        # Define the intensity histogram for all cells (Klf4: bins=20, kernel bw= 4.5 / Sox9: bins=12, kernel bw=1.0)
        if stn1 == 'Sox9':
            p_h=sns.distplot(stn1_all, bins=12, kde=False, hist=True, norm_hist=False, hist_kws={'weights':w_data_c,"range":(0.0,25.0), "edgecolor":'black', "linewidth":0.5, "alpha":0.0} , color='grey', ax=axes)
        else:
            p_h=sns.distplot(stn1_all, bins=20, kde=False, hist=True, norm_hist=False, hist_kws={'weights':w_data_c,"range":(0.0,25.0), "edgecolor":'black', "linewidth":0.5, "alpha":0.0} , color='grey', ax=axes)
        bar=[h.get_height() for h in p_h.patches] # Values contained in the histogram bars 
        ax_b = axes.twinx() # Create a twin y axis for ploting
        # Define & plot the smoothed intensity  histogram (KDE) for all cells
        p_k=sns.distplot(stn1_all, kde=True, hist=False, kde_kws={"shade":False,"clip":(0.0,25.0), "color":'black', "linewidth":1.0,"bw":1.5}, ax=ax_b)
        x_k,y_k=p_k.get_lines()[0].get_data() # Extract x,y values of the KDE curve
        ax_b.yaxis.set_ticks([]) 
        if stn1 == 'Sox9':
            y_max=80.0 # Sox9: 80 
        else:
            y_max=65.0 # Klf4: 65.0
        scl=(y_max/np.max(bar)) # Scaling factor on y axis between the hist and kde plot
        axes.set_ylim([0.0,y_max])
        ax_b.set_ylim([0.0,np.max(y_k)*scl])
        axes.set_xlim([-2.0,25.0])
        axes.plot([p,p],[0.0,110.0], linestyle='--', linewidth=1.5, color='black', alpha=0.75) # Plot a line to show the p percentile threshold value
        mask1 = x_k > p # Create a mask for intensity values of the histogram above the p  percentile threshold value
        x_k, y_k = x_k[mask1], y_k[mask1]
        p_k.fill_between(x_k, y1=y_k, alpha=0.5, facecolor='grey') # Highlight in a different colour values of the histogram above the p  percentile threshold value
        axes.annotate((data_m1[j].astype('|S4') + r'$\pm$' + data_e1[j].astype('|S4') + r'$\%$'), xy=(0.6,0.87), xycoords="axes fraction", fontsize=16, color='grey', alpha=0.75) 
        if stn1 == 'Sox9':
            axes.set_xticks([0.0,10.0,20.0]) #Sox9
            axes.set_xticklabels([0,10,20])
        else:
            axes.set_xticks([0.0,25.0,50.0,75.0,100.0]) #Klf4
            axes.set_xticklabels([0,25,50,75,100])
        axes.set_xlabel(r'\,'+stn1+'\, intensity (a.u)', fontsize=22)
        axes.set_ylabel(r'Cell counts ($\%$) ', fontsize=22)
        axes.tick_params(axis='both', which='major', labelsize=20)
        fig1.tight_layout()
#        # Save Fig
        fig1.savefig(output_path+'Fig2_Norm_Int_hist_'+stn1+'_'+repr(p_stn1)+'_'+t_pts[j]+'.pdf', dpi=400)
        fig1.savefig(output_path+'Fig2_Norm_Int_hist_'+stn1+'_'+repr(p_stn1)+'_'+t_pts[j]+'.png', dpi=400)
    return()

# Fig.2 - Discretised voronoi tissue maps for values above q percentile of staining 1
def Fig2_percentile_intensity_map_single(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L):
    # Import intensity data sorted accorded p & q percentile thresholds
    (p,q,data_p_stn1,data_p_stn2,data_q_stn1,data_q_stn2,data_pq_stn1,data_pq_stn2,data_n_stn1,data_n_stn2)=percentile_sort_data_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)
    # Import discretised intensity maps according p & q percentile thresholds      
    (p,q,data_s)=percentile_discret_maps_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)  
    print(p,q)
    # Plot the data 
    for j in range (0,len(t_pts)): 
        for l in range(0,len(b_rs)):
            for i in range (0,len(t_rs)):
                fig1,axes = plt.subplots(1,1,figsize=(4.0,4.0))
                data=np.genfromtxt(input_path+b_rs[l]+'_'+t_rs[i]+'_'+t_pts[j]+'.csv', delimiter=",", skip_header=1) # Load data file 
                # Find cells within and outside a ban of width d around the image
                b_b,(x_b,y_b),b_c,(x_c,y_c)=f_bd_pts_coord((data[:,2],data[:,3]),d)
                # Tissue voronoi diagram()
                vor=voronoi((data[:,2],data[:,3])) # Voronoi diagram 
                regions,vertices=voronoi_finite_cells(vor) # Finite cells voronoi diagram             
                # Assign to each voronoi cell a colour on the basis of p & q percentile threshold and sort them accordingly
                P_b=[] # Border cells
                H_Klf4=[]
                H_Sox9=[]
                H_both=[]
                L_both=[]
                k=-1
                for region in regions:
                    k=k+1
                    polygon=vertices[region]
                    if any(b_b==k):
                        P_b.append(polygon)
                    else:
                        if data_s[j,l,i][k]==1.0:
                            H_both.append(polygon)
                        elif data_s[j,l,i][k]==2.0:
                            H_Sox9.append(polygon)
                        elif data_s[j,l,i][k]==3.0:
                            H_Klf4.append(polygon)
                        else:
                            L_both.append(polygon)
                p_P_b=PolyCollection(P_b,facecolor='white', edgecolor='black', linewidths=0.1, alpha=1.0, closed=True)
                p_H_Klf4=PolyCollection(H_Klf4,facecolor='red',edgecolor='black',linewidths=0.75,alpha=0.75,closed=True)
                p_H_Sox9=PolyCollection(H_Sox9,facecolor='white',edgecolor='black',linewidths=0.75,alpha=0.75,closed=True)
                p_H_both=PolyCollection(H_both,facecolor='red',edgecolor='black',linewidths=0.75,alpha=0.75,closed=True)
                p_L_both=PolyCollection(L_both, facecolor='white', edgecolor='black',linewidths=0.75,alpha=0.75, closed=True)
                # Plot the colored voronoi cells
                axes.add_collection(p_H_Klf4)
                axes.add_collection(p_H_Sox9)
                axes.add_collection(p_H_both)
                axes.add_collection(p_L_both)                    
                axes.add_collection(p_P_b)
                axes.scatter(x_c,y_c, marker='.', color='black', s=1.5, alpha=1.0) # Scatter plot of cells centroids positions  
                axes.set_ylim([0.0,L])
                axes.set_xlim([0.0,L])
                axes.set_xticks([])
                axes.set_yticks([])            
                fig1.tight_layout()
                # Save Fig
                fig1.savefig(output_path+'Fig2_percentile_intensity_map_above_'+stn1+'_'+repr(p_stn1)+'th_percentile_'+b_rs[l]+'_'+t_rs[i]+'_'+t_pts[j]+'.pdf', dpi=400)
                fig1.savefig(output_path+'Fig2_percentile_intensity_map_above_'+stn1+'_'+repr(p_stn1)+'th_percentile_'+b_rs[l]+'_'+t_rs[i]+'_'+t_pts[j]+'.png', dpi=400)
    return()
    
# Fig.2 - Discretised voronoi tissue maps for values above p percentile of staining 1 and q percentile of staining 2
def Fig2_percentile_intensity_map_double(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L):
    # Import intensity data sorted accorded p & q percentile thresholds
    (p,q,data_p_stn1,data_p_stn2,data_q_stn1,data_q_stn2,data_pq_stn1,data_pq_stn2,data_n_stn1,data_n_stn2)=percentile_sort_data_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)
    # Import discretised intensity maps according p & q percentile thresholds      
    (p,q,data_s)=percentile_discret_maps_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)  
    print(p,q)
    # Plot the data 
    for j in range (0,len(t_pts)): 
        for l in range(0,len(b_rs)):
            for i in range (0,len(t_rs)):
                fig1,axes = plt.subplots(1,1,figsize=(4.0,4.0))
                data=np.genfromtxt(input_path+b_rs[l]+'_'+t_rs[i]+'_'+t_pts[j]+'.csv', delimiter=",", skip_header=1) # Load data file 
                # Find cells within and outside a ban of width d around the image
                b_b,(x_b,y_b),b_c,(x_c,y_c)=f_bd_pts_coord((data[:,2],data[:,3]),d)
                # Tissue voronoi diagram()
                vor=voronoi((data[:,2],data[:,3])) # Voronoi diagram 
                regions,vertices=voronoi_finite_cells(vor) # Finite cells voronoi diagram             
                # Assign to each voronoi cell a colour on the basis of p & q percentile threshold and sort them accordingly
                P_b=[] # Border cells
                H_Klf4=[]
                H_Sox9=[]
                H_both=[]
                L_both=[]
                k=-1
                for region in regions:
                    k=k+1
                    polygon=vertices[region]
                    if any(b_b==k):
                        P_b.append(polygon)
                    else:
                        if data_s[j,l,i][k]==1.0:
                            H_both.append(polygon)
                        elif data_s[j,l,i][k]==2.0:
                            H_Sox9.append(polygon)
                        elif data_s[j,l,i][k]==3.0:
                            H_Klf4.append(polygon)
                        else:
                            L_both.append(polygon)
                p_P_b=PolyCollection(P_b,facecolor='white',edgecolor='black', linewidths=0.1, alpha=1.0, closed=True)
                p_H_Klf4=PolyCollection(H_Klf4,facecolor='red',edgecolor='black',linewidths=1.0,alpha=0.75,closed=True)
                p_H_Sox9=PolyCollection(H_Sox9,facecolor='grey',edgecolor='black',linewidths=1.0,alpha=0.75,closed=True)
                p_H_both=PolyCollection(H_both,facecolor='darkorange',edgecolor='black',linewidths=1.0,alpha=0.75,closed=True)
                p_L_both=PolyCollection(L_both, facecolor='white', edgecolor='black',linewidths=1.0,alpha=0.75, closed=True)
                # Plot the colored voronoi cells
                axes.add_collection(p_H_Klf4)
                axes.add_collection(p_H_Sox9)
                axes.add_collection(p_H_both)
                axes.add_collection(p_L_both)  
                axes.add_collection(p_P_b)                  
                axes.scatter(x_c, y_c, marker='.', color='black', s=1.5, alpha=1.0) # Scatter plot of cells centroids positions  
                axes.set_ylim([0.0,L])
                axes.set_xlim([0.0,L])
                axes.set_xticks([])
                axes.set_yticks([])            
                fig1.tight_layout()
                # Save Fig
                fig1.savefig(output_path+'Fig2_percentile_intensity_map_above_'+stn1+'_'+repr(p_stn1)+'th_percentile_'+stn2+'_'+repr(p_stn2)+'th_percentile_'+b_rs[l]+'_'+t_rs[i]+'_'+t_pts[j]+'.pdf', dpi=400)
                fig1.savefig(output_path+'Fig2_percentile_intensity_map_above_'+stn1+'_'+repr(p_stn1)+'th_percentile_'+stn2+'_'+repr(p_stn2)+'th_percentile_'+b_rs[l]+'_'+t_rs[i]+'_'+t_pts[j]+'.png', dpi=400)
    return()
    
# Fig.2 - Boxplots of the fraction of cells whose staining 1 values are above a given p percentile & staining 2 values are above a given q percentile
def Fig2_intensity_percentile_pos_cells_bxplts(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L):
    # Import intensity data sorted accorded p & q percentile thresholds
    (p,q,data_p_stn1,data_p_stn2,data_q_stn1,data_q_stn2,data_pq_stn1,data_pq_stn2,data_n_stn1,data_n_stn2)=percentile_sort_data_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)
    # Import discretised intensity maps according p & q percentile thresholds      
    (p,q,data_s)=percentile_discret_maps_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)  
    # Create an array to store the final data
    data_c=np.zeros((len(t_pts)), dtype=object)
    for j in range (0,len(t_pts)): 
        l_X_p=[]
        for l in range(0,len(b_rs)):
            for i in range (0,len(t_rs)):
                data=np.genfromtxt(input_path+b_rs[l]+'_'+t_rs[i]+'_'+t_pts[j]+'.csv', delimiter=",", skip_header=1) # Load data file 
                # Find cells on the boundary of the image  
                b_c=f_bd_pts((data[:,2],data[:,3]),d)
                # Assign to each voronoi cell a colour on the basis of p & q percentile threshold and sort them accordingly
                P_b=[] # Border cells
                H_Stn1=[] 
                H_Stn2=[]
                H_both=[]
                L_both=[]
                for k in range(0,len(data[:,2])):
                    if any(b_c==k):
                        P_b.append(k)
                    else:
                        if data_s[j,l,i][k]==1.0:
                            H_both.append(k)
                        elif data_s[j,l,i][k]==2.0:
                            H_Stn2.append(k)
                        elif data_s[j,l,i][k]==3.0:
                            H_Stn1.append(k)
                        else:
                            L_both.append(k)   
                # Total number of cell - number of border cells
                N_tot=np.float(np.subtract(len(data[:,2]),len(P_b)))
                # Number of cells above the p percentile for stn1
                N_p=np.float(np.add(len(H_Stn1),len(H_both)))
                # Fraction of cells above the p percentile for stn1
                X_p=np.multiply(np.divide(N_p,N_tot),100.0)
                l_X_p.append(X_p)
        data_c[j]=l_X_p
    data_c=np.delete(data_c,0) # Remove P2 prior ploting
    # Plot the data   
    fig1,axes = plt.subplots(1,1,figsize=(4.4,4.0))
    axes.plot
    axes= sns.swarmplot(data=list(data_c.T), size=4.0, color='0.25', alpha=0.75, zorder=1)
    axes=sns.boxplot(data=list(data_c.T), color="white", width=0.5, whis=1.5, showmeans=True, meanline=True, showfliers=False, whiskerprops=dict(linewidth=1.5, color='black', alpha=0.7), capprops=dict(linewidth=1.5, color='black', alpha=0.7), boxprops=dict(linewidth=1.5, edgecolor='black', alpha=0.7), medianprops=dict(linestyle='-', linewidth=1.5, color='black',alpha=0.7), meanprops=dict(linestyle='--', linewidth=1.5, color='red', alpha=0.7), zorder=1)
    axes.set_ylim([-3.0,60.0])
    axes.set_ylabel(str(stn1) + r'\, positive cells ($\%$)', size=22)
    axes.set_xticklabels(t_pts_p, fontsize=22) # x-axis label without P2
    axes.tick_params(axis='both', which='major', labelsize=20)
    fig1.tight_layout()
    # Save Fig
    fig1.savefig(output_path+'Fig2_fraction_cells_above_'+stn1+'_'+repr(p_stn1)+'th_percentile_boxplots'+'.pdf', dpi=400)
    fig1.savefig(output_path+'Fig2_fraction_cells_above_'+stn1+'_'+repr(p_stn1)+'th_percentile_boxplots'+'.png', dpi=400)
   # Save data as csv file with appropriate header
    b = list(data_c)
    df = pd.DataFrame(b)
    dfs=df.T
    dfs.to_csv(output_path +'Fig2_fraction_cells_above_'+stn1+'_'+repr(p_stn1)+'th_percentile_boxplots_Data.csv', header=t_pts_p, sep = ",")
    return()
      
# Fig.2 - Boxplots of staining intensity 
def Fig2_intensity_bxplts(b_rs,t_rs,t_pts,stn,d,L):
    # Create an array to store the final data
    data_c=np.zeros((len(t_pts)), dtype=object)
    for i in range (0, len(t_pts)):
        l_b=[]
        l_d=[]
        for j in range (0, len(b_rs)):
            for k in range (0, len(t_rs)):
                data=np.genfromtxt(input_path+b_rs[j]+'_'+t_rs[k]+'_'+t_pts[i]+'.csv', delimiter=",", skip_header=1) # Load data file 
                # Extract relevent data and normalise them by minimum for Sox9 & Klf4 and by the mean for DAPI & Sox2
                if stn=='DAPI' or stn=='Sox2':
                    Int=np.divide(data[:,stn_dic[stn]],np.min(data[:,stn_dic[stn]]))  
                else:
                    Int=np.divide(data[:,stn_dic[stn]],np.min(data[:,stn_dic[stn]]))
                # Cells on the boundary of the image  
                b_c=f_bd_pts((data[:,2],data[:,3]),d)
                for l in range (0,len(data[:,2])):
                    if any(b_c==l):
                        l_b.append(l)
                    else:
                        l_d.append(Int[l])
        data_c[i]=l_d
    data_c=np.delete(data_c,0) # Remove P2 prior ploting if needed
    fig1,axes = plt.subplots(1,1,figsize=(4.4,4.0))
    axes.plot
    axes=sns.violinplot(data=list(data_c.T), cut=0, scale="count", inner=None, width=0.8, linewidth=0.0, color=[1.0, 0.5, 0.5], alpha=0.7)# Sox2: lightgreen  # Klf4: [1.0, 0.5, 0.5] # Sox9: lightgrey
    axes=sns.boxplot(data=list(data_c.T), width=0.15, whis=1.5, showcaps=False, showmeans=True, meanline=True, showfliers=False, boxprops=dict(linewidth=1.0, facecolor='none', edgecolor='black', alpha=0.75, zorder=1), whiskerprops=dict(linewidth=1.0, color='black', alpha=0.75), medianprops=dict(linestyle='-', linewidth=1.0, color='black',alpha=0.75), meanprops=dict(linestyle='-', linewidth=1.0, color='red', alpha=0.75))
    axes.set_ylim([-1.0,60.0]) # 25 / 45 / 110
    axes.set_ylabel(str(stn) + r'\, intensity (a.u)', size=22)
    axes.set_xticklabels(t_pts_p, fontsize=22) # x-axis label without P2
    axes.tick_params(axis='both', which='major', labelsize=20)
    fig1.tight_layout()
    # Save Fig
    fig1.savefig(output_path+'Fig2_Intensity_Boxplots_'+stn+'.pdf', dpi=400)
    fig1.savefig(output_path+'Fig2_Intensity_Boxplots_'+stn+'.png', dpi=400)
    # Save data as csv file with appropriate header
    b = list(data_c)
    df = pd.DataFrame(b)
    dfs=df.T
    dfs.to_csv(output_path +'Fig2_Intensity_Boxplots_'+stn+'_Data.csv', header=t_pts_p, sep = ",")
    return() 
    
# Fig.2 - Scatter plot of staining stn1 & stn2 highlighting values above arbitrary p & q percentile thresholds
def Fig2_intensity_scatter_plot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L):
    # Import discretised intensity maps according p & q percentile thresholds      
    (p,q,data_s)=percentile_discret_maps_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)  
    print(p,q)
    # Create an array to store the final data
    data_m1=np.zeros((len(t_pts)), dtype=float)
    data_m2=np.zeros((len(t_pts)), dtype=float)
    data_e1=np.zeros((len(t_pts)), dtype=float)
    data_e2=np.zeros((len(t_pts)), dtype=float)
    for j in range (0,len(t_pts)): 
        l_X_p=[]
        l_X_q=[]
        for l in range(0,len(b_rs)):
            for i in range (0,len(t_rs)):
                data=np.genfromtxt(input_path+b_rs[l]+'_'+t_rs[i]+'_'+t_pts[j]+'.csv', delimiter=",", skip_header=1) # Load data file 
                # Find cells on the boundary of the image  
                b_c=f_bd_pts((data[:,2],data[:,3]),d)
                # Assign to each voronoi cell a colour on the basis of p & q percentile threshold and sort them accordingly
                P_b=[] # Border cells
                H_Stn1=[] 
                H_Stn2=[]
                H_both=[]
                L_both=[]
                for k in range(0,len(data[:,2])):
                    if any(b_c==k):
                        P_b.append(k)
                    else:
                        if data_s[j,l,i][k]==1.0:
                            H_both.append(k)
                        elif data_s[j,l,i][k]==2.0:
                            H_Stn2.append(k)
                        elif data_s[j,l,i][k]==3.0:
                            H_Stn1.append(k)
                        else:
                            L_both.append(k)   
                # Total number of cell - number of border cells
                N_tot=np.float(np.subtract(len(data[:,2]),len(P_b)))
                # Number of cells above the p percentile for stn1
                N_p=np.float(np.add(len(H_Stn1),len(H_both)))
                # Fraction of cells above the p percentile for stn1
                X_p=np.multiply(np.divide(N_p,N_tot),100.0)
                # Number of cells above the q percentile for stn2
                N_q=np.float(np.add(len(H_Stn2),len(H_both)))
                # Fraction of cells above the p percentile for stn1
                X_q=np.multiply(np.divide(N_q,N_tot),100.0)
                l_X_p.append(X_p)
                l_X_q.append(X_q)
        data_m1[j]=np.round(np.mean(l_X_p),2)
        data_m2[j]=np.round(np.mean(l_X_q),2)
        data_e1[j]=np.round(np.divide(np.std(l_X_p),np.sqrt(len(b_rs)*len(t_rs))),2) # SEM
        data_e2[j]=np.round(np.divide(np.std(l_X_q),np.sqrt(len(b_rs)*len(t_rs))),2) #SEM
    print(data_m1)
    print(data_m2)
    # Import intensity data sorted accorded p & q percentile thresholds
    (p,q,data_p_stn1,data_p_stn2,data_q_stn1,data_q_stn2,data_pq_stn1,data_pq_stn2,data_n_stn1,data_n_stn2)=percentile_sort_data_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)
    # Plot the data   
    for j in range (0, len(t_pts)):
        stn1_all=np.concatenate((data_p_stn1[j],data_q_stn1[j],data_pq_stn1[j],data_n_stn1[j]),axis=None) 
        stn2_all=np.concatenate((data_q_stn2[j],data_p_stn2[j],data_pq_stn2[j],data_n_stn2[j]),axis=None) 
        fig1,axes = plt.subplots(1,1,figsize=(6.0,6.0))
        # Create grid structure for an orthoganal view plot
        gs = gridspec.GridSpec(3,3,height_ratios=[0.35,1.0,0.45], width_ratios=[1.0,0.35,0.35], wspace=0.05, hspace=0.05)
        # Scatter plot
        ax1 = plt.subplot(gs[1:3,0:2]) 
        ax1.scatter(data_n_stn1[j], data_n_stn2[j], s=1.0, color='lightgrey', alpha=0.5) 
        ax1.scatter(data_pq_stn1[j], data_pq_stn2[j], s=1.5, color='darkorange', alpha=0.75)   
        ax1.scatter(data_q_stn1[j], data_q_stn2[j], s=1.5, color='grey', alpha=0.85)  
        ax1.scatter(data_p_stn1[j], data_p_stn2[j], s=1.5, color='C3', alpha=0.75)     
        ax1.plot([p,p],[-10.0,200.0], linestyle='--', linewidth=1.0, color='black', alpha=0.75) # Plot a line to show the p percentile threshold value
        ax1.plot([-10.0,200.0], [q,q], linestyle='--', linewidth=1.0, color='black', alpha=0.75) # Plot a line to show the p percentile threshold value
        ax1.annotate((t_pts[j]), xy=(0.82,0.87), xycoords="axes fraction", bbox=dict(facecolor='none', edgecolor='black', pad=5.0), fontsize=20) 
        ax1.set_xlim([-5.0,110.0])
        ax1.set_ylim([-1.0,25.0])
        ax1.set_xticks([0.0,25.0,50.0,75.0,100.0])
        ax1.set_xticklabels([0,25,50,75,100])
        ax1.set_yticks([0.0,10.0,20.0])
        ax1.set_yticklabels([0,10,20])
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.set_xlabel(r'\,'+stn1+'\, intensity (a.u)', fontsize=22)
        ax1.set_ylabel(r'\,'+stn2+'\, intensity (a.u)', fontsize=22)   
        # Plot marginal distribution 1 i.e y-axis 
        ax1v = plt.subplot(gs[1:3,2])
        w_data_c=(np.ones_like(stn2_all)/np.float(len(stn2_all)))*100.0 #Compute probability weights for percentage histograms
        # Define (but don't plot) the intensity histogram for all cells
        p_h=sns.distplot(stn2_all, bins=12, kde=False, hist=True, norm_hist=False, hist_kws={'weights':w_data_c,"range":(0.0,100.0), "edgecolor":'black', "linewidth":0.5, "alpha":0.0} , color='grey', ax=ax1v)
        bar=[h.get_height() for h in p_h.patches] # Values contained in the histogram bars 
        ax_b = ax1v.twiny() # Create a twin y axis for ploting
        # Define & plot the smoothed intensity  histogram (KDE) for all cells
        p_k=sns.distplot(stn2_all, vertical=True, kde=True, hist=False, kde_kws={"shade":False,"clip":(0.0,100.0), "color":'black', "linewidth":1.0,"bw":1.0}, ax=ax_b)
        y_k,x_k=p_k.get_lines()[0].get_data() # Extract x,y values of the KDE curve
        x_max=100.0 # y axis of the vertical histogram is x axis of the plot
        y_max=25.0  # x axis of the vertical histogram is y axis of the plot
        scl=(x_max/np.max(bar)) 
        ax_b.set_ylim([0.0,np.max(x_k)*scl])
        ax_b.set_xlim([0.0,0.33])
        ax1v.set_ylim([-1.0,y_max])
        ax1v.set_xlim([0.0,x_max])
        ax1v.plot([0.0,x_max],[q,q], linestyle='--', linewidth=1.0, color='black', alpha=0.75) # Plot a line to show the q percentile threshold value
        mask1 = x_k > q  # Create a mask for intensity values of the histogram above the q percentile threshold value
        x_k1, y_k1 = x_k[mask1], y_k[mask1]
        x_k2=np.full(len(y_k1),q,dtype=float)
#        ax_b.plot(y_k1,x_k1, color='grey')
#        ax_b.plot(y_k1,x_k2, color='red')
        ax_b.fill_between(y_k1,y1=x_k1, y2=x_k2, alpha=0.5, facecolor='grey') # Highlight in a different colour values of the histogram above the p  percentile threshold value
        ax_b.xaxis.set_ticks([]) 
        ax1v.annotate((data_m2[j].astype('|S4') + r'$\pm$' + data_e2[j].astype('|S4') + r'$\%$'), xy=(0.72,0.92), xycoords="axes fraction",  rotation=-90, fontsize=14, color='grey', alpha=0.75) 
        ax1v.set_xticks([])
        ax1v.set_yticks([])
        ax1v.set_xticklabels([])
        ax1v.set_yticklabels([])
        # Plot marginal distribution 2 i.e x-axis 
        ax1h = plt.subplot(gs[0,0:2])  
        w_data_c=(np.ones_like(stn1_all)/np.float(len(stn1_all)))*100.0 # Compute probability weights for percentage histograms
        p_h=sns.distplot(stn1_all, bins=20, kde=False, hist=True, norm_hist=False, hist_kws={'weights':w_data_c,"range":(0.0,110.0), "edgecolor":'black', "linewidth":0.5, "alpha":0.0} , color='grey', ax=ax1h) # Define (but don't plot) the intensity histogram for all cells
        bar=[h.get_height() for h in p_h.patches] # Values contained in the histogram bars 
        ax_b = ax1h.twinx() # Create a twin x axis for ploting
        p_k=sns.distplot(stn1_all, kde=True, hist=False, kde_kws={"shade":False,"clip":(0.0,100.0), "color":'black', "linewidth":1.0,"bw":4.5}, ax=ax_b) # Define & plot the smoothed intensity  histogram (KDE) for all cells
        x_k,y_k=p_k.get_lines()[0].get_data() # Extract x,y values of the KDE curve
        y_max=70.0
        x_max=110.0
        scl=(y_max/np.max(bar)) # Scaling factor on y axis between the hist and kde plot
        ax_b.set_ylim([0.0,np.max(y_k)*scl])
        ax1h.set_ylim([0.0,y_max])
        ax1h.set_xlim([-5.0,x_max])
        ax1h.plot([p,p],[0.0,y_max], linestyle='--', linewidth=1.0, color='black', alpha=0.75) # Plot a line to show the p percentile threshold value
        mask1 = x_k > p # Create a mask for intensity values of the histogram above the p  percentile threshold value
        x_k1, y_k1 = x_k[mask1], y_k[mask1]
        y_k2=np.full(len(y_k1),0,dtype=float)
#        ax_b.plot(x_k1,y_k1, color='red', alpha=0.5)
#        ax_b.plot(x_k2, y_k1, color='red')
        ax_b.fill_between(x_k1, y1=y_k2, y2=y_k1, alpha=0.75, facecolor='C3') # Highlight in a different colour values of the histogram above the p percentile threshold value
        ax_b.yaxis.set_ticks([]) 
        ax1h.annotate((data_m1[j].astype('|S4') + r'$\pm$' + data_e1[j].astype('|S4') + r'$\%$'), xy=(0.72,0.74), xycoords="axes fraction", fontsize=14, color='red', alpha=0.75) 
        ax1h.set_xticks([])
        ax1h.set_yticks([])
        ax1h.set_xticklabels([])
        ax1h.set_yticklabels([]) 
        fig1.tight_layout()
        # Save Fig
        fig1.savefig(output_path+'Fig2_sct_plt_marg_hist_intensity_'+stn1+'_'+repr(p_stn1)+'_'+stn2+'_'+repr(p_stn2)+'_'+t_pts[j]+'.pdf', dpi=300)
        fig1.savefig(output_path+'Fig2_sct_plt_marg_hist_intensity_'+stn1+'_'+repr(p_stn1)+'_'+stn2+'_'+repr(p_stn2)+'_'+t_pts[j]+'.png', dpi=400)
    return()
    
# Fig.2 - Plots of the fraction of cells whose staining 1 values are above a given p percentile
def Fig2_intensity_percentile_pos_cells_plts(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L):
    # Import discretised intensity maps according p & q percentile thresholds      
    (p,q,data_s)=percentile_discret_maps_tot(b_rs,t_rs,t_pts,stn1,stn2,p_stn1,p_stn2,t_p_stn1,t_p_stn2,d,L)  
    print(p,q)
    # Create an array to store the final data
    data_m1=np.zeros((len(t_pts)), dtype=float)
    data_m2=np.zeros((len(t_pts)), dtype=float)
    data_e1=np.zeros((len(t_pts)), dtype=float)
    data_e2=np.zeros((len(t_pts)), dtype=float)
    for j in range (0,len(t_pts)): 
        l_X_p=[]
        l_X_q=[]
        for l in range(0,len(b_rs)):
            for i in range (0,len(t_rs)):
                data=np.genfromtxt(input_path+b_rs[l]+'_'+t_rs[i]+'_'+t_pts[j]+'.csv', delimiter=",", skip_header=1) # Load data file 
                # Find cells on the boundary of the image  
                b_c=f_bd_pts((data[:,2],data[:,3]),d)
                # Assign to each voronoi cell a colour on the basis of p & q percentile threshold and sort them accordingly
                P_b=[] # Border cells
                H_Stn1=[] 
                H_Stn2=[]
                H_both=[]
                L_both=[]
                for k in range(0,len(data[:,2])):
                    if any(b_c==k):
                        P_b.append(k)
                    else:
                        if data_s[j,l,i][k]==1.0:
                            H_both.append(k)
                        elif data_s[j,l,i][k]==2.0:
                            H_Stn2.append(k)
                        elif data_s[j,l,i][k]==3.0:
                            H_Stn1.append(k)
                        else:
                            L_both.append(k)   
                # Total number of cell - number of border cells
                N_tot=np.float(np.subtract(len(data[:,2]),len(P_b)))
                # Number of cells above the p percentile for stn1
                N_p=np.float(np.add(len(H_Stn1),len(H_both)))
                # Fraction of cells above the p percentile for stn1
                X_p=np.multiply(np.divide(N_p,N_tot),100.0)
                # Number of cells above the q percentile for stn2
                N_q=np.float(np.add(len(H_Stn2),len(H_both)))
                # Fraction of cells above the p percentile for stn1
                X_q=np.multiply(np.divide(N_q,N_tot),100.0)
                l_X_p.append(X_p)
                l_X_q.append(X_q)
        data_m1[j]=np.mean(l_X_p) # Mean
        data_m2[j]=np.mean(l_X_q) # Mean
        data_e1[j]=np.divide(np.std(l_X_p),np.sqrt(len(b_rs)*len(t_rs))) # SEM
        data_e2[j]=np.divide(np.std(l_X_q),np.sqrt(len(b_rs)*len(t_rs))) #SEM
    data_m1=np.delete(data_m1,0) # Remove P2 prior ploting if needed
    data_m2=np.delete(data_m2,0) # Remove P2 prior ploting if needed
    data_e1=np.delete(data_e1,0) # Remove P2 prior ploting if needed
    data_e2=np.delete(data_e2,0) # Remove P2 prior ploting if needed
    tpts=np.array([7.0,14.0,28.0,49.0,70.0],dtype=float)
    print(data_m1)
    print(data_m2)
    # Plot the data
    fig1,axes = plt.subplots(1,1,figsize=(4.0,4.0))
    axes.errorbar(tpts,data_m1,data_e1, marker='.', linestyle='-', linewidth=1.5, capsize=2.5, color='C3',alpha=0.75) # Klf4
    axes.errorbar(tpts,data_m2,data_e2, marker='.', linestyle='-', linewidth=1.5, capsize=2.5, color='grey',alpha=0.75) #Sox9
    axes.legend((str(stn1)+r'$^+$',str(stn2)+r'$^+$'), loc='upper right', fontsize=18) 
    axes.set_xlim([0.0,77.0])
    axes.set_ylim([-2.5,55.0])
    axes.set_ylabel(r'Percentage of positive cells ($\%$)', size=20)
    axes.set_xlabel(r'Time (days)', size=20)
    axes.set_xticks([0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0])
    axes.set_xticklabels([0,10,20,30,40,50,60,70])
    axes.tick_params(axis='both', which='major', labelsize=20)
    fig1.tight_layout()
    # Save Fig
    fig1.savefig(output_path+'Fig2_fraction_pos_cells_'+stn1+'_'+repr(p_stn1)+'_'+stn2+'_'+repr(p_stn2)+'.pdf', dpi=400)
    fig1.savefig(output_path+'Fig2_fraction_pos_cells_'+stn1+'_'+repr(p_stn1)+'_'+stn2+'_'+repr(p_stn2)+'.png', dpi=400)
    return()

#################### Plot commands #################### 
    
#f2a=Fig2_intensity_hist('BR1',t_rs,'P70','Klf4',L)

#f2b=Fig2_percentile_intensity_hist(b_rs,t_rs,t_pts,'Sox9','Klf4', 80, 80,'P2','P70', 12.0, L)

#f2c=Fig2_percentile_intensity_map_single(b_rs,t_rs,t_pts,'Klf4','Sox9', 80, 80,'P70','P2',12.0,L)

#f2d=Fig2_percentile_intensity_map_double(b_rs,t_rs,t_pts,'Klf4','Sox9', 80, 80,'P70','P2',12.0,L)

#f2e=Fig2_intensity_percentile_pos_cells_bxplts(b_rs,t_rs,t_pts,'Klf4', 'Sox9', 80, 80,'P70','P2', 12.0,L)

#f2f=Fig2_intensity_bxplts(b_rs,t_rs,t_pts,'Klf4',12.0,L)

#f2g=Fig2_intensity_scatter_plot(b_rs,t_rs,t_pts,'Klf4','Sox9',80,80,'P70','P2',12.0,L)

#f2h=Fig2_intensity_percentile_pos_cells_plts(b_rs,t_rs,t_pts,'Klf4','Sox9',80,80,'P70','P2',12.0,L)


#################### Figure 4 #################### 

# Fig.4 - Boxplots of average cell density
def Fig4_cell_density_bxplts(b_rs,t_rs,t_pts,d,L):
    # Create an array to store the final data
    data_c=np.zeros((len(t_pts)), dtype=object)
    # Import raw intensity data, normalise them according to their type and concatenate together the different technical replicate along a single column
    for i in range (0, len(t_pts)):
        l_d=[]
        l_b=[]
        for j in range (0,len(b_rs)):
            for k in range (0, len(t_rs)):
                data=np.genfromtxt(input_path+b_rs[j]+'_'+t_rs[k]+'_'+t_pts[i]+'.csv', delimiter=",", skip_header=1) # Load data file 
                # Cells on the boundary of the image  
                b_c=f_bd_pts((data[:,2],data[:,3]),d)
                for l in range (0,len(data[:,2])):
                    if any(b_c==l):
                        l_b.append(l)
                    else:
                        density=np.divide((len(data[:,2])-len(b_c)),((L-(2.0*d))*10**-3)**2)
                l_d.append(density)
        data_c[i]=l_d
    data_c=np.delete(data_c,0) # Remove P2 
    fig1,axes = plt.subplots(1,1,figsize=(4.0,4.0))
    axes.plot
    axes= sns.swarmplot(data=list(data_c.T), size=4.0, color='0.25', alpha=0.75, zorder=1)
    axes=sns.boxplot(data=list(data_c.T), color="white", width=0.6, whis=1.5, showmeans=True, meanline=True, showfliers=False, whiskerprops=dict(linewidth=1.5, color='black', alpha=0.7), capprops=dict(linewidth=1.5, color='black', alpha=0.7), boxprops=dict(linewidth=1.5, edgecolor='black', alpha=0.7), medianprops=dict(linestyle='-', linewidth=1.5, color='black',alpha=0.7), meanprops=dict(linestyle='--', linewidth=1.5, color='red', alpha=0.7), zorder=1)
    axes.set_ylim([1.7*10**4,3.75*10**4])
    axes.set_ylabel(r'Cell density - $\rho$ ($cells/mm^2$)', size=22)
    axes.set_xticklabels(t_pts_p, fontsize=22)
    axes.tick_params(axis='both', which='major', labelsize=20)
    axes.ticklabel_format(style='sci', scilimits=(1,3), axis='y')
    fig1.tight_layout()
    # Save Fig
    fig1.savefig(output_path+'Fig4_Cell_Density_Boxplots.pdf', dpi=400)
    fig1.savefig(output_path+'Fig4_Cell_Density_Boxplots.png', dpi=400)
    # Save data as csv file with appropriate header
    b = list(data_c)
    df = pd.DataFrame(b)
    dfs=df.T
    dfs.to_csv(output_path +'Fig4_Cell_Density_Boxplots_Data.csv', header=t_pts_p, sep = ",")
    return()

# Fig.4 - Boxplots of Shape Anisotropy
def Fig4_shape_anisotropy_bxplts(b_rs,t_rs,t_pts,gm1,gm2,d,L):
    # Create an array to store the final data
    data_c=np.zeros((len(t_pts)), dtype=object)
    # Import raw intensity data, normalise them according to their type and concatenate together the different technical replicate along a single column
    for i in range (0, len(t_pts)):
        l_d=[]
        l_b=[]
        for k in range (0,len(b_rs)):
            for j in range (0, len(t_rs)):
                data=np.genfromtxt(input_path+b_rs[k]+'_'+t_rs[j]+'_'+t_pts[i]+'.csv', delimiter=",", skip_header=1) # Load data file 
                long_axis=data[:,gm_dic[gm1]]
                short_axis=data[:,gm_dic[gm2]]
                S_aniso= np.divide(np.subtract(long_axis,short_axis), np.sqrt(np.square(long_axis)+np.square(short_axis)))
                # Cells on the boundary of the image  
                b_c=f_bd_pts((data[:,2],data[:,3]),d)
                for l in range (0,len(data[:,2])):
                    if any(b_c==l):
                        l_b.append(l)
                    else:
                        l_d.append(S_aniso[l])
        data_c[i]=l_d
    data_c=np.delete(data_c,0) # Remove P2 prior ploting
    fig1,axes = plt.subplots(1,1,figsize=(4.0,4.0))
    axes.plot
    axes=sns.violinplot(data=list(data_c.T), cut=0, scale="count", inner=None, width=0.8, linewidth=0.0, color='lightgrey', alpha=0.7)
    axes=sns.boxplot(data=list(data_c.T), width=0.25, whis=1.0, showcaps=False, showmeans=True, meanline=True, showfliers=False, boxprops=dict(linewidth=1.0, facecolor='none', edgecolor='black', alpha=0.75, zorder=1), whiskerprops=dict(linewidth=1.0, color='black', alpha=0.75), medianprops=dict(linestyle='-', linewidth=1.0, color='black',alpha=0.75), meanprops=dict(linestyle='-', linewidth=1.0, color='red', alpha=0.75))
#    axes.annotate((r'Aspect Ratio $=\frac{l_{axis}}{s_{axis}}$'), xy=(0.45, 0.88), xycoords="axes fraction", bbox=dict(facecolor='none', edgecolor='black', pad=7.0), fontsize=18) 
    axes.set_ylim([-0.02,0.9])
    axes.set_ylabel(r'Shape anisotropy', size=22)
    axes.set_xticklabels(t_pts_p, fontsize=22)
    axes.tick_params(axis='both', which='major', labelsize=20)
    fig1.tight_layout()
    # Save Fig
    fig1.savefig(output_path+'Fig4_Shape_Anisotropy_Boxplots.pdf', dpi=400)
    fig1.savefig(output_path+'Fig4_Shape_Anisotropy_Boxplots.png', dpi=400)
    # Save data as csv file with appropriate header
    b = list(data_c)
    df = pd.DataFrame(b)
    dfs=df.T
    dfs.to_csv(output_path +'Fig4_Shape_Anisotropy_Boxplots_Data.csv', header=t_pts_p, sep = ",")
    return()    
    
# Fig.4 - Boxplots of the nematic order parameter 
def Fig4_nematic_OP_bxplts(b_rs,t_rs,t_pts,gm,d,L):
    # Create an array to store the final data
    data_c=np.zeros((len(t_pts)), dtype=object)
    # Import raw intensity data, normalise them according to their type and concatenate together the different technical replicate along a single column
    for i in range (0, len(t_pts)):
        l_d=[]
        l_b=[]
        for k in range (0,len(b_rs)):
            # Find the max value amongst replicates and time points
#            data_max=f_max_val(b_rs[k],t_rs,t_pts,gm)
            for j in range (0, len(t_rs)):
                data=np.genfromtxt(input_path+b_rs[k]+'_'+t_rs[j]+'_'+t_pts[i]+'.csv', delimiter=",", skip_header=1) # Load data file 
                angle=data[:, gm_dic[gm]]
                # Cells on the boundary of the image  
                b_c=f_bd_pts((data[:,2],data[:,3]),d)
                for l in range (0,len(data[:,2])):
                    if any(b_c==l):
                        l_b.append(l)
                    else:
                        q=np.sqrt(np.square(np.mean(np.cos(2.0*np.deg2rad(angle)))) + np.square(np.mean(np.sin(2.0*np.deg2rad(angle))))) 
                l_d.append(q)
        data_c[i]=l_d
    data_c=np.delete(data_c,0) # Remove P2 prior ploting 
    fig1,axes = plt.subplots(1,1,figsize=(4.0,4.0))
#    axes= sns.swarmplot(data=list(data_c.T), size=3.75, color='0.25', alpha=0.75, zorder=1)
#    axes=sns.boxplot(data=list(data_c.T), palette=col_bp, width=0.5, showmeans=True, meanline=True, showfliers=False, whiskerprops=dict(linewidth=1.5, color='black', alpha=0.6), capprops=dict(linewidth=1.5, color='black', alpha=0.6), boxprops=dict(linewidth=1.5, alpha=0.6), medianprops=dict(linestyle='-', linewidth=1.5, color='black',alpha=0.6), meanprops=dict(linestyle='--', linewidth=1.5, color='black', alpha=0.6), zorder=0)
    axes= sns.swarmplot(data=list(data_c.T), size=4.0, color='0.25', alpha=0.75, zorder=1)
    axes=sns.boxplot(data=list(data_c.T), color="white", width=0.6, whis=1.5, showmeans=True, meanline=True, showfliers=False, whiskerprops=dict(linewidth=1.5, color='black', alpha=0.7), capprops=dict(linewidth=1.5, color='black', alpha=0.7), boxprops=dict(linewidth=1.5, edgecolor='black', alpha=0.7), medianprops=dict(linestyle='-', linewidth=1.5, color='black',alpha=0.7), meanprops=dict(linestyle='--', linewidth=1.5, color='red', alpha=0.7), zorder=1)
    axes.set_ylim([0.0,1.0])
    axes.set_ylabel(r'Nematic order parameter', size=22)
    axes.set_xticklabels(t_pts_p, fontsize=22)
    axes.tick_params(axis='both', which='major', labelsize=20)
    axes.plot
    fig1.tight_layout()
    # Save Fig
    fig1.savefig(output_path+'Fig4_Nematic_OP_Boxplots.pdf', dpi=400)
    fig1.savefig(output_path+'Fig4_Nematic_OP_Boxplots.png', dpi=400)
    # Save data as csv file with appropriate header
    b = list(data_c)
    df = pd.DataFrame(b)
    dfs=df.T
    dfs.to_csv(output_path +'Fig4_Nematic_OP_Boxplots_Data.csv', header=t_pts_p, sep = ",")
    return()
    
# Fig.4 - 2D Structure Factor S(k_x,k_y) of the tissue 
def Fig4_SF_2D_tissue(b_r,t_rs,t_pt,N,L):
    data_rho=np.zeros((len(t_rs)), dtype=object) 
    data_S=np.zeros((len(t_rs)), dtype=object) 
    # Import raw data and average on technical replicates for a given biological replicate
    for i in range (0,len(t_rs)):
        data=np.genfromtxt(input_path+b_r+'_'+t_rs[i]+'_'+t_pt+'.csv', delimiter=",", skip_header=1) # Load data file 
        (rho,m,k_x,k_y,S)=structure_factor_2D((data[:,2],data[:,3]),N,L)
        data_rho[i]=rho
        data_S[i]=S
    rho_av=np.mean(data_rho)
    S_av=np.mean(data_S)
    x=np.divide(k_x,(2.0*np.pi*np.sqrt(rho_av)))
    y=np.divide(k_y,(2.0*np.pi*np.sqrt(rho_av)))
    # Plot 
    fig1,axes = plt.subplots(1,1,figsize=(4.5,4.0))
    im=axes.imshow(S_av, cmap='magma', interpolation='nearest', origin='lower', aspect='auto', extent=(np.min(x), np.max(x), np.min(y), np.max(y)))
    cbar=fig1.colorbar(im,ax=axes,cax=make_axes_locatable(plt.gca()).append_axes("right", "5%", pad="3%"))
    axes.set_ylabel(r'$k_y/2 \pi \sqrt{\rho}$', fontsize=20)
    axes.set_xlabel(r'$k_x/2 \pi \sqrt{\rho}$', fontsize=20)
    axes.tick_params(axis='both', which='major', labelsize=20)
    fig1.tight_layout()
    # Save Fig 
    fig1.savefig(output_path+'Fig4_SF_2D_'+b_r+'_'+t_pt+'.pdf', dpi=400)
    fig1.savefig(output_path+'Fig4_SF_2D_'+b_r+'_'+t_pt+'.png', dpi=400)
    return()

# Fig.4 - Plot of shape anisotropy tensors as color-coded ellipses
def Fig4_anistropy_tensor_map(b_r,t_r,t_pt,gm1,gm2,gm3,d,L):
    # Import data
    data=np.genfromtxt(input_path+b_r+'_'+t_r+'_'+t_pt+'.csv', delimiter=",", skip_header=1)
    # Coordinates of cells centroids 
    x=data[:,2]
    y=data[:,3] 
    # Long axis angle data
    d_g1=data[:,gm_dic[gm1]]
    ang_l=conv_angle(d_g1) # Convert angles from (0, pi) to (-90,90)
    # Short axis angle data
    d_g2=data[:,gm_dic[gm1]]+90
    ang_s=conv_angle(d_g2) # Convert angles from (0, pi) to (-90,90)
    # Long axis data
    l_ax=data[:,gm_dic[gm2]] 
    # Short axis data
    s_ax=data[:,gm_dic[gm3]] 
    # Normalize the chosen colour map and make it mappable to the stainings intensity values
    norm=mpl.colors.Normalize(vmin=-90.0,vmax=90.0)
    mapper=cm.ScalarMappable(norm=norm,cmap='hsv')   
    mapper.set_array([]) # Allow to associate a colorbar
    # Tissue voronoi diagram
    vor=voronoi((data[:,2],data[:,3])) # Voronoi diagram 
    regions,vertices=voronoi_finite_cells(vor) # Finite cells voronoi diagram 
    # Find cells on the boundary of the image  
    b_c=f_bd_pts((data[:,2],data[:,3]),d)
    # Sort cells as an ellipse collection
    E_b=[]
    E_c=[]
    k=-1
    for region in regions:
        k=k+1
        elp=Ellipse((x[k],y[k]), l_ax[k], s_ax[k], angle=ang_l[k], facecolor=mapper.to_rgba(ang_l[k]))
        if any(b_c==k):
            E_b.append(elp)
        else:
            E_c.append(elp)
        p_E_c=PatchCollection(E_c, match_original=True, edgecolor='black', linewidth=0.5, alpha=0.75)
        p_E_b=PatchCollection(E_b, facecolor='none', edgecolor='black', linewidth=0.25, alpha=0.75) # Remove colormaping for cells on the tissue border
    # Vector components of long axis
    u_l = np.multiply(np.cos(ang_l),l_ax)
    v_l = np.multiply(np.sin(ang_l),l_ax)
    # Vector components of short axis
#    u_s = np.multiply(np.cos(ang_s),s_ax)
#    v_s = np.multiply(np.sin(ang_s),s_ax)
    # Plot 
    fig1,axes = plt.subplots(1,1,figsize=(4.75,4.0))
    # Plot the ellipses
    axes.add_collection(p_E_b)   
    axes.add_collection(p_E_c)
    # Plot long axes
    axes.quiver(x,y,u_l,v_l, angles=ang_l, pivot='middle', scale=1.25, units='xy', width=0.5, headaxislength=0.0, headlength=0.0, headwidth=0.0, alpha=0.8) # Quiver plot of long axis 
    # Plot short axes
#    axes.quiver(x,y,u_s,v_s, angles=ang_s, pivot='middle', scale=1.0, units='xy', width=0.25, headaxislength=0.0, headlength=0.0, headwidth=0.0, alpha=0.8) # Quiver plot of long axis
    # Add color bar for long axis angle 
    cbar=fig1.colorbar(mapper, ax=axes, cax=make_axes_locatable(plt.gca()).append_axes("right", "5%", pad="3%"))
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=18)
    axes.set_ylim([0.0,L])
    axes.set_xlim([0.0,L])
    axes.set_xticks([])
    axes.set_yticks([])
    fig1.tight_layout()
    # Save Fig 
    fig1.savefig(output_path+'Fig4_anistropy_tensor_map_'+b_r+'_'+t_r+'_'+t_pt+'.pdf', dpi=400)
    fig1.savefig(output_path+'Fig4_anistropy_tensor_map_'+b_r+'_'+t_r+'_'+t_pt+'.png', dpi=400)
    return()
 
# Fig.4 - Anisotropy of the 2D Structure Factor S(k_x,k_y) of the tissue 
def Fig4_SF_anisotropy_2D_tissue(b_rs,t_rs,t_pts,N,L):
    # Create an array to store the final data
    data_c=np.zeros((len(t_pts)), dtype=object)
    # Import raw intensity data, normalise them according to their type and concatenate together the different technical replicate along a single column
    for i in range (0, len(t_pts)):
        l_d=[]
        for k in range (0,len(b_rs)):
            for j in range (0, len(t_rs)):
                data=np.genfromtxt(input_path+b_rs[k]+'_'+t_rs[j]+'_'+t_pts[i]+'.csv', delimiter=",", skip_header=1) # Load data file 
                (rho,m,k_x,k_y,S)=structure_factor_2D((data[:,2],data[:,3]),N,L) # Compute 2D SF of tissue
                # Image processing
                S_im=gaussian(S,sigma=4) # Gaussian filter of kernel size sigma
                val=threshold_otsu(S_im) # Segmentation threshold via the otsu method
                mask=S_im>val # Segmented image
                mask_p=closing(mask,square(3)) # Morphological closing
                mask_l=label(mask_p,background=1) # Mask with detected foreground objects
                regions=regionprops(mask_l) # Individual objects detected in the mask
                print(t_pts[i],b_rs[k],t_rs[j])
                for region in regions:
                    if region.area >= 100: # Filter non-desirable small objects
                        # Centroid coordinates
                        x0,y0=region.centroid
                        # Long and short axes of best fit ellipse
                        l_ax=region.major_axis_length
                        s_ax=region.minor_axis_length
                        # Long axis angle with image x-axis
                        alpha=-1.0*np.degrees(region.orientation)
                        # Shape anisotropy
                        S_asf= np.divide(np.subtract(l_ax,s_ax), np.sqrt(np.square(l_ax)+np.square(s_ax)))
                        l_d.append(S_asf) # Store SF Anisotropy values
        data_c[i]=l_d
    data_c=np.delete(data_c,0) # Remove P2 prior ploting if needed
    # Plot boxplot of the shape anisotropy 
    fig2,axes = plt.subplots(1,1,figsize=(4.0,4.0))
    axes= sns.swarmplot(data=list(data_c.T), size=4.0, color='0.25', alpha=0.75, zorder=1)
    axes=sns.boxplot(data=list(data_c.T), color="white", width=0.6, whis=1.5, showmeans=True, meanline=True, showfliers=False, whiskerprops=dict(linewidth=1.5, color='black', alpha=0.7), capprops=dict(linewidth=1.5, color='black', alpha=0.7), boxprops=dict(linewidth=1.5, edgecolor='black', alpha=0.7), medianprops=dict(linestyle='-', linewidth=1.5, color='black',alpha=0.7), meanprops=dict(linestyle='--', linewidth=1.5, color='red', alpha=0.7), zorder=1)
    axes.set_ylim([-0.02,0.60])
    axes.set_ylabel(r'Structure factor anisotropy', size=22)
    axes.set_xticklabels(t_pts_p, fontsize=22)
    axes.tick_params(axis='both', which='major', labelsize=20)
    axes.plot
    fig2.tight_layout()
    # Save Fig
    fig2.savefig(output_path+'Fig4_SF_Anisotropy_2D_Boxplots.pdf', dpi=400)
    fig2.savefig(output_path+'Fig4_SF_Anisotropy_2D_Boxplots.png', dpi=400)
    # Save data as csv file with appropriate header
    b = list(data_c)
    df = pd.DataFrame(b)
    dfs=df.T
    dfs.to_csv(output_path +'Fig4_SF_Anisotropy_2D_Boxplots_Data.csv', header=t_pts_p, sep = ",")
    return()
 
#################### Plot Commands #################### 

#f4a=Fig4_cell_density_bxplts(b_rs,t_rs,t_pts,12.0,L)

#f4b=Fig4_shape_anisotropy_bxplts(b_rs,t_rs,t_pts,'l_axis','s_axis',12.0,L)

#f4c=Fig4_nematic_OP_bxplts(b_rs,t_rs,t_pts,'angle',12.0,L)
    
#f4d=Fig4_SF_2D_tissue('BR3',t_rs,'P14',100,L)

#f4e=Fig4_anistropy_tensor_map('BR1','TR1','P70','angle', 'l_axis', 's_axis',12.0,L)
    
#f4f=Fig4_SF_anisotropy_2D_tissue(b_rs,t_rs,t_pts,100,L)

