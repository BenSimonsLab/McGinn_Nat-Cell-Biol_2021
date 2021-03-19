############################################################################################################
#----------------------------------------------------------------------------------------------------------#
#------------------------------------------- SHG Imaging Analysis -----------------------------------------#
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

# Import python sub-libraries
import matplotlib.pyplot as plt

# Import built-in functions from matplotlib
from matplotlib import rc

# Use Latex for font rendering in figures
rc('font',**{'family':'sans-serif','sans-serif':['cm']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']

################################################### Data Path ###################################################

"""
Provide local paths to input data and output data  
"""

input_path='/home/adrien/Desktop/Image_Analysis/SHG_Imaging_Analysis/Data/Data_'

output_path='/home/adrien/Desktop/Image_Analysis/SHG_Imaging_Analysis/Plots/'

##################################################### Data ######################################################

"""
~~~~ Data structure dictionaries ~~~~

"""
# Dictonary of time points
tpts_dic={'P7':0,'P28':1,'P70':2}


"""
~~~~ Data structure list ~~~~

"""
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
~~~~ Scale ~~~~rc('font',**{'family':'sans-serif','sans-serif':['cm']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']

"""
# Image size (microns)
L=387.88

################################################# Data analysis functions ###############################################

def remove_values_from_list(the_list, val):
    """
    Function to remove multiple occurences of a given element in a list.
    
    """
    return [value for value in the_list if value != val]

def mean_hist(t_pt,b_r):    
    """
     Create an averaged histogram from technical replicates for a given biological replicate.
     
    """
    data=np.genfromtxt(input_path+b_r+'_'+t_pt+'.csv', delimiter=",", skip_header=1) # Load data file 
    b=data[:,:2] # Index and bin columns
    d=data[:,2:] # Array without index and bin columns
    d_p=np.mean(d,axis=1,dtype=np.float64) # Compute mean value of the number of elements in each bin amongst technical replicates
    data_p=np.column_stack((b,d_p))
    return(data_p)

def cor_mean_hist(t_pt,b_r):
    """
    Correct histograms for average dominant orientation of the image.
    """
    # Import pooled histogram
    data_p=mean_hist(t_pt,b_r)
    b=data_p[:,:2] # Index column
    d=data_p[:,2:] # Array without index and bin columns
    if t_pt=='P28' or t_pt=='P70':
        i_max=np.argmax(data_p[:,2]) # Index of the dominant orientation 
        ang_i=data_p[i_max,1] # Value of  dominant orientation 
        i_slice=np.int(np.around(np.abs(ang_i)))  # Number of histogram bins to be re-sliced.
        print(i_max,ang_i,i_slice)
        if ang_i < 0:
            d_f=d[:-(i_slice-1),:] # First N-i_slice elements of initial array
            d_l=d[-(i_slice-1):,:] # Last i_slice elements of initial array
            d_n=np.row_stack((d_l,d_f)) # Concatenate d_l on top of d_f = correcterd histogram 
            d_corr=np.column_stack((b,d_n))
        else:
            d_f=d[:(i_slice+1),:] # First i_slice elements of initial array
            d_l=d[(i_slice+1):,:] # Last N-i_slice elements of initial array
            d_n=np.row_stack((d_l,d_f)) # Concatenate d_l on top of d_f = correcterd histogram 
            d_corr=np.column_stack((b,d_n))
    else:
        d_corr=data_p
    return(d_corr)
    
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
    
################################################## Plots ######################################################  

#################### Extended Data Figure 6 ####################  

# Extended Data Fig.6 - Polar coordinates histogram of orientation
def Ex_Dat_Fig6_Orientation_Polar_Hist(t_pt,b_r):
    # Import pooled histogram
    data=cor_mean_hist(t_pt,b_r)
    # Select appropriate color given time point (P7: C3 (red) / P28: C0 (blue) / P70: C2 (green)
    if t_pt=='P7':
        col='C3'
    elif t_pt=='P28':
        col='C0'
    else:
        col='C2' 
    # Plot polar coordinates histogram 
    fig1,axes = plt.subplots(1,1,figsize=(4.5,4.5), subplot_kw=dict(projection='polar'))
    axes.bar(np.deg2rad(data[:,1]), data[:,2], np.deg2rad(1.0), bottom=0.0, align='edge', color=col, alpha=0.5)   
    axes.set_thetamin(-90.0)
    axes.set_thetamax(90.0)
#    axes.set_rmax(np.max(data[:,2]))s
    axes.set_theta_zero_location('N')
    axes.set_theta_direction('clockwise')
    axes.xaxis.grid(True,color='grey',linestyle='--',linewidth=0.5)
    axes.yaxis.grid(True,color='grey',linestyle='--',linewidth=0.5)
#    axes.set_xticklabels([-90, -75,-50,-25,0,25,50,75,90], fontsize=16)
    axes.set_yticklabels([])
    fig1.tight_layout()
    # Save Fig
    fig1.savefig(output_path+'Ex_Dat_Fig6_Orientation_Histogram_'+t_pt+'_'+b_r+'.pdf', dpi=400)
    fig1.savefig(output_path+'Ex_Dat_Fig6_Orientation_Histogram_'+t_pt+'_'+b_r+'.png', dpi=400)
    return(data)
    
################################################## Plots Commands ######################################################  

a1=Ex_Dat_Fig6_Orientation_Polar_Hist('P28','BR2')

