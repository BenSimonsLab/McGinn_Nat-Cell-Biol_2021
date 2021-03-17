############################################################################################################
#----------------------------------------------------------------------------------------------------------#
#------------------------- Oesophagus Epithelium F-Actin Stainings Data Analysis --------------------------#
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
from scipy.spatial import Delaunay, Voronoi, ConvexHull

# Use Latex for font rendering in figures
rc('font',**{'family':'sans-serif','sans-serif':['cm']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']

################################################ Data Path ###############################################

"""
Provide local paths to input data and output data  
"""

input_path= '/home/adrien/Desktop/Image_Analysis_McGinn_Nat-Cell-Biol_2021/Oesophagus_Epithelium_F-Actin_Stainings/Data/Data_'

output_path='/home/adrien/Desktop/Image_Analysis_McGinn_Nat-Cell-Biol_2021/Oesophagus_Epithelium_F-Actin_Stainings/Plots/'

########################################## Data Structure #############################################

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
    return [value for value in the_list if value != val]
    
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
                # Extract relevent data and normalise them by the min intensity
                data=np.genfromtxt(input_path+b_rs[j]+'_'+t_rs[k]+'_'+t_pts[i]+'.csv', delimiter=",", skip_header=1) # Load data file
                d1=np.divide(data[:,stn_dic[stn1]],np.min(data[:,stn_dic[stn1]]))     
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
                 # Extract relevent data and normalise them by mini intensity
                 data=np.genfromtxt(input_path+b_rs[j]+'_'+t_rs[k]+'_'+t_pts[i]+'.csv', delimiter=",", skip_header=1) # Load data file         
                 d1=np.divide(data[:,stn_dic[stn1]],np.min(data[:,stn_dic[stn1]]))     
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
    Find indices of nuclei centroids within an arbitrary distance d of domain boundaries. 

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
    Find indices and coordinates of nuclei centroids within an arbitrary distance d of domain boundaries. 

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
  
def structure_factor_2D(c_xy,N,L):    
    """
    Compute 2D tissue structure factor S(k) for a range of wavenumber k using nuclei centroid positions as an input.

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

######################################################## Plots ##########################################################  
    
#################### Figure 2 #################### 

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
                Int=np.divide(data[:,stn_dic[stn]],np.min(data[:,stn_dic[stn]]))
                # Cells on the boundary of the image  
                b_c=f_bd_pts((data[:,2],data[:,3]),d)
                for l in range (0,len(data[:,2])):
                    if any(b_c==l):
                        l_b.append(l)
                    else:
                        l_d.append(Int[l])
        data_c[i]=l_d
    fig1,axes = plt.subplots(1,1,figsize=(4.4,4.0))
    axes.plot
    axes=sns.violinplot(data=list(data_c.T), cut=0, scale="count", inner=None, width=0.6, linewidth=0.0, color='lightgrey')
    axes=sns.boxplot(data=list(data_c.T), width=0.15, whis=1.5, showcaps=False, showmeans=True, meanline=True, showfliers=False, boxprops=dict(linewidth=1.0, facecolor='none', edgecolor='black', alpha=0.75, zorder=1), whiskerprops=dict(linewidth=1.0, color='black', alpha=0.75), medianprops=dict(linestyle='-', linewidth=1.0, color='black',alpha=0.75), meanprops=dict(linestyle='-', linewidth=1.0, color='red', alpha=0.75))
#    axes.set_ylim([0.5,6.0])
    axes.set_ylabel(str(stn) + r'\, mean intensity (a.u)', size=22)
    axes.set_xticklabels(t_pts, fontsize=22) 
    axes.tick_params(axis='both', which='major', labelsize=20)
    fig1.tight_layout()
    # Save Fig
    fig1.savefig(output_path+'Fig2_Int_Boxplots_'+stn+'.pdf', dpi=400)
    fig1.savefig(output_path+'Fig2_Int_Boxplots_'+stn+'.png', dpi=400)
    # Save data as csv file with appropriate header
    b = list(data_c)
    df = pd.DataFrame(b)
    dfs=df.T
    dfs.to_csv(output_path +'Fig5_Int_Boxplots_'+stn+'_Data.csv', header=t_pts, sep = ",")
    return()  
 
    
# Fig.2 - Boxplots of average cell density
def Fig2_cell_density_bxplts(b_rs,t_rs,t_pts,d,L):
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
    fig1,axes = plt.subplots(1,1,figsize=(4.0,4.0))
    axes.plot
    axes= sns.swarmplot(data=list(data_c.T), size=4.0, color='0.25', alpha=0.7, zorder=1)
    axes=sns.boxplot(data=list(data_c.T), color="white", width=0.6, whis=1.5, showmeans=True, meanline=True, showfliers=False, whiskerprops=dict(linewidth=1.5, color='black', alpha=0.7), capprops=dict(linewidth=1.5, color='black', alpha=0.7), boxprops=dict(linewidth=1.5, edgecolor='black', alpha=0.7), medianprops=dict(linestyle='-', linewidth=1.5, color='black',alpha=0.7), meanprops=dict(linestyle='--', linewidth=1.5, color='red', alpha=0.7), zorder=1)
#    axes.set_ylim([1.90*10**4,5.15*10**4])
    axes.set_ylabel(r'Cell density ($cells/mm^2$)', size=22)
    axes.set_xticklabels(t_pts, fontsize=22)
    axes.tick_params(axis='both', which='major', labelsize=20)
    axes.ticklabel_format(style='sci', scilimits=(1,3), axis='y')
    fig1.tight_layout()
    # Save Fig
    fig1.savefig(output_path+'Fig2_Cell_Density_Boxplots.pdf', dpi=400)
    fig1.savefig(output_path+'Fig2_Cell_Density_Boxplots.png', dpi=400)
    # Save data as csv file with appropriate header
    b = list(data_c)
    df = pd.DataFrame(b)
    dfs=df.T
    dfs.to_csv(output_path +'Fig2_Cell_Density_Boxplots_Data.csv', header=t_pts, sep = ",")
    return()
    
# Fig.2 - Boxplots of cells area 
def Fig2_cell_Area_bxplts(b_rs,t_rs,t_pts,gm1,d,L):
    # Create an array to store the final data
    data_c=np.zeros((len(t_pts)), dtype=object)
    # Import raw intensity data, normalise them according to their type and concatenate together the different technical replicate along a single column
    for i in range (0, len(t_pts)):
        l_b=[]
        l_d=[]
        for j in range (0,len(b_rs)):
            for k in range (0, len(t_rs)):
                data=np.genfromtxt(input_path+b_rs[j]+'_'+t_rs[k]+'_'+t_pts[i]+'.csv', delimiter=",", skip_header=1) # Load data file 
                data_1=data[:, gm_dic[gm1]]
                # Cells on the boundary of the image  
                b_c=f_bd_pts((data[:,2],data[:,3]),d)
                for l in range (0,len(data[:,2])):
                    if any(b_c==l):
                        l_b.append(l)
                    else:
                        l_d.append(data_1[l])
        data_c[i]=l_d
    fig1,axes = plt.subplots(1,1,figsize=(4.0,4.0))
    axes.plot
    for i in [10,20,30,40,50,60,70,80,90,100,110]:
        axes.plot([-100,200], [i,i], linestyle='--', linewidth=1.0, color='black', alpha=0.45)
    axes=sns.violinplot(data=list(data_c.T), cut=0, scale="count", inner=None, width=0.6, linewidth=0.0, color='lightsteelblue', alpha=0.7)
    axes=sns.boxplot(data=list(data_c.T), width=0.25, whis=1.5, showcaps=False, showmeans=True, meanline=True, showfliers=False, boxprops=dict(linewidth=1.0, facecolor='none', edgecolor='black', alpha=0.75, zorder=1), whiskerprops=dict(linewidth=1.0, color='black', alpha=0.75), medianprops=dict(linestyle='-', linewidth=1.0, color='black',alpha=0.75), meanprops=dict(linestyle='-', linewidth=1.0, color='red', alpha=0.75))
    axes.set_ylim([0.0,100.00])
    axes.set_ylabel(r'Area ($\mu m^2$)', size=22)
    axes.set_xticklabels(t_pts, fontsize=22)
    axes.tick_params(axis='both', which='major', labelsize=20)
    axes.plot
    fig1.tight_layout()
    # Save Fig
    fig1.savefig(output_path+'Fig2_Area_Boxplots.pdf', dpi=400)
    fig1.savefig(output_path+'Fig2_Area_Boxplots.png', dpi=400)
    # Save data as csv file with appropriate header
    b = list(data_c)
    df = pd.DataFrame(b)
    dfs=df.T
    dfs.to_csv(output_path +'Fig2_Area_Boxplots_Data.csv', header=t_pts, sep = ",")
    return()
    
# Fig.2 - Boxplots of the nematic order parameter 
def Fig2_nematic_OP_bxplts(b_rs,t_rs,t_pts,gm,d,L):
    # Create an array to store the final data
    data_c=np.zeros((len(t_pts)), dtype=object)
    # Import raw intensity data, normalise them according to their type and concatenate together the different technical replicate along a single column
    for i in range (0, len(t_pts)):
        l_d=[]
        l_b=[]
        for k in range (0,len(b_rs)):
            for j in range (0, len(t_rs)):
                data=np.genfromtxt(input_path+b_rs[k]+'_'+t_rs[j]+'_'+t_pts[i]+'.csv', delimiter=",", skip_header=1) # Load data file 
                angle=conv_angle(data[:, gm_dic[gm]])
                l_ang=[]
                # Filter cells on the boundary of the image  
                b_c=f_bd_pts((data[:,2],data[:,3]),d)
                for l in range (0,len(data[:,2])):
                    if any(b_c==l):
                        l_b.append(l)
                    else:
                        l_ang.append(angle[l])
                #Compute the nematic order parameter for each Technical Replicate
                q=np.sqrt(np.square(np.mean(np.cos(2.0*np.deg2rad(l_ang)))) + np.square(np.mean(np.sin(2.0*np.deg2rad(l_ang))))) 
                l_d.append(q)
        data_c[i]=l_d
    fig1,axes = plt.subplots(1,1,figsize=(4.0,4.0))
    axes= sns.swarmplot(data=list(data_c.T), size=4.0, color='0.25', alpha=0.75, zorder=1)
    axes=sns.boxplot(data=list(data_c.T), color="white", width=0.6, whis=1.5, showmeans=True, meanline=True, showfliers=False, whiskerprops=dict(linewidth=1.5, color='black', alpha=0.7), capprops=dict(linewidth=1.5, color='black', alpha=0.7), boxprops=dict(linewidth=1.5, edgecolor='black', alpha=0.7), medianprops=dict(linestyle='-', linewidth=1.5, color='black',alpha=0.7), meanprops=dict(linestyle='--', linewidth=1.5, color='red', alpha=0.7), zorder=1)
    axes.set_ylim([0.5,1.0])
    axes.set_ylabel(r'Nematic order parameter', size=22)
    axes.set_xticklabels(t_pts, fontsize=22)
    axes.tick_params(axis='both', which='major', labelsize=20)
    axes.plot
    fig1.tight_layout()
    # Save Fig
    fig1.savefig(output_path+'Fig2_Nematic_OP_Boxplots.pdf', dpi=400)
    fig1.savefig(output_path+'Fig2_Nematic_OP_Boxplots.png', dpi=400)
    # Save data as csv file with appropriate header
    b = list(data_c)
    df = pd.DataFrame(b)
    dfs=df.T
    dfs.to_csv(output_path +'Fig2_Nematic_OP_Boxplots_Data.csv', header=t_pts, sep = ",")
    return()
    
# Fig.2 - Boxplots of Cell Shape Anisotropy
def Fig2_shape_anisotropy_bxplts(b_rs,t_rs,t_pts,gm1,gm2,d,L):
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
    fig1,axes = plt.subplots(1,1,figsize=(4.0,4.0))
    axes.plot 
    axes=sns.violinplot(data=list(data_c.T), cut=0, scale="count", inner=None, width=0.6, linewidth=0.0, color='lightsteelblue', alpha=0.7)
    axes=sns.boxplot(data=list(data_c.T), width=0.25, whis=1.0, showcaps=False, showmeans=True, meanline=True, showfliers=False, boxprops=dict(linewidth=1.0, facecolor='none', edgecolor='black', alpha=0.75, zorder=1), whiskerprops=dict(linewidth=1.0, color='black', alpha=0.75), medianprops=dict(linestyle='-', linewidth=1.0, color='black',alpha=0.75), meanprops=dict(linestyle='-', linewidth=1.0, color='red', alpha=0.75))
    axes.set_ylim([-0.02,1.0])
    axes.set_ylabel(r'Cell Shape Anisotropy', size=22)
    axes.set_xticklabels(t_pts, fontsize=22)
    axes.tick_params(axis='both', which='major', labelsize=20)
    fig1.tight_layout()
    # Save Fig
    fig1.savefig(output_path+'Fig2_Shape_Anisotropy_Boxplots.pdf', dpi=400)
    fig1.savefig(output_path+'Fig2_Shape_Anisotropy_Boxplots.png', dpi=400)
    # Save data as csv file with appropriate header
    b = list(data_c)
    df = pd.DataFrame(b)
    dfs=df.T
    dfs.to_csv(output_path +'Fig2_Shape_Anisotropy_Boxplots_Data.csv', header=t_pts, sep = ",")
    return()   

#################### Figure 5 ####################  
 
# Fig.5 - Boxplots of membrane 
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

#f2a=Fig2_intensity_bxplts(b_rs,t_rs,t_pts,'F-Actin',12.0,L)

#f2b=Fig2_cell_density_bxplts(b_rs,t_rs,t_pts,12.0,L)
        
#f2c=Fig2_cell_Area_bxplts(b_rs,t_rs,t_pts,'area',12.0,L)
 
#f2d=Fig2_nematic_OP_bxplts(b_rs,t_rs,t_pts,'angle',12.0,L)

#f2e=Fig2_shape_anisotropy_bxplts(b_rs,t_rs,t_pts,'l_axis','s_axis',12.0,L)

#f5a=Fig5_memb_intensity_bxplts(b_rs,t_rs,t_pts,'F-Actin')


