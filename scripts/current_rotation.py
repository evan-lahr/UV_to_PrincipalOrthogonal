import matplotlib.pyplot as plt
import numpy as np

def rotate_UV(u, v, theta_degrees):
    """
    Rotate velocity components u, v by an angle theta.
    
    Parameters:
    u (array-like): Eastward velocity components
    v (array-like): Northward velocity components
    theta_degrees (float): Angle (in degrees) to rotate by
    
    Returns:
    V1 (array-like): Rotated velocity component along principal axis
    V2 (array-like): Rotated velocity component orthogonal to principal axis
    """
    theta_radians = np.radians(theta_degrees)
    
    # Rotate components along both axes
    v1 = u * np.cos(theta_radians) + v * np.sin(theta_radians)
    v2 = -u * np.sin(theta_radians) + v * np.cos(theta_radians)
    
    return v1, v2


def velocity_to_angle(u, v):
    """
    Convert eastward and northward velocity components to angle from north.
    Parameters:
    u (array-like): Eastward velocity components
    v (array-like): Northward velocity components
    Returns:
    numpy array: Angles in degrees from north, in the range [0, 360)
    """
    theta_radians = np.arctan2(u, v)
    theta_degrees = np.degrees(theta_radians)
    theta_degrees = (theta_degrees + 360) % 360
    return theta_degrees


def rot_principal(u, v):
    """
    Convert UV velocity components to velocity components along/across principal axis.
   
    Parameters:
    u (array-like): Eastward velocity components
    v (array-like): Northward velocity components
    Returns:
    V1 (array-like): Principal velocity components
    V2 (array-like): Principal-orthogonal velocity components
    """
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 3))
    
    # UV components - compute current angle and plot angle histogram
    current_angle = velocity_to_angle(u, v)
    hist_data = ax1.hist(current_angle, 100,color= 'tab:blue',alpha=0.5)
    
    u_cs = u.cumsum()
    v_cs = v.cumsum()
    ax2.scatter(u_cs, v_cs, s=2, c='tab:blue')
    
    # Compute principal orientation
    principal_orientation = hist_data[1][np.argmax(hist_data[0])]
    orthogonal_orientation = (principal_orientation + 180) % 360
    
    # Adjust the principal orientation so the peak is at 0°
    corrected_principal_orientation = (360 - principal_orientation) % 360
    
    # Rotate to principal and orthogonal orientations
    v1, v2 = rotate_UV(u, v, corrected_principal_orientation)
    rotated_current_angle = velocity_to_angle(v1, v2)
    xx=ax1.hist(rotated_current_angle,100,alpha=0.5,color='tab:red')
    
    # Plot progressive vector
    v1_cs = v1.cumsum()
    v2_cs = v2.cumsum()
    ax2.scatter(v1_cs, v2_cs, s=2, c='tab:red')
    
    # Annotate plots
    ax1.axvline(x=principal_orientation, ymin=0, ymax=hist_data[0][np.argmax(hist_data[0]) + 1], c='0.5')
    ax1.axvline(x=orthogonal_orientation, ymin=0, ymax=hist_data[0][np.argmax(hist_data[0]) + 1], c='0.5')
    ax1.text(.015, .9, f'V1: {np.round(principal_orientation, 1)}', transform=ax1.transAxes, fontsize='large',fontweight='bold',c='k')
    ax1.text(.015, .8, f'V2: {np.round(orthogonal_orientation, 1)}', transform=ax1.transAxes, fontsize='large',fontweight='bold',c='k')
    ax2.text(.05, .9, f'Before', transform=ax2.transAxes, fontsize='large',fontweight='bold',c='tab:blue')
    ax2.text(.05, .8, f'After', transform=ax2.transAxes, fontsize='large',fontweight='bold',c='tab:red')    
    ax2.scatter(0, 0, s=100, c='k')
    ax2.axis('equal')
    ax1.set_title('Current Angle Histogram')
    ax2.set_title('Progressive Vector')
    plt.suptitle('Current Rotation: UV to principal current orientation',fontweight='bold',y=1.05)
    return v1, v2