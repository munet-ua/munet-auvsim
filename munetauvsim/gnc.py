"""
Utility functions supporting GNC algorithms and vehicle dynamics.

Provides mathematical and computational support for guidance, navigation, and
control algorithms. Includes angle normalization, matrix transformations,
hydrodynamic force calculations, and system utilities based on Fossen's marine
vehicle dynamics framework.

Functions
---------
**Angle and Matrix Operations**
    ssa(angle) : Produces shortest signed angle to [-pi to pi).
    Smtrx(a) : Create 3x3 skew-symmetric matrix from vector.
    Hmtrx(r) : Compute 6x6 system transformation matrix.
    m2c(M, nu) : Converts mass matrix and velocity to Coriolis matrix.
**Hydrodynamic Forces**
    Hoerner(B, T) : Compute 2D Hoerner cross-flow drag coefficient.
    crossFlowDrag(L, B, T, nu) : Calculates cross-flow drag forces.
    forceLiftDrag(b, S, CD_0, alpha, U_r) : Computes lift and drag forces.
    gvect(W, B, theta, phi, r_bg, r_bb) : Computes restoring forces.
**Utilities**
    saturation(value, limit, maxLimit) : clamp value to interval.

References
----------
[1] Fossen, T. I. and Perez, T. (2004). Marine Systems Simulator (MSS).
https://github.com/cybergalactic/MSS

[2] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
https://github.com/cybergalactic/PythonVehicleSimulator
"""

from typing import Optional
from numpy.typing import NDArray
import numpy as np
import math
from munetauvsim import logger

#-----------------------------------------------------------------------------#

# Type Aliases
NPFltArr = NDArray[np.float64]

# Global Variables
log = logger.addLog('gnc')

###############################################################################

def ssa(angle:float)->float:
    """
    Compute the smallest signed angle to range [-pi, pi).

    Two angles define two points on a circle and the smallest signed angle (SSA)
    is the smaller of the two arcs created between them. The sign corresponds to
    the direction taken around the circle with the clock-wise direction being
    positive.

    
    Parameters
    ----------
    angle : float
        Angle in radians (any value).
        
    Returns
    -------
    ssa_angle : float
        Angle wrapped to [-pi, pi).
        
    Notes
    -----
    If the angle provided is the difference between two angles, then this
    function computes the smallest signed angle, and in other cases the result
    is to normalize the input angle to the range [-pi, pi).
    """

    return (angle + math.pi) % (2 * math.pi) - math.pi

###############################################################################

### Called by: gnc.Hmtrx(), gnc.m2c()
def Smtrx(a:NPFltArr)->NPFltArr:
    """
    Create 3x3 skew-symmetric matrix from vector.
    
    Parameters
    ----------
    a : array_like, shape (3,)
        Input vector [a1, a2, a3].
        
    Returns
    -------
    S : ndarray, shape (3, 3)
        Skew-symmetric matrix satisfying a x b = S(a)b.
        
    Notes
    -----
    - S(a) = -S(a)' (antisymmetric property).
    - Used in cross product and Coriolis matrix calculations.
    """
 
    S = np.array([ 
        [ 0,        -a[2],   a[1] ],
        [ a[2],      0,     -a[0] ],
        [-a[1],      a[0],     0  ]])

    return S

###############################################################################

### Called by: vehicle.__init__()
def Hmtrx(r:NPFltArr)->NPFltArr:
    """
    Compute 6x6 system transformation matrix.
    
    Parameters
    ----------
    r : array_like, shape (3,)
        Position vector [x, y, z] in meters.
        
    Returns
    -------
    H : ndarray, shape (6, 6)
        System transformation matrix.
        
    Notes
    -----
    H = [     eye(3),     S'
          zeros(3,3), eye(3) ]
    
    With the property: inv(H(r)) = H(-r)

    If r = r_bg is the vector from the CO to the CG, the model matrices in CO 
    and CG are related by: M_CO = H(r_bg)' * M_CG * H(r_bg).
    
    Generalized position and force satisfy: 
        eta_CO = H(r_bg)' * eta_CG and tau_CO = H(r_bg)' * tau_CG
    """

    H = np.identity(6,float)
    H[0:3, 3:6] = Smtrx(r).T

    return H

###############################################################################

### Called by: vehicle.dynamics()
def m2c(M:NPFltArr, nu:NPFltArr)->NPFltArr:
    """
    Compute Coriolis-centripetal matrix from mass matrix and velocity.
    
    Parameters
    ----------
    M : ndarray, shape (n, n)
        Mass matrix (n = 3 or 6 DOF).
    nu : ndarray, shape (n,)
        Body-frame velocity [u, v, w, p, q, r] (6-DOF)
        or [u, v, r] (3-DOF).
        
    Returns
    -------
    C : ndarray, shape (n, n)
        Coriolis-centripetal matrix.

    Notes
    -----
    - For 6-DOF: C computed from velocity-dependent skew-symmetric terms.
    - For 3-DOF: Simplified yaw-only formulation.
    - Used in dynamics: M d/dt nu + C(nu)*nu + D(nu)*nu + g(eta) = tau.
    """

    M = 0.5 * (M + M.T)     # systematization of the inertia matrix

    if (len(nu) == 6):      #  6-DOF model
    
        M11 = M[0:3,0:3]
        M12 = M[0:3,3:6] 
        M21 = M12.T
        M22 = M[3:6,3:6] 
    
        nu1 = nu[0:3]
        nu2 = nu[3:6]
        dt_dnu1 = np.matmul(M11,nu1) + np.matmul(M12,nu2)
        dt_dnu2 = np.matmul(M21,nu1) + np.matmul(M22,nu2)

        #C  = [  zeros(3,3)      -Smtrx(dt_dnu1)
        #      -Smtrx(dt_dnu1)  -Smtrx(dt_dnu2) ]
        C = np.zeros( (6,6) )    
        C[0:3,3:6] = -Smtrx(dt_dnu1)
        C[3:6,0:3] = -Smtrx(dt_dnu1)
        C[3:6,3:6] = -Smtrx(dt_dnu2)
            
    else:   # 3-DOF model (surge, sway and yaw)
        #C = [ 0             0            -M(2,2)*nu(2)-M(2,3)*nu(3)
        #      0             0             M(1,1)*nu(1)
        #      M(2,2)*nu(2)+M(2,3)*nu(3)  -M(1,1)*nu(1)          0  ]    
        C = np.zeros( (3,3) ) 
        C[0,2] = -M[1,1] * nu[1] - M[1,2] * nu[2]
        C[1,2] =  M[0,0] * nu[0] 
        C[2,0] = -C[0,2]       
        C[2,1] = -C[1,2]
        
    return C

###############################################################################

### Called by: gnc.crossFlowDrag()
def Hoerner(B:float, T:float)->float:
    """
    Compute 2D Hoerner cross-flow drag coefficient.
    
    Parameters
    ----------
    B : float
        Beam (width of hull at it's widest point) in meters.
    T : float
        Draft (depth of hull's deepest point below waterline) in meters.
        
    Returns
    -------
    CY_2D : float
        Cross-flow drag coefficient (dimensionless).
        
    Notes
    -----
    Uses interpolation of Hoerner's empirical curve based on B/(2T) ratio.
    """
    
    # DATA = [B/2T  C_D]
    DATA1 = np.array([
        0.0109,0.1766,0.3530,0.4519,0.4728,0.4929,0.4933,0.5585,0.6464,0.8336,
        0.9880,1.3081,1.6392,1.8600,2.3129,2.6000,3.0088,3.4508, 3.7379,4.0031 
        ])
    DATA2 = np.array([
        1.9661,1.9657,1.8976,1.7872,1.5837,1.2786,1.2108,1.0836,0.9986,0.8796,
        0.8284,0.7599,0.6914,0.6571,0.6307,0.5962,0.5868,0.5859,0.5599,0.5593 
        ])

    CY_2D = np.interp( B / (2 * T), DATA1, DATA2 )
        
    return CY_2D

###############################################################################

### Called by: vehicle.dynamics()
def crossFlowDrag(L:float,
                  B:float,
                  T:float,
                  nu_r:NPFltArr,
                  )->NPFltArr:
    """
    Compute cross-flow drag forces using strip theory.
    
    Parameters
    ----------
    L : float
        Hull length in meters.
    B : float
        Beam (width of hull at it's widest point) in meters.
    T : float
        Draft (depth of hull's deepest point below waterline) in meters.
    nu_r : ndarray, shape (6,)
        Relative linear and angular velocity [u, v, w, p, q, r] in body frame.
        
    Returns
    -------
    tau_crossflow : ndarray, shape (6,)
        Generalized force vector [X, Y, Z, K, M, N].
        Only Y (sway) and N (yaw) components nonzero.
        
    Notes
    -----
    - Divides hull into 20 strips and integrates drag along length.
    - Apply as:  M d/dt nu + C(nu)*nu + D(nu)*nu + g(eta) = tau + tau_crossflow.
    """

    rho = 1026               # density of water
    n = 20                   # number of strips

    dx = L/20             
    Cd_2D = Hoerner(B,T)    # 2D drag coefficient based on Hoerner's curve

    Yh = 0
    Nh = 0
    xL = -L/2
    
    for i in range(0,n+1):
        v_r = nu_r[1]             # relative sway velocity
        r = nu_r[5]               # yaw rate
        Ucf = abs(v_r + xL * r) * (v_r + xL * r)
        Yh = Yh - 0.5 * rho * T * Cd_2D * Ucf * dx         # sway force
        Nh = Nh - 0.5 * rho * T * Cd_2D * xL * Ucf * dx    # yaw moment
        xL += dx
        
    tau_crossflow = np.array([0, Yh, 0, 0, 0, Nh],float)

    return tau_crossflow

###############################################################################

### Called by: vehicle.dynamics()
def forceLiftDrag(b:float,
                  S:float,
                  CD_0:float,
                  alpha:float,
                  U_r:float,
                  )->NPFltArr:
    """
    Compute lift and drag forces on submerged wing profile.
    
    Parameters
    ----------
    b : float
        Wing span in meters.
    S : float
        Wing area in m^2.
    CD_0 : float
        Parasitic drag coefficient (at alpha=0), typically 0.1-0.2.
    alpha : float
        Angle of attack in radians.
    U_r : float
        Relative flow speed in m/s.
        
    Returns
    -------
    tau_liftdrag : ndarray, shape (6,)
        Generalized force [X, Y, Z, K, M, N] in body frame.
        Only X (surge) and Z (heave) components nonzero.
        
    Notes
    -----
    - Lift: CL(alpha) = CL_alpha * alpha (linear model)
    - Drag: CD(alpha) = CD_0 + CL^2 / (pi * e * AR)
    - Forces: F = 1/2 rho U^2 * S * C
    - Apply as: M d/dt nu + C(nu)nu + D(nu)nu + g(eta) = tau + tau_liftdrag.
    - Based on Beard & McLain (2012) aerodynamic model.
    """

    # constants
    rho = 1026

    def coeffLiftDrag(b:float,
                      S:float,
                      CD_0:float,
                      alpha:float,
                      sigma:float,
                      )->NPFltArr:
        """
        Return the hydrodynamic lift CL(alpha) and drag CD(alpha) coefficients
        as a function of alpha (angle of attack) of a submerged "wing profile"
        (Beard and McLain 2012)
        
        Parameters
        ----------
        b:
            Wing span (m)
        S:
            Wing area (m^2)
        CD_0:
            Parasitic drag (alpha = 0), typically 0.1-0.2 for a streamlined body
        alpha:
            Angle of attack, scalar or vector (rad)
        sigma:
            Blending parameter between 0 and 1, use sigma = 0 for linear lift 
        
        Returns
        -------
        CL:
            Lift coefficient as a function of alpha   
        CD:
            Drag coefficient as a function of alpha   

        Application
        -----------
        CD(alpha) = CD_p + (CL_0 + CL_alpha * alpha)^2 / (pi * e * AR)
        CL(alpha) = CL_0 + CL_alpha * alpha

        where CD_p is the parasitic drag (profile drag of wing, friction and
        pressure drag of control surfaces, hull, etc.), CL_0 is the zero angle
        of attack lift coefficient, AR = b^2/S is the aspect ratio and e is the
        Oswald efficiency number. For lift it is assumed that

        CL_0 = 0
        CL_alpha = pi * AR / ( 1 + sqrt(1 + (AR/2)^2) );

        implying that for alpha = 0, CD(0) = CD_0 = CD_p and CL(0) = 0. For
        high angles of attack the linear lift model can be blended with a
        nonlinear model to describe stall

        CL(alpha) = (1-sigma) * CL_alpha * alpha + ...
                    sigma * 2 * sign(alpha) * sin(alpha)^2 * cos(alpha) 

        where 0 <= sigma <= 1 is a blending parameter. 

        Example
        -------
        # Cylinder-shaped AUV with length L = 1.8, diameter D = 0.2 and 
        # CD_0 = 0.3
        >>> alpha = 0.1 * pi/180
        >>> [CL,CD] = coeffLiftDrag(0.2, 1.8*0.2, 0.3, alpha, 0.2)
        """
         
        e = 0.7             # Oswald efficiency number
        AR = b**2 / S       # wing aspect ratio

        # linear lift
        CL_alpha = math.pi * AR / ( 1 + math.sqrt(1 + (AR/2)**2) )
        CL = CL_alpha * alpha

        # parasitic and induced drag
        CD = CD_0 + CL**2 / (math.pi * e * AR)
        
        # nonlinear lift (blending function)
        CL = (1-sigma) * CL + sigma * 2 * np.sign(alpha) \
            * math.sin(alpha)**2 * math.cos(alpha)

        return [CL, CD]

    
    [CL, CD] = coeffLiftDrag(b,S,CD_0,alpha,0) 
    
    F_drag = 1/2 * rho * U_r**2 * S * CD    # drag force
    F_lift = 1/2 * rho * U_r**2 * S * CL    # lift force

    # transform from FLOW axes to BODY axes using angle of attack
    tau_liftdrag = np.array([
        math.cos(alpha) * (-F_drag) - math.sin(alpha) * (-F_lift),
        0,
        math.sin(alpha) * (-F_drag) + math.cos(alpha) * (-F_lift),
        0,
        0,
        0 ])

    return tau_liftdrag
    
###############################################################################

### Called by: vehicle.dynamics()
def gvect(W:float,
          B:float,
          theta:float,
          phi:float,
          r_bg:NPFltArr,
          r_bb:NPFltArr,
          )->NPFltArr:
    """
    Compute restoring force vector for submerged body.
    
    Parameters
    ----------
    W : float
        Weight in kg.
    B : float
        Buoyancy in kg.
    theta : float
        Pitch angle in radians.
    phi : float
        Roll angle in radians.
    r_bg : array_like, shape (3,)
        Position [x_g, y_g, z_g] of CG relative to CO (meters).
    r_bb : array_like, shape (3,)
        Position [x_b, y_b, z_b] of CB relative to CO (meters).
        
    Returns
    -------
    g : ndarray, shape (6,)
        Restoring force vector [X, Y, Z, K, M, N] about CO.
        
    Notes
    -----
    - Accounts for weight, buoyancy, and moment arms.
    - Neutral buoyancy: W = B -> only moment terms nonzero.
    - Apply as: M d/dt nu + C(nu)nu + D(nu)nu + g(eta) = tau.
    """

    sth  = math.sin(theta)
    cth  = math.cos(theta)
    sphi = math.sin(phi)
    cphi = math.cos(phi)

    g = np.array([
        (W-B) * sth,
        -(W-B) * cth * sphi,
        -(W-B) * cth * cphi,
        -(r_bg[1]*W-r_bb[1]*B)*cth*cphi + (r_bg[2]*W-r_bb[2]*B) * cth * sphi,
        (r_bg[2]*W-r_bb[2]*B)*sth       + (r_bg[0]*W-r_bb[0]*B) * cth * cphi,
        -(r_bg[0]*W-r_bb[0]*B)*cth*sphi - (r_bg[1]*W-r_bb[1]*B) * sth
        ])
    
    return g

###############################################################################

def saturation(value:float,
               limit:float,
               maxLimit:Optional[float]=None,
               )->float:
    """
    Clamp value to specified interval.
    
    Parameters
    ----------
    value : float
        Value to limit.
    limit : float
        Lower limit if maxLimit provided, else absolute limit for symmetric interval.
    maxLimit : float, optional
        Upper limit. If None, uses symmetric interval [-limit, +limit].
        
    Returns
    -------
    clamped : float
        Value restricted to [limit, maxLimit] or [-limit, +limit].
        
    Examples
    --------
    >>> saturation(5.0, 2.0)         # interval [-2, 2]
    2.0
    >>> saturation(5.0, 0.0, 10.0)   # interval [0, 10]
    5.0
    >>> saturation(-5.0, -3.0, 3.0)  # interval [-3, 3]
    -3.0
    """

    # Determine Maximum and Minimum Boundary Values
    if (maxLimit is None):
        limit = abs(limit)
        maxLimit = limit
        lowLimit = -limit
    else:
        lowLimit = limit

    # Saturate Value with Boundary Values
    return np.clip(value, lowLimit, maxLimit)

###############################################################################