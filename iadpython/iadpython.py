import numpy as np
import ctypes
from ctypes.util import find_library

# libiad is available on github

#if finding the library fails, then just specify the location of your shared
#library explicitly
#libiad = ctypes.CDLL('/usr/local/lib/libiad.dylib')

libiad = ctypes.CDLL(find_library('iad'))

libiad.ez_RT.argtypes = (ctypes.c_int,         # n quadrature points
                         ctypes.c_double,      # slab index of refraction
                         ctypes.c_double,      # top slide index of refraction
                         ctypes.c_double,      # bottom slide index of refraction
                         ctypes.c_double,      # albedo mus/(mus+mua)
                         ctypes.c_double,      # optical thickness d*(mua+mus)
                         ctypes.c_double,      # scattering anisotropy g
                         ctypes.POINTER(ctypes.c_double),   # UR1
                         ctypes.POINTER(ctypes.c_double),   # UT1
                         ctypes.POINTER(ctypes.c_double),   # URU
                         ctypes.POINTER(ctypes.c_double)    # UTU
                        )

def basic_rt(n, nslab, ntop, nbot, a, b, g):
    """
    Calculates the total reflection and transmission for a turbid slab.
    
    basic_rt(n,nslab,ntop,nbot,a,b,g) returns [UR1,UT1,URU,UTU] for a slab optionally
    bounded by glass slides.  The slab is characterized by an albedo a, an
    optical thickness b, a scattering anisotropy g, and an index of refraction nslab.
    The top glass slides have an index of refraction ntop and the bottom slide has
    an index nbot.  If there are no glass slides, set ntop and nbottom to 1.
    n is the number of quadrature angles (more angles give better accuracy but slow
    the calculation by n**3, n must be a multiple of 4, typically 16 is good for forward
    calculations and 4 is appropriate for inverse calculations).  
    
    All parameters must be scalars.
    
    UR1 is the total reflection for normally incident collimated light.
    UT1 is the total transmission for normally incident collimated light.
    URU is the total reflection for diffuse incident light.
    UTU is the total transmission for diffuse incident light.
    """
    ur1 = ctypes.c_double()
    ut1 = ctypes.c_double()
    uru = ctypes.c_double()
    utu = ctypes.c_double()
    libiad.ez_RT(n, nslab, ntop, nbot, a, b, g, ur1, ut1, uru, utu)
    return ur1.value, ut1.value, uru.value, utu.value

libiad.ez_RT_unscattered.argtypes = (
    ctypes.c_int,         # n quadrature points
    ctypes.c_double,      # slab index of refraction
    ctypes.c_double,      # top slide index of refraction
    ctypes.c_double,      # bottom slide index of refraction
    ctypes.c_double,      # albedo mus/(mus+mua)
    ctypes.c_double,      # optical thickness d*(mua+mus)
    ctypes.c_double,      # scattering anisotropy g
    ctypes.POINTER(ctypes.c_double),   # UR1
    ctypes.POINTER(ctypes.c_double),   # UT1
    ctypes.POINTER(ctypes.c_double),   # URU
    ctypes.POINTER(ctypes.c_double)    # UTU
    )

def basic_rt_unscattered(n, nslab, ntop, nbot, a, b, g):
    """
    Calculates the unscattered reflection and transmission for a turbid slab.
    
    basic_rt_unscattered(n,nslab,ntop,nbot,a,b,g) returns the unscattered
    light for normal and diffuse incidence [UR1,UT1,URU,UTU] for a slab 
    bounded by glass slides.  The slab is characterized by an albedo a, an
    optical thickness b, a scattering anisotropy g, and an index of refraction nslab.
    The top glass slides have an index of refraction ntop and the bottom slide has
    an index nbot.  If there are no glass slides, set ntop and nbottom to 1.

    n is the number of quadrature angles (more angles give better accuracy but slow
    the calculation by n**3, n must be a multiple of 4, typically 16 is good for forward
    calculations and 4 is appropriate for inverse calculations).  
    
    All parameters must be scalars.
    
    UR1 is the unscattered reflection for normally incident collimated light.
    UT1 is the unscattered transmission for normally incident collimated light.
    URU is the unscattered reflection for diffuse incident light.
    UTU is the unscattered transmission for diffuse incident light.
    """
    ur1 = ctypes.c_double()
    ut1 = ctypes.c_double()
    uru = ctypes.c_double()
    utu = ctypes.c_double()
    libiad.ez_RT_unscattered(n, nslab, ntop, nbot, a, b, g, ur1, ut1, uru, utu)
    return ur1.value, ut1.value, uru.value, utu.value

libiad.ez_RT_Cone.argtypes = (
    ctypes.c_int,         # n quadrature points
    ctypes.c_double,      # slab index of refraction
    ctypes.c_double,      # top slide index of refraction
    ctypes.c_double,      # bottom slide index of refraction
    ctypes.c_double,      # albedo mus/(mus+mua)
    ctypes.c_double,      # optical thickness d*(mua+mus)
    ctypes.c_double,      # scattering anisotropy g
    ctypes.c_double,      # cosine of cone angle
    ctypes.POINTER(ctypes.c_double),   # UR1
    ctypes.POINTER(ctypes.c_double),   # UT1
    ctypes.POINTER(ctypes.c_double),   # URU
    ctypes.POINTER(ctypes.c_double)    # UTU
    )

def basic_rt_cone(n, nslab, ntop, nbot, a, b, g, cos_cone_angle):
    """
    Calculates reflection and transmission for a turbid slab exiting within a cone.
    
    basic_rt_cone(n,nslab,ntop,nbot,a,b,g,cos_cone_angle) assumes normally
    incident or uniformly diffuse incident light and returns the total reflected
    or transmitted light that exits within a cone. The cosine of the cone angle 
    is given by cos_cone.  The returned values are [UR1,UT1,URU,UTU].  
    
    The slab is characterized by an albedo a, an
    optical thickness b, a scattering anisotropy g, and an index of refraction nslab.
    The top glass slides have an index of refraction ntop and the bottom slide has
    an index nbot.  If there are no glass slides, set ntop and nbottom to 1.

    n is the number of quadrature angles (more angles give better accuracy but slow
    the calculation by n**3, n must be a multiple of 4, typically 16 is good for forward
    calculations and 4 is appropriate for inverse calculations).  
    
    All parameters must be scalars.
    
    UR1 is the total reflection within a cone for normally incident collimated light.
    UT1 is the total transmission within a cone for normally incident collimated light.
    URU is the total reflection within a cone for diffuse incident light.
    UTU is the total transmission within a cone for diffuse incident light.
    """
    ur1 = ctypes.c_double()
    ut1 = ctypes.c_double()
    uru = ctypes.c_double()
    utu = ctypes.c_double()
    libiad.ez_RT_Cone(n, nslab, ntop, nbot, a, b, g, cos_cone_angle, ur1, ut1, uru, utu)
    return ur1.value, ut1.value, uru.value, utu.value

libiad.ez_RT_Oblique.argtypes = (
    ctypes.c_int,         # n quadrature points
    ctypes.c_double,      # slab index of refraction
    ctypes.c_double,      # top slide index of refraction
    ctypes.c_double,      # bottom slide index of refraction
    ctypes.c_double,      # albedo mus/(mus+mua)
    ctypes.c_double,      # optical thickness d*(mua+mus)
    ctypes.c_double,      # scattering anisotropy g
    ctypes.c_double,      # cosine of oblique angle
    ctypes.POINTER(ctypes.c_double),   # UR1
    ctypes.POINTER(ctypes.c_double),   # UT1
    ctypes.POINTER(ctypes.c_double),   # URU
    ctypes.POINTER(ctypes.c_double)    # UTU
    )

def basic_rt_oblique(n, nslab, ntop, nbot, a, b, g, cos_oblique):
    """
    Calculates reflection and transmission for light incident at a oblique angle.

    basic_rt_oblique(n,nslab,ntop,nbot,a,b,g,cos_oblique) returns the total R and T
    for light incident at an oblique angle or incident withn a cone.  The cosine 
    of the oblique angle (or the cone) is cos_oblique. The returned values are
    [URx,UTx,URU,UTU].
    
    The slab is characterized by an albedo a, an optical thickness b, a scattering 
    anisotropy g, and an index of refraction nslab. The top glass slide have an 
    index of refraction of ntop and the bottom slide has an index nbot.  If there 
    are no glass slides, set ntop and nbottom to 1.
    
    n is the number of quadrature angles (more angles give better accuracy but slow
    the calculation by n**3, n must be a multiple of 4, typically 16 is good for forward
    calculations and 4 is appropriate for inverse calculations).  
    
    All parameters must be scalars.
    
    URx is the total reflection for obliquely incident collimated light.
    UTx is the total transmission for obliquely incident collimated light.
    URU is the total reflection for diffuse light incident within a cone.
    UTU is the total transmission for diffuse light incident within a cone.
    """
    ur1 = ctypes.c_double()
    ut1 = ctypes.c_double()
    uru = ctypes.c_double()
    utu = ctypes.c_double()
    libiad.ez_RT_Oblique(n, nslab, ntop, nbot, a, b, g, cos_oblique, ur1, ut1, uru, utu)
    return ur1.value, ut1.value, uru.value, utu.value

libiad.ez_Inverse_RT.argtypes = (
    ctypes.c_double,      # slab index of refraction
    ctypes.c_double,      # slide index of refraction
    ctypes.c_double,      # UR1
    ctypes.c_double,      # UT1
    ctypes.c_double,      # unscattered transmission
    ctypes.POINTER(ctypes.c_double),   # a
    ctypes.POINTER(ctypes.c_double),   # b
    ctypes.POINTER(ctypes.c_double),   # g
    ctypes.POINTER(ctypes.c_int)       # error
    )

def basic_rt_inverse(nslab, nslide, ur1, ut1, tc):
    """
    Calculates optical properties given reflection and transmission values.
    
    basic_inverse_rt(nslab, nslide, ur1, ut1, tc) finds [a,b,g] for a slab
    with total reflectance ur1, total transmission ut1, unscattered transmission Tc. 
    The index of refraction of the slab is nslab, the index of refraction of the 
    top and bottom slides is nslide.
    
    All parameters must be scalars
    """
    a = ctypes.c_double()
    b = ctypes.c_double()
    g = ctypes.c_double()
    error = ctypes.c_int()
    libiad.ez_Inverse_RT(nslab, nslide, ur1, ut1, tc, a, b, g, error)
    return a.value, b.value, g.value, error.value

def rt(nslab, nslide, a, b, g):
    """
    Calculates the total reflection and transmission for a turbid slab.

    rt(nslab,ntop,nbot,a,b,g) returns [UR1,UT1,URU,UTU] for a slab optionally
    bounded by glass slides.  The slab is characterized by an albedo a, an
    optical thickness b, a scattering anisotropy g, and an index of refraction nslab.
    The top glass slides have an index of refraction ntop and the bottom slide has
    an index nbot.  If there are no glass slides, set ntop and nbottom to 1.
    
    a, b, and g can be scalars or arrays.  
    
    UR1 is the total reflection for normally incident collimated light.
    UT1 is the total reflection for normally incident collimated light.
    URU is the total reflection for diffuse incident light.
    UTU is the total reflection for diffuse incident light.
    """
    N_QUADRATURE = 16  #should be a multiple of 16
    
    try :
        len_a = len(a)
    except :
        len_a = 0
        aa = a
        
    try :
        len_b = len(b)
    except :
        len_b = 0
        bb = b

    try :
        len_g = len(g)
    except :
        len_g = 0
        gg = g

    thelen = max(len_a,len_b,len_g)

    if thelen==0 :
        return basic_rt(N_QUADRATURE,nslab,nslide,nslide,aa,bb,gg)
     
    if len_a and len_b and len_a!=len_b :
        raise RuntimeError('rt: a and b arrays must be same length')
        
    if len_a and len_g and len_a!=len_g :
        raise RuntimeError('rt: a and g arrays must be same length')
    
    if len_b and len_g and len_b!=len_g :
        raise RuntimeError('rt: b and g arrays must be same length')
        
    ur1  = np.empty(thelen)
    ut1  = np.empty(thelen)
    uru  = np.empty(thelen)
    utu  = np.empty(thelen)

    for i in range(thelen):
        if len_a>0 :
            aa = a[i]

        if len_b>0 :
            bb = b[i]
        
        if len_g>0 :
            gg = g[i]

        ur1[i], ut1[i], uru[i], utu[i] = basic_rt(N_QUADRATURE,nslab,nslide,nslide,aa,bb,gg)

    return ur1, ut1, uru, utu

def rt_unscattered(nslab, nslide, a, b, g):
    """
    Calculates the unscattered reflection and transmission for a turbid slab.
    
    rt_unscattered(nslab,nslide,a,b,g) returns the unscattered portion of
    light for normal and diffuse incidence [UR1,UT1,URU,UTU] for a slab 
    bounded by glass slides.  The slab is characterized by an albedo a, an
    optical thickness b, a scattering anisotropy g, and an index of refraction nslab.
    The top and bottom glass slides have index nslide. If there are no glass slides, 
    set ntop and nbottom to 1.

    a,b,g may be scalars or arrays.
    
    UR1 is the unscattered reflection for normally incident collimated light
    UT1 is the unscattered reflection for normally incident collimated light
    URU is the unscattered reflection for diffuse incident light
    UTU is the unscattered reflection for diffuse incident light
    """
    N_QUADRATURE = 16  #should be a multiple of 16
    
    try :
        len_a = len(a)
    except :
        len_a = 0
        aa = a
        
    try :
        len_b = len(b)
    except :
        len_b = 0
        bb = b

    try :
        len_g = len(g)
    except :
        len_g = 0
        gg = g

    thelen = max(len_a,len_b,len_g)

    if thelen==0 :
        return basic_rt_unscattered(N_QUADRATURE,nslab,nslide,nslide,aa,bb,gg)
     
    if len_a and len_b and len_a!=len_b :
        raise RuntimeError('rt_unscattered: a and b arrays must be same length')
        
    if len_a and len_g and len_a!=len_g :
        raise RuntimeError('rt_unscattered: a and g arrays must be same length')
    
    if len_b and len_g and len_b!=len_g :
        raise RuntimeError('rt_unscattered: b and g arrays must be same length')
        
    ur1  = np.empty(thelen)
    ut1  = np.empty(thelen)
    uru  = np.empty(thelen)
    utu  = np.empty(thelen)

    for i in range(thelen):
        if len_a>0 :
            aa = a[i]

        if len_b>0 :
            bb = b[i]
        
        if len_g>0 :
            gg = g[i]

        ur1[i], ut1[i], uru[i], utu[i] = basic_rt_unscattered(N_QUADRATURE,nslab,nslide,nslide,aa,bb,gg)

    return ur1, ut1, uru, utu

def rt_cone(nslab, nslide, a, b, g, cos_cone):
    """
    Calculates reflection and transmission for a turbid slab exiting within a cone.
    
    rt_cone(nslab,nslide,a,b,g,cos_cone) assumes normally
    incident or uniformly diffuse incident light and returns the total reflected
    or transmitted light that exits within a cone. The cosine of the cone angle 
    is cos_cone.  The returned values are [UR1,UT1,URU,UTU].  
    
    The slab is characterized by an albedo a, an
    optical thickness b, a scattering anisotropy g, and an index of refraction 
    nslab. The top and bottom glass slides have an index of refraction nslide. 
    If there are no glass slides, set nslide to 1.
    
    a, b, g, and cos_cone may be scalars or arrays.
    
    UR1 is the total reflection within a cone for normally incident collimated light.
    UT1 is the total transmission within a cone for normally incident collimated light.
    URU is the total reflection within a cone for diffuse incident light.
    UTU is the total transmission within a cone for diffuse incident light.
    """
    N_QUADRATURE = 16  #should be a multiple of 16
    
    try :
        len_a = len(a)
    except :
        len_a = 0
        aa = a
        
    try :
        len_b = len(b)
    except :
        len_b = 0
        bb = b

    try :
        len_g = len(g)
    except :
        len_g = 0
        gg = g

    try :
        len_mu = len(cos_cone)
    except :
        len_mu = 0
        mu = cos_cone

    thelen = max(len_a,len_b,len_g,len_mu)

    if thelen==0 :
        return basic_rt_cone(N_QUADRATURE,nslab,nslide,nslide,aa,bb,gg,mu)
     
    if len_a and len_b and len_a!=len_b :
        raise RuntimeError('rt_cone: a and b arrays must be same length')
        
    if len_a and len_g and len_a!=len_g :
        raise RuntimeError('rt_cone: a and g arrays must be same length')
    
    if len_a and len_mu and len_a!=len_mu :
        raise RuntimeError('rt_cone: a and mu arrays must be same length')
    
    if len_b and len_g and len_b!=len_g :
        raise RuntimeError('rt_cone: b and g arrays must be same length')

    if len_b and len_mu and len_b!=len_mu :
        raise RuntimeError('rt_cone: b and mu arrays must be same length')

    if len_g and len_mu and len_g!=len_mu :
        raise RuntimeError('rt_cone: g and mu arrays must be same length')

    ur1  = np.empty(thelen)
    ut1  = np.empty(thelen)
    uru  = np.empty(thelen)
    utu  = np.empty(thelen)

    for i in range(thelen):
        if len_a>0 :
            aa = a[i]

        if len_b>0 :
            bb = b[i]
        
        if len_g>0 :
            gg = g[i]

        if len_mu>0 :
            mu = cos_cone[i]


        ur1[i], ut1[i], uru[i], utu[i] = basic_rt_cone(N_QUADRATURE,nslab,nslide,nslide,aa,bb,gg,mu)

    return ur1, ut1, uru, utu

def rt_oblique(nslab, nslide, a, b, g, cos_oblique):
    """
    Calculates reflection and transmission for light incident at a oblique angle.

    rt_oblique(n,nslide,a,b,g,cos_oblique) returns the total R and T
    for light incident at an oblique angle or incident withn a cone.  The cosine 
    of the oblique angle (or the cone) is cos_oblique. The returned values are
    [URx,UTx,URU,UTU].
    
    The slab is characterized by an albedo a, an optical thickness b, a scattering 
    anisotropy g, and an index of refraction nslab. The top and bottom glass slides 
    have an index of refraction nslide. If there are no glass slides, set nslide to 1.
    
    a, b, g, and cos_oblique may be scalars or arrays.
    
    URx is the total reflection for obliquely incident collimated light.
    UTx is the total transmission for obliquely incident collimated light.
    URU is the total reflection for diffuse light incident within a cone.
    UTU is the total transmission for diffuse light incident within a cone.
    """
    N_QUADRATURE = 16  #should be a multiple of 16
    
    try :
        len_a = len(a)
    except :
        len_a = 0
        aa = a
        
    try :
        len_b = len(b)
    except :
        len_b = 0
        bb = b

    try :
        len_g = len(g)
    except :
        len_g = 0
        gg = g

    try :
        len_mu = len(cos_oblique)
    except :
        len_mu = 0
        mu = cos_oblique

    thelen = max(len_a,len_b,len_g,len_mu)

    if thelen==0 :
        return basic_rt_oblique(N_QUADRATURE,nslab,nslide,nslide,aa,bb,gg,mu)
     
    if len_a and len_b and len_a!=len_b :
        raise RuntimeError('rt_oblique: a and b arrays must be same length')
        
    if len_a and len_g and len_a!=len_g :
        raise RuntimeError('rt_oblique: a and g arrays must be same length')
    
    if len_a and len_mu and len_a!=len_mu :
        raise RuntimeError('rt_oblique: a and mu arrays must be same length')
    
    if len_b and len_g and len_b!=len_g :
        raise RuntimeError('rt_oblique: b and g arrays must be same length')

    if len_b and len_mu and len_b!=len_mu :
        raise RuntimeError('rt_oblique: b and mu arrays must be same length')

    if len_g and len_mu and len_g!=len_mu :
        raise RuntimeError('rt_oblique: g and mu arrays must be same length')

    ur1  = np.empty(thelen)
    ut1  = np.empty(thelen)
    uru  = np.empty(thelen)
    utu  = np.empty(thelen)

    for i in range(thelen):
        if len_a>0 :
            aa = a[i]

        if len_b>0 :
            bb = b[i]
        
        if len_g>0 :
            gg = g[i]

        if len_mu>0 :
            mu = cos_oblique[i]

        ur1[i], ut1[i], uru[i], utu[i] = basic_rt_oblique(N_QUADRATURE,nslab,nslide,nslide,aa,bb,gg,mu)

    return ur1, ut1, uru, utu

def rt_inverse(nslab, nslide, ur1, ut1, t_unscattered):
    """
    rt_inverse(nslab, nslide, ur1, ut1, t_unscattered) calculates [a,b,g] for a slab
    with total reflectance ur1, total transmission ut1, unscattered transmission t_unscattered. 
    The index of refraction of the slab is nslab, the index of refraction of the 
    top and bottom slides is nslide.
    
    ur1, ut1, and t_unscattered may be scalars or arrays.
    """
    try :
        len_r1 = len(ur1)
    except :
        len_r1 = 0
        r1 = len_ur1
        
    try :
        len_t1 = len(ut1)
    except :
        len_t1 = 0
        t1 = len_ut1

    try :
        len_tc = len(t_unscattered)
    except :
        len_tc = 0
        tc = t_unscattered

    thelen = max(len_r1,len_t1,len_tc)

    if thelen==0 :
        return basic_inverse_rt(nslab,nslide,ur1,ut1,tc)
     
    if len_r1 and len_t1 and len_r1!=len_t1 :
        raise RuntimeError('inverse_rt: ur1 and ut1 arrays must be same length')
        
    if len_r1 and len_tc and len_r1!=len_tc :
        raise RuntimeError('inverse_rt: ur1 and tc arrays must be same length')

    if len_t1 and len_tc and len_t1!=len_tc :
        raise RuntimeError('inverse_rt: t1 and tc arrays must be same length')

    a  = np.empty(thelen)
    b  = np.empty(thelen)
    g  = np.empty(thelen)
    error  = np.empty(thelen)

    for i in range(thelen):
        if len_r1>0 :
            r1 = ur1[i]

        if len_t1>0 :
            t1 = ut1[i]
        
        if len_tc>0 :
            tc = t_unscattered[i]

        a[i], b[i], g[i], error[i] = basic_inverse_rt(nslab,nslide,ur1,ut1,tc)

    return a, b, g, error