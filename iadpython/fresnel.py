# pylint: disable = invalid-name
# pylint: disable = arguments-out-of-order
# pylint: disable = too-many-arguments
# pylint: disable = too-many-locals

"""
Module for generating boundary matrices.

Two types of starting methods are possible.

    import iadpython.fresnel

    n = 4
    slab = iadpython.start.Slab(a = 0.9, b = 10, g = 0.9, n = 1.5)
    method = iadpython.start.Method(slab)
    r, t = iad.start.init_layer(slab, method)
    print(r)
    print(t)

"""
import sys
import numpy as np

__all__ = ('cos_critical_angle',
           'cos_snell',
           'absorbing_glass_RT',
           'specular_nu_RT',
           'specular_nu_RT_flip',
           'diffuse_glass_R',
           'glass',
           )

def cos_critical_angle(ni, nt):
    """
    @*2 The critical angle.

     'cos_critical_Angle' calculates the cosine of the critical angle.
    If there is no critical angle then 0.0 is returned (i.e.,  cos( pi/2)).
    Note that no trigonmetric functions are required.
    Recalling snell's law

    n_i  sin theta_i = n_t sin theta_t

    To find the critical angle, let  theta_t = pi/2 and then

     theta_c = sin^-1 n_t / n_i

    The cosine of this angle is then

     cos theta_c = cos  left( sin^-1 n_t / n_i right) = sqrtn_i^2-n_t^2 / n_i

    or more simply

     cos theta_c = sqrt1-n^2

    where n = n_t/n_i.
    """

    if nt >= ni:
        return 0.0

    return np.sqrt(1.0 - (nt/ni)**2)



def cos_snell(n_i, nu_i, n_t):
    """
    Snell's Law.

    'cos_snell' returns the cosine of the angle that the light propagates through
    a medium given the cosine of the angle of incidence and the indices of refraction.
    Let the cosine of the angle of incidence be  nu_t, the transmitted cosine as  nu_t,
    the index of refraction of the incident material n_i and that of the transmitted material
    be n_t.

    Snell's law states

    n_i  sin theta_i = n_t  sin theta_t

    but if the angles are expressed as cosines,  nu_i = cos theta_i then

    n_i  sin( cos^-1 nu_i) = n_t  sin( cos^-1 nu_t)

    Solving for  nu_t yields

    nu_t = cos  sin^-1[(n_i/n_t) sin( cos^-1 nu_i)]

    which is pretty ugly.  However, note that  sin( cos^-1 nu) = sqrt1- nu^2
    and the above becomes

    nu_t = sqrt1-(n_i/n_t)^2 (1- nu_i^2)

    and no trigonmetric calls are necessary.  Hooray!

    A few final notes.  I check to make sure that the index of refraction of
    changes before calculating a bunch of stuff.  This routine should
    not be passed incident angles greater
    than the critical angle, but I shall program defensively and test
    to make sure that the argument of the 'sqrt' function is non-negative.
    If it is, then I return  nu_t = 0 i.e.,  theta_t = 90^ circ.

    I also pretest for the common but trivial case of normal incidence.
    """
    if nu_i == 1.0:
        return 1.0

    if n_i == n_t:
        return nu_i

    temp = n_i/n_t
    temp = 1.0-temp*temp*(1.0 - nu_i*nu_i)
    if temp < 0:
        return 0.0

    return np.sqrt(temp)



def fresnel_reflection(n_i, n_t, nu_i):
    """
    Fresnel Reflection.

    'Fresnel' calculates the specular reflection for light incident at
    an angle  theta_i from the normal (having a cosine equal to  nu_i)
    in a medium with index of
    refraction 'n_i' onto a medium with index of refraction 'n_t' .

    The usual way to calculate the total reflection for unpolarized light is
    to use the Fresnel formula

    R = 1 / 2 left[  sin^2( theta_i- theta_t) /  sin^2( theta_i+ theta_t)
                         + tan^2( theta_i- theta_t) /  tan^2( theta_i+ theta_t)  right]

    where  theta_i and  theta_t represent the angle (from normal) that light is incident
    and the angle at which light is transmitted.
    There are several problems with calculating the reflection using this formula.
    First, if the angle of incidence is zero, then the formula results in division by zero.
    Furthermore, if the angle of incidence is near zero, then the formula is the ratio
    of two small numbers and the results can be inaccurate.
    Second, if the angle of incidence exceeds the critical angle, then the calculation of
     theta_t results in an attempt to find the arcsine of a quantity greater than
    one.  Third, all calculations in this program are based on the cosine of the angle.
    This routine forces the calling routine to find  theta_i = cos^-1  nu.
    Fourth, the routine also gives problems when the critical angle is exceeded.

    Closer inspection reveals that this is the wrong formulation to use.  The formulas that
    should be used for parallel and perpendicular polarization are

    R_ parallel = left[n_t cos theta_i-n_i cos theta_t /
                                            n_t cos theta_i+n_i cos theta_t right]^2,
     qquad qquad
    R_ perp = left[ n_i cos theta_i-n_t cos theta_t /
                                            n_i cos theta_i+n_t cos theta_t right]^2.

    The formula for unpolarized light, written in terms of  nu_i = cos theta_i and
     nu_t = cos theta_t is

    R = 1 / 2 left[n_t nu_i-n_i nu_t / n_t nu_i+n_i nu_t right]^2
    +1 / 2 left[n_i nu_i-n_t nu_t / n_i nu_i+n_t nu_t right]^2


    This formula has the advantage that no trig routines need to be called and that the
    case of normal irradiance does not cause division by zero.  Near normal incidence
    remains numerically well-conditioned.  In the routine below, I test for matched
    boundaries and normal incidence to eliminate unnecessary calculations.  I also
    test for total internal reflection to avoid possible division by zero.  I also
    find the ratio of the indices of refraction to avoid an extra multiplication and
    several intermediate variables.
    """
    if n_i == n_t:
        return 0.0

    if nu_i == 1.0:
        return ((n_i-n_t)/(n_i+n_t))**2

    if nu_i == 0.0:
        return 1.0

    nu_t = cos_snell(n_i, nu_i, n_t)
    if nu_t == 0.0:
        return 1.0

    ratio = n_i/n_t
    temp = ratio*nu_t
    temp1 = (nu_i-temp)/(nu_i+temp)
    temp = ratio*nu_i
    temp = (nu_t-temp)/(nu_t+temp)
    return (temp1**2+temp**2)/2



def glass(n_i, n_g, n_t, nu_i):
    """
    Reflection from a glass slide.

    'glass' calculates the total specular reflection (i.e., including
    multiple internal reflections) based on
    the indices of refraction of the incident medium 'n_i', the glass 'n_g',
    and medium into which the light is transmitted 'n_t' for light incident at
    an angle from the normal having cosine 'nu_i'.

    In many tissue optics problems, the sample is constrained by a piece of glass
    creating an air-glass-tissue sequence.
    The adding-doubling formalism can calculate the effect that the layer of glass will
    have on the radiative transport properties by including a layer for the glass-tissue
    interface and a layer for the air-glass interface.  However, it is simpler to find net
    effect of the glass slide and include only one layer for the glass boundary.

    The first time I implemented this routine, I did not include multiple internal
    reflections.  After running test cases, it soon became apparent that  the
    percentage errors were way too
    big for media with little absorption and scattering.  It is not hard to find the
    result for the reflection from a non-absorbing glass layer (equation A2.21
    in my dissertation) in which multiple reflections are properly accounted for

    r_g = r_1 + r_2 - 2  r_1  r_2  / 1 - r_1  r_2

    Here r_1 is the reflection at the air-glass interface and r_2 is the
    reflection at the glass-sample interface.

    There is one pitfall in calculating r_g.  When the angle
    of incidence exceeds the critical angle then the formula above causes
    division by zero.  If this is the case then r_1 = 1 and can easily
    be tested for.

    To eliminate unnecessary computation, I check to make sure that
    it really is necessary to call the 'Fresnel' routine twice.
    It is noteworthy that the formula for r_g works correctly if the
    the first boundary is not totally reflecting but the second one is.
    Note that  nu_g gets calculated twice
    (once in the first call to 'Fresnel' and once directly).
    """
    if n_i == n_g:
        return fresnel_reflection(n_g, n_t, nu_i)

    r1 = fresnel_reflection(n_i, n_g, nu_i)
    if r1 >= 1.0 or n_g == n_t:
        return r1

    nu_g = cos_snell(n_i, nu_i, n_g)
    r2 = fresnel_reflection(n_g, n_t, nu_g)
    temp = r1*r2
    return (r1 + r2 - 2 * temp) / (1 - temp)


def absorbing_glass_RT(n_i, n_g, n_t, nu_i, b):
    """
    Reflection from an absorbing slide.

    'absorbing_glass_RT' calculates the total specular reflection and transmission
    (i.e., including
    multiple internal reflections) based on
    the indices of refraction of the incident medium 'n_i', the glass 'n_g',
    and medium into which the light is transmitted 'n_t' for light incident at
    an angle from the normal having cosine 'nu_i'.  The optical thickness of
    the glass b = nu_a d is measured normal to the glass.

    This routine was generated to help solve a problem with the inverse adding-doubling
    program associated with samples with low absorbances.  A particular situation
    arises when the slides have significant absorption relative to the sample
    absorption.  Anyway, it is not hard to extend the result for non-absorbing slides
    to the absorbing case

    r = r_1 + (1-2r_1)r_2*exp(-2b/nu_g) / [1 - r_1*r_2*exp(-2b/nu_g)]

    Here r_1 is the reflection at the sample-glass interface and r_2 is the
    reflection at the glass-air interface and  nu_g is the cosine of the
    angle inside the glass.  Note that if b ne0 then the reflection depends
    on the order of the indices of refraction, otherwise 'n_i' and 'n_t'
    can be switched and the result should be the same.

    The corresponding result for transmission is

    t = (1-r_1)*(1-r_2)*exp(-b/nu_g) / [1 - r_1*r_2*exp(-2b/nu_g)]

    There are two potential pitfalls in the calculation.  The first is
    when the angle of incidence exceeds the critical angle then the formula above causes
    division by zero.  If this is the case, 'Fresnel' will return r_1 = 1 and
    this routine responds appropriately.  The second case is when the optical
    thickness of the slide is too large.

    I don't worry too much about optimal coding, because this routine does
    not get called all that often and also because 'Fresnel' is pretty good
    at avoiding unnecessary computations.  At worst this routine just has
    a couple of extra function calls and a few extra multiplications.

    I also check to make sure that the exponent is not too small.
    """
    r1 = fresnel_reflection(n_i, n_g, nu_i)
    if r1 >= 1.0 or b == np.inf or nu_i == 0.0:
        return r1, 0

    nu_g = cos_snell(n_i, nu_i, n_g)
    r2 = fresnel_reflection(n_g, n_t, nu_g)

    if b == 0.0:
        r = (r1 + r2 - 2.0 * r1 *r2) / (1- r1 * r2)
        return r, 1-r

    if 2 * (-b/nu_g) <= sys.float_info.min_10_exp * 2.3025851:
        return r1, 0.0

    expo = np.exp(-b/nu_g)
    denom = 1.0-r1*r2*expo**2
    r = (r1 + (1.0 - 2.0 * r1)*r2*expo**2) / denom
    t = (1.0 - r1)*(1.0 - r2)*expo / denom

    return r, t


def specular_nu_RT_flip(flip, n_top, n_slab, n_bottom, tau_top, tau_slab, tau_bottom, nu):
    """
    Unscattered refl and trans for a sample.

    Find the reflectance to incorporate flipping of the sample.  This
    is needed when the sample is flipped between measurements.
    """

    r, t = specular_nu_RT(n_top, n_slab, n_bottom, tau_top, tau_slab, tau_bottom, nu)

    if flip and n_top != n_bottom and tau_top != tau_bottom:
        _, t = specular_nu_RT(n_bottom, n_slab, n_top, tau_bottom, tau_slab, tau_top, nu)

    return r, t


def specular_nu_RT(n_top, n_slab, n_bottom, tau_top, tau_slab, tau_bottom, nu):
    """
    Unscattered reflection and transmission through a glass-slab-glass sandwich.

    Light is incident at
    an angle having a cosine 'nu' from air onto a possibly absorbing glass plate with index 'n_top'
    on a sample with index 'n_slab' resting on another possibly absorbing glass plate with index
    'n_bottom' and then exiting into air again.

    The optical thickness of the slab is 'tau_slab'.
    """
    r_top, t_top = absorbing_glass_RT(1.0, n_top, n_slab, nu, tau_top)
    nu_slab = cos_snell(1.0, nu, n_slab)

    if nu_slab == 0:
        return r_top, 0

    r_bottom, t_bottom = absorbing_glass_RT(n_slab, n_bottom, 1.0, nu_slab, tau_bottom)

    # avoid underflow errors and division by zero.
    if tau_slab == np.inf:
        return r_top, 0

    if 2*(-tau_slab/nu_slab) <= sys.float_info.min_10_exp * 2.3025851:
        return r_top, 0

    # If r_top is the reflection for the top and r_bottom is that for the
    # bottom surface then the total reflection will be
    #
    # r = r_top + r_bottom*t_top^2 * exp(-2*tau/nu)  /
    #      [1 - r_rm top  r_rm bottom  exp(-2 tau/ nu)]
    #
    # and the transmission is
    #
    # t = t_ rm top t_ rm bottom  exp(-tau/ nu) /
    #     [1 - r_rm top  r_rm bottom  exp(-2*tau/nu)]
    #
    # where nu is the angle inside the slab and  tau is the optical
    # thickness of the slab.

    beer = np.exp(-tau_slab/nu_slab)
    temp = t_top * beer
    denom = 1 - r_top * r_bottom * beer * beer
    r = r_top + r_bottom*temp * temp  / denom
    t = t_bottom * temp / denom

    return r, t


def R1(ni, nt):
    """
    Total diffuse reflection.

    Calculate the first moment of the Fresnel reflectance using the analytic
    solution of Walsh.

    The integral of the first moment of the Fresnel reflection (R_1)
    has been found analytically by Walsh, [see Ryde 1931]

    R_1 & = 1 / 2 + (m-1)(3m+1) / 6(m+1)^2
            + left[ m^2(m^2-1)^2 / (m^2+1)^3 right] log( m-1 / m+1) cr
            & qquad- 2m^3 (m^2+2m-1) / (m^2+1)(m^4-1) +
             left[ 8m^4(m^4+1) / (m^2+1)(m^4-1)^2 right] log(m)

    where Walsh's parameter m = n_t/n_i.    This equation is only valid when
    n_i<n_t.  If n_i>n_t then using (see Egan and Hilgeman 1973),

    1-R_1(n_i/n_t) / n_t^2 = 1-R_1(n_t/n_i) / n_i^2

    or

    R(1/m) = 1-m^2[1-R(m)]
    """

    if ni == nt:
        return 0.0

    if ni < nt:
        m = nt/ni
    else:
        m = ni/nt

    m2 = m * m
    m4 = m2 * m2
    mm1 = m - 1
    mp1 = m + 1
    temp = (m2 - 1)/(m2 + 1)

    r = 0.5 + mm1 * (3 * m + 1) / 6 / mp1 / mp1
    r += m2 * temp**2 / (m2 + 1) * np.log(mm1 / mp1)
    r -= 2 * m * m2 * (m2 + 2 * m - 1) / (m2 + 1) / (m4 - 1)
    r += 8 * m4 * (m4 + 1) / (m2 + 1) / (m4-1)/(m4-1) * np.log(m)

    if ni < nt:
        return r

    return 1 - (1 - r) / m2


def diffuse_glass_R(nair, nslide, nslab):
    """
    Diffusion reflection from a glass slide.

    returns the total diffuse specular reflection from the air-glass-tissue
    interface
    """
    r_airglass = R1(nair, nslide)
    r_glasstissue = R1(nslide, nslab)
    rtemp = r_airglass * r_glasstissue
    if rtemp >= 1:
        return 1.0

    return (r_airglass + r_glasstissue - 2 * rtemp) / (1 - rtemp)
