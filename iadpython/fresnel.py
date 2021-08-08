# pylint: disable = invalid-name
# pylint: disable = arguments-out-of-order
# pylint: disable = too-many-arguments
# pylint: disable = too-many-locals
# pylint: disable = consider-using-in

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
import numpy as np
import iadpython

__all__ = ('cos_critical',
           'cos_snell',
           'absorbing_glass_RT',
           'specular_nu_RT',
           'diffuse_glass_R',
           'glass',
           )


def cos_critical(n_i, n_t):
    """
    Calculate the cosine of the critical angle.

    This works for arrays too.  If there is no critical angle then
    cos(pi/2)=0 is returned.

    theta_c = arcsin(n_t / n_i)

    The cosine of this angle is then

    cos(theta_c) = cos(arcsin(n_t / n_i)) = sqrt(1-(n_t/n_i)**2)

    Args:
        n_i: index of refraction of incident medium
        n_t: index of refraction of transmitted medium

    Returns:
        cosine of the critical angle
    """
    temp = 1.0 - (n_t / n_i)**2

    if not np.isscalar(temp):
        np.place(temp, temp < 0, 0)
    elif temp < 0:
        return 0

    return np.sqrt(temp)


def cos_snell(n_i, nu, n_t):
    """
    Return the cosine of the transmitted angle.

    Snell's law states

    n_i * sin(theta_i) = n_t*sin(theta_t)

    but if the angles are expressed as cosines,  nu_i = cos(theta_i) then

    n_i*sin(arccos(nu_i)) = n_t*sin(arccos(nu_t))

    Solving for nu_t yields

    nu_t = cos(arcsin[(n_i/n_t) sin(arccos(nu_i))]

    which is pretty ugly.  However, note that sin(arccos(nu)) = sqrt(1-nu**2)
    and the above becomes

    nu_t = sqrt(1-(n_i/n_t)**2 (1- nu_i**2))

    Args:
        n_i: index of refraction of incident medium
        n_t: index of refraction of transmitted medium

    Returns:
        cosine of transmitted angle
    """
    temp = 1.0 - (n_i / n_t)**2 * (1.0 - nu**2)

    if not np.isscalar(temp):
        np.place(temp, temp < 0, 0)
    elif temp < 0:
        return 0

    return np.sqrt(temp)


def reflection(n_i, nu_i, n_t):
    """
    Fresnel Reflection.

    'Fresnel' calculates the specular reflection for light incident at
    an angle  theta_i from the normal (having a cosine equal to  nu_i)
    in a medium with index of
    refraction 'n_i' onto a medium with index of refraction 'n_t' .

    The usual way to calculate the total reflection for unpolarized light is
    to use the Fresnel formula

    R = 1 / 2 left[  sin**2( theta_i- theta_t) /  sin**2( theta_i+ theta_t)
                         + tan**2( theta_i- theta_t) /  tan**2( theta_i+ theta_t)  right]

    where  theta_i and  theta_t represent the angle (from normal) that light is incident
    and the angle at which light is transmitted.
    There are several problems with calculating the reflection using this formula.
    First, if the angle of incidence is zero, then the formula results in division by zero.
    Furthermore, if the angle of incidence is near zero, then the formula is the ratio
    of two small numbers and the results can be inaccurate.
    Second, if the angle of incidence exceeds the critical angle, then the calculation of
    theta_t results in an attempt to find the arcsine of a quantity greater than
    one.  Third, all calculations in this program are based on the cosine of the angle.
    This routine forces the calling routine to find  theta_i = cos**-1  nu.
    Fourth, the routine also gives problems when the critical angle is exceeded.

    Closer inspection reveals that this is the wrong formulation to use.  The formulas that
    should be used for parallel and perpendicular polarization are

    R_ parallel = left[n_t cos theta_i-n_i cos theta_t /
                                            n_t cos theta_i+n_i cos theta_t right]**2,

    R_ perp = left[ n_i cos theta_i-n_t cos theta_t /
                                            n_i cos theta_i+n_t cos theta_t right]**2.

    The formula for unpolarized light, written in terms of  nu_i = cos theta_i and
     nu_t = cos theta_t is

    R = 1 / 2 (n_t nu_i-n_i nu_t)**2/(n_t nu_i+n_i nu_t)**2
    +1 / 2 left[n_i nu_i-n_t nu_t / n_i nu_i+n_t nu_t right]**2


    This formula has the advantage that no trig routines need to be called and that the
    case of normal irradiance does not cause division by zero.  Near normal incidence
    remains numerically well-conditioned.  In the routine below, I test for matched
    boundaries and normal incidence to eliminate unnecessary calculations.  I also
    test for total internal reflection to avoid possible division by zero.  I also
    find the ratio of the indices of refraction to avoid an extra multiplication and
    several intermediate variables.

    The tricky things about this implementation are to get handle angles above
    the critical angle properly and trying to keep everything working when
    arrays are passed.  If n_i==n_t then we want to return zero, otherwise
    we want to return 1.
    """
    if n_i == n_t:
        if np.isscalar(nu_i):
            return 0
        return np.zeros_like(nu_i)

    nu_t = cos_snell(n_i, nu_i, n_t)

    sum1 = (n_t * nu_i + n_i * nu_t) ** 2
    dif1 = (n_t * nu_i - n_i * nu_t) ** 2
    sum2 = (n_i * nu_i + n_t * nu_t) ** 2
    dif2 = (n_i * nu_i - n_t * nu_t) ** 2

    if np.isscalar(sum1):
        if nu_i == 0:       # angle is greater than critical angle
            return 1
        return (dif1 / sum1 + dif2 / sum2) / 2

    # when dif1==sum1==0, then ratio should be one
    out1 = np.divide(dif1, sum1, out=np.ones_like(dif1), where=dif1 != sum1)
    out2 = np.divide(dif2, sum2, out=np.ones_like(dif2), where=dif2 != sum2)
    return (out1 + out2) / 2


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
    if n_i == n_g or n_g == n_t:
        return reflection(n_i, nu_i, n_t)

    r1 = reflection(n_i, nu_i, n_g)
    nu_g = cos_snell(n_i, nu_i, n_g)
    r2 = reflection(n_g, nu_g, n_t)
    denom = 1 - r1 * r2
    numer = r1 + r2 - 2 * r1 * r2

    if np.isscalar(denom):
        if numer == denom:
            return 1
        return numer / denom

    return np.divide(numer, denom, out=np.ones_like(numer), where=numer != denom)


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
    if b == 0:
        r = glass(n_i, n_g, n_t, nu_i)
        return r, 1 - r

    r1 = reflection(n_i, nu_i, n_g)
    nu_g = cos_snell(n_i, nu_i, n_g)

    # too thick for any light to make it through the sample
    if b > iadpython.AD_MAX_THICKNESS:
        return r1, np.zeros_like(r1)

    r2 = reflection(n_g, nu_g, n_t)

    expo = np.exp(-b / nu_g)
    denom = 1.0 - r1 * r2 * expo**2
    r = (r1 + (1.0 - 2.0 * r1) * r2 * expo**2) / denom
    t = (1.0 - r1) * (1.0 - r2) * expo / denom

    return r, t


def _specular_nu_RT(n_top, n_slab, n_bottom, b_slab, nu, b_top=0, b_bottom=0):
    """
    Unscattered reflection and transmission through a glass-slab-glass sandwich.

    Light is incident at nu=cos(theta) from air onto a absorbing glass
    plate onto a slab resting on another absorbing glass plate before
    exiting into air again.

    If r_top and r_bottom are the reflectances for the top and bottom then

    r = r_top + r_bottom*t_top**2*expo**2 / [1-r_top*r_bottom*expo**2]

    and the transmission is

    t = t_top*t_bottom*expo / [1-r_top*r_bottom*expo**2]

    where expo = exp(-b_slab/nu)

    Args:
        n_top: index of glass slide on top
        n_slab: index of the slab
        n_bottom: index of glass on bottom
        b_top: optical thickness of top slide
        b_slab: optical thickness of the slab
        b_bottom: optical thickness of the bottom slide
        nu: cosine of angle(s) in slab
    Returns
        r, t: unscattered reflectance(s) and transmission(s)
    """
    # simplest case of no glass slides
    if (n_top == 1 and n_bottom == 1) or (n_top == n_slab and n_bottom == n_slab):
        return absorbing_glass_RT(1, n_slab, 1, nu, b_slab)

    # backwards because nu is measured in the slab
    r_top, t_top = absorbing_glass_RT(n_slab, n_top, 1.0, nu, b_top)

    # avoid underflow errors and division by zero.
    if b_slab > iadpython.AD_MAX_THICKNESS:
        return r_top, 0

    r_bottom, t_bottom = absorbing_glass_RT(n_slab, n_bottom, 1.0, nu, b_bottom)

    # if b==0, no attenuation.
    if np.isscalar(nu):
        if b_slab == 0:
            expo = 1
        elif nu == 0:
            expo = 0
        else:
            expo = np.exp(-b_slab / nu)
    else:
        if b_slab == 0:
            expo = np.ones_like(nu)
        else:
            ratio = nu / b_slab
            np.place(ratio, ratio == 0, 0.01)
            expo = np.exp(-1 / ratio)

    denom = 1 - r_top * r_bottom * expo**2
    numer = r_bottom * t_top**2 * expo**2

    if np.isscalar(denom):
        denom = 1
    else:
        np.place(denom, denom == 0, 1)

    r = r_top + numer / denom
    t = t_bottom * t_top * expo / denom
    return r, t


def specular_nu_RT(n_top, n_slab, n_bottom, b_slab, nu, b_top=0, b_bottom=0, flip=False):
    """
    Unscattered refl and trans for a sample.

    Find the reflectance to incorporate flipping of the sample.  This
    is needed when the sample is flipped between measurements.
    """
    r, t = _specular_nu_RT(n_top, n_slab, n_bottom, b_slab, nu, b_top, b_bottom)

    if not flip:
        return r, t

    if n_top == n_bottom and b_top == b_bottom:
        return r, t

    _, t = _specular_nu_RT(n_bottom, n_slab, n_top, b_slab, nu, b_bottom, b_top)

    return r, t


def R1(ni, nt):
    """
    Total diffuse reflection.

    Calculate the first moment of the Fresnel reflectance using the analytic
    solution of Walsh.

    The integral of the first moment of the Fresnel reflection (R_1)
    has been found analytically by Walsh, [see Ryde 1931]

    R_1 & = 1 / 2 + (m-1)(3m+1) / 6(m+1)**2
            + [ m**2(m**2-1)**2 / (m**2+1)**3 ] log( m-1 / m+1) cr
            - 2m**3 (m**2+2m-1) / (m**2+1)(m**4-1) +
            + [ 8m**4(m**4+1) / (m**2+1)(m**4-1)**2 ] log(m)

    where Walsh's parameter m = n_t/n_i.    This equation is only valid when
    n_i<n_t.  If n_i>n_t then using (see Egan and Hilgeman 1973),

    1-R_1(n_i/n_t) / n_t**2 = 1-R_1(n_t/n_i) / n_i**2

    or

    R(1/m) = 1-m**2[1-R(m)]
    """
    if ni == nt:
        return 0.0

    if ni < nt:
        m = nt / ni
    else:
        m = ni / nt

    m2 = m * m
    m4 = m2 * m2
    mm1 = m - 1
    mp1 = m + 1
    temp = (m2 - 1) / (m2 + 1)

    r = 0.5 + mm1 * (3 * m + 1) / 6 / mp1 / mp1
    r += m2 * temp**2 / (m2 + 1) * np.log(mm1 / mp1)
    r -= 2 * m * m2 * (m2 + 2 * m - 1) / (m2 + 1) / (m4 - 1)
    r += 8 * m4 * (m4 + 1) / (m2 + 1) / (m4 - 1) / (m4 - 1) * np.log(m)

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
