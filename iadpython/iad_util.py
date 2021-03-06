"""IAD Utilities."""

BIG_A_VALUE = 999999.0
SMALL_A_VALUE = 0.000001

def What_Is_B(slab, Tc):
    """
    Finding optical thickness.

    This routine figures out what the optical thickness of a slab
    based on the index of refraction of the slab and the amount
    of collimated light that gets through it.

    It should be pointed out right here in the front that this
    routine does not work for diffuse irradiance, but then the whole
    concept of estimating the optical depth for diffuse irradiance
    is bogus anyway. 
    """

    # The first thing to do is to find the specular reflection for light
    # interacting with the top and bottom air-glass-sample interfaces.  I make
    # a simple check to ensure that the the indices are different before 
    # calculating the bottom reflection.  Most of the time the |r1==r2|,
    # but there are always those annoying special cases.

    r1, t1 = Absorbing_Glass_RT(1.0, slab.n_top_slide, slab.n_slab, 
                       slab.cos_angle, slab.b_top_slide)

    mu_in_slab = Cos_Snell(1.0, slab.cos_angle, slab.n_slab)
    
    r2, t2 = Absorbing_Glass_RT(slab.n_slab, slab.n_bottom_slide, 1.0, 
                       mu_in_slab, slab.b_bottom_slide)

    # Bad values for the unscattered transmission are those that 
    # are non-positive, those greater than one, and those greater than
    # are possible in a non-absorbing medium.
    
    # Since this routine has no way to report errors, I just set the 
    # optical thickness to the natural values in these cases.

    if Tc <= 0:
        return HUGE_VAL
        
    if Tc >= t1 * t2 / (1 - r1 * r2):
        return (0.001)

    # If either |r1| or |r2==0| then things are very simple
    # because the sample does not sustain multiple internal reflections
    # and the unscattered transmission is
    # $$
    # T_c = t_1 t_2 \exp(-b/\nu)
    # $$
    # where |b| is the optical thickness and $\nu$ is |slab.cos_angle|.  Clearly,
    # $$
    # b = - \nu\ln\left({T_c\over t_1 t_2} \right)
    # $$

    if r1 == 0 or r2 == 0:
            return (-slab.cos_angle*log(Tc / t1 / t2))

    # Well I kept putting it off, but now comes the time to solve
    # the following equation for |b|
    # $$
    # T_c = {t_1 t_2\exp(-b)\over 1-r_1r_2 \exp(-2b)}
    # $$
    # We note immediately that this is a quadratic equation in 
    # $x=\exp(-b)$. 
    # $$
    # r_1r_2T_c x^2 + t_1 t_2 x -T_c =0
    # $$
    # Sufficient tests
    # have been made above to ensure that none of the coefficients
    # are exactly zero. However, it is clear that the leading quadratic term has
    # a much smaller coefficient than the other two.  Since
    # $r_1$ and $r_2$ are typically about four percent the product is
    # roughly $10^{-3}$.  The collimated transmission can be very small
    # and this makes things even worse.  A further complication is that
    # we need to choose the only positive root.  
    # 
    # Now the roots of $ax^2+bx+c=0$ can be found using the
    # standard quadratic formula, 
    # $$
    # x = {-b\pm\sqrt{b^2-4ac}\over 2a}
    # $$
    # This is very bad for small values of $a$.  Instead I use
    # $$
    # q=-{1\over 2} \left[b+\sgn(b)\sqrt{b^2-4ac}\right]
    # $$
    # with the two roots
    # $$
    # x={q\over a}\qquad\hbox{and}\qquad x= {c\over q}
    # $$
    # Substituting our coefficients
    # $$
    # q=-{1\over 2} \left[t_1 t_2+\sqrt{t_1^2 t_2^2+4r_1r_2T_c^2}\right]
    # $$
    # With some algebra, this can be shown to be 
    # $$
    # q = -t_1 t_2 \left[1+{r_1r_2T_c^2\over t_1^2 t_2^2}+\cdots \right]
    # $$
    # The only positive root is $x=-T_c/q$.  Therefore
    # $$
    # x = { 2 T_c \over t_1 t_2 +\sqrt{t_1^2 t_2^2+4r_1r_2T_c^2}}
    # $$
    # (Not very pretty, but straightforward enough.)

    B = t1 * t2
    return (-slab.cos_angle*log(2 * Tc /(B+sqrt(B*B+ 4*Tc * Tc * r1 * r2))))


def Estimate_RT(m, r):
    """
    Estimating R and T.

    In several places, it is useful to know an estimate for the values of the
    reflection and transmission of the sample based on the measurements.  This
    routine provides such an estimate, but it currently ignores anything 
    corrections that might be made for the integrating spheres.

    Good values are only really obtainable when |num_measures==3|, otherwise 
    we need to make pretty strong assumptions about the reflection and transmission
    values.  If |num_measures<3|, then we will assume that no collimated light makes it all
    the way through the sample.  The specular reflection is then just that for a
    semi-infinite sample and $Tc=0$. If |num_measures==1|, then |Td| is also set
    to zero.

    {\settabs\+\qquad\qquad&variable&\qquad description of variable&\cr%sample line
    \+&|rt|  & total reflection\cr
    \+&|rc|  & primary or specular reflection\cr
    \+&|rd|  & diffuse or scattered reflection\cr
    \+&|tt|  & total transmission\cr
    \+&|tp|  & primary or unscattered transmission\cr
    \+&|td|  & diffuse or scattered transmission\cr}
    """

    # If there are three measurements then the specular reflection can
    # be calculated pretty well.  If there are fewer then
    # the unscattered transmission is assumed to be zero.  This is not
    # necessarily the case, but after all, this routine only makes estimates
    # of the various reflection and transmission quantities.
    # 
    # If there are three measurements, the optical thickness of the sample
    # is required.  Of course if there are three measurements then the 
    # illumination must be collimated and we can call |What_Is_B| to
    # find out the optical thickness.  We pass this value to a routine
    # in the \.{fresnel.h} unit and sit back and wait.
    # 
    # All the above is true if sphere corrections are not needed.
    # Now, we just fob this off on another function.

    Calculate_Minimum_MR(m,r,rc,tc)

    # Finding the diffuse reflection is now just a matter of checking
    # whether V1\% contains the specular reflection from the sample or
    # not and then just adding or subtracting the specular reflection as
    # appropriate.

    if m.fraction_of_rc_in_mr:
        rt = m.m_r 
        rd = *rt - m.fraction_of_rc_in_mr * (rc)
        if rd < 0:
            rd = 0
            rc = *rt
    else:
        rd = m.m_r
        rt = rd + rc

    if Debug(DEBUG_SEARCH):
        s += "        rt = %.5f\n" % rt
        s += "    est rd = %.5f\n" % rd

    # The transmission values follow in much the same way as the 
    # diffuse reflection values --- just subtract the specular
    # transmission from the total transmission.

    if m.num_measures == 1:
        tt = 0.0
        td = 0.0

    elif m.fraction_of_tc_in_mt:
        tt = m.m_t
        td = tt - tc
        if td < 0:
            tc = tt
            td = 0
    else:
        td = m.m_t
        tt = td + tc
    
    if Debug(DEBUG_SEARCH):
        s += "        tt = %.5f\n" % tt
        s += "    est td = %.5f\n" % td

    return rt, tt, rd, rc, td, tc

def a2acalc(a):
    """
    Convert albedo to calculation space albedo.
    
    This does the albedo transformations according to
    $$
    a_{calc} = {2a-1\over a(1-a)}
    $$
    Care is taken to avoid division by zero.  Why was this
    function chosen?  Well mostly because it maps the region
    between $[0,1]\rightarrow (-\infty,+\infty)$.
    """
    if a <= 0:
        return -BIG_A_VALUE

    if a >= 1:
        return BIG_A_VALUE

    return (2 * a - 1) / a / (1 - a)


def acalc2a(acalc):
    """
    @ |acalc2a| is used for the albedo transformations
    Now when we solve
    $$
    a_calc = {2a-1\over a(1-a)}
    $$
    we obtain the quadratic equation
    $$
    a_{calc} a^2 + (2-a_{calc}) a - 1 =0
    $$
    The only root of this equation between zero and one is
    $$
    a = {-2+a_{calc}+\sqrt{a_{calc}^2 +4}\over 2 a_{calc}}
    $$
    I suppose that I should spend the time to recast this using
    the more appropriate numerical solutions of the quadratic
    equation, but this worked and I will leave it as it is for now.
    """
    if acalc == BIG_A_VALUE:
        return 1.0
    if acalc == -BIG_A_VALUE:
        return 0.0
    if fabs(acalc) < SMALL_A_VALUE:
        return 0.5

    return ((-2+acalc+ sqrt(acalc * acalc + 4)) / (2 * acalc))


def g2gcalc(g):
    """
    |g2gcalc| is used for the anisotropy transformations
    according to
    $$
    g_{calc} = {g\over 1+\vert g \vert}
    $$
    which maps $(-1,1)\rightarrow(-\infty,+\infty)$.
    """
    if g <= -1: 
        return -HUGE_VAL

    if g >= 1: 
        return HUGE_VAL
    
    return g / (1 - fabs(g))


def gcalc2g(gcalc):
    """
    @ |gcalc2g| is used for the anisotropy transformations
    it is the inverse of |g2gcalc|.  The relation is
    $$
    g = {g_{calc}\over 1+\vert g_{calc}\vert}
    $$
    """
    if gcalc == -HUGE_VAL:
        return -1.0
    if gcalc == HUGE_VAL:
        return 1.0
    return gcalc / (1 + fabs(gcalc))


def b2bcalc(b):
    """
    @ |b2bcalc| is used for the optical depth transformations
    it is the inverse of |bcalc2b|.  The relation is
    $$
    b_{calc} = \ln(b)
    $$
    The only caveats are to ensure that I don't take the logarithm
    of something big or non-positive.
    """
    if b ==  HUGE_VAL:
        return HUGE_VAL
    if b <= 0:
        return 0.0
    return log(b)


def bcalc2b(bcalc):
    """
    @ |bcalc2b| is used for the anisotropy transformations
    it is the inverse of |b2bcalc|.  The relation is
    $$
    b = \exp(b_{calc})
    $$
    The only tricky part is to ensure that I don't exponentiate
    something big and get an overflow error.  In ANSI \Cee\ 
    the maximum value for $x$ such that $10^x$ is in the range
    of representable finite floating point numbers (for doubles)
    is given by |DBL_MAX_10_EXP|.  Thus if we want to know if 
    $$
    e^{b_{calc}} > 10^x
    $$
    or
    $$
    b_{calc}> x\ln(10) \approx 2.3 x
    $$
    and this is the criterion that I use.
    """
    if bcalc ==  HUGE_VAL:
        return HUGE_VAL
    if bcalc > 2.3 *  DBL_MAX_10_EXP:
        return HUGE_VAL
    return exp(bcalc)


def twoprime(a, b, g):
    """
    |twoprime| converts the true albedo |a|, optical depth |b| to
    the reduced albedo |ap| and reduced optical depth |bp| that
    correspond to $g=0$.
    """
    if a == 1 and g == 1:
        ap = 0.0
    else:
        ap = (1 - g) * a / (1 - a * g)

    if b == HUGE_VAL:
        bp = HUGE_VAL
    else:
        bp = (1 - a * g) * b

    return ap, bp


def twounprime(ap, bp, g):
    """
    |twounprime| converts the reduced albedo |ap| and reduced optical depth |bp|
    (for $g=0$) to the true albedo |a| and optical depth |b| for an anisotropy |g|.
    """

    a = ap / (1 - g + ap * g)
    if bp == HUGE_VAL:
        b = HUGE_VAL
    else:
        b = (1 + ap * g / (1 - g)) * bp
        
    return a, b


def abgg2ab(a1, b1, g1, g2):
    """
    |abgg2ab| assume |a|, |b|, |g|, and |g1| are given
    this does the similarity translation that you
    would expect it should by converting it to the
    reduced optical properties and then transforming back using
    the new value of |g|
    """
    a, b = twoprime(a1, b1, g1)
    a2,b2 = twounprime(a, b, g2)
    return a2, b2


def abgb2ag(a1, b1, b2):
    """
    |abgb2ag| translates reduced optical properties to unreduced
    values assuming that the new optical thickness is given
    i.e., |a1| and |b1| are $a'$ and $b'$ for $g=0$.  This routine
    then finds the appropriate anisotropy and albedo which
    correspond to an optical thickness |b2|.

    If both |b1| and |b2| are zero then just assume $g=0$ for the unreduced
    values.
    """
    if b1 == 0 or b2 == 0:
       a2 = a1
       g2 = 0

    if b2 < b1:
        b2 = b1

    if a1 == 0:
        a2 = 0.0
    else:
        if a1 == 1: 
            a2 = 1.0
        elif b1 == 0 or b2 == HUGE_VAL:
            a2 = a1
        else:
            a2 = 1 + b1 / b2 * (a1 - 1)

    if a2 == 0 or b2 == 0 or b2 == HUGE_VAL:
        g2 = 0.5
    else:
        g2 = (1 - b1 / b2) / a2

    return a2, b2, g2


def quick_guess(m, r):
    """
    Guessing an inverse.

    """
    UR1, UT1, rd, rc, td, tc = Estimate_RT(m, r)

    # Estimate aprime
    if UT1 == 1: 
        aprime = 1.0
        
    elif rd / (1 - UT1) >= 0.1:
        tmp = (1 - rd - UT1) / (1 - UT1)
        aprime = 1 - 4.0 / 9.0 * tmp * tmp
        
    elif rd < 0.05 and UT1 < 0.4: 
        aprime = 1 - (1-10*rd)*(1-10*rd)
        
    elif rd < 0.1 and UT1 < 0.4:
        aprime = 0.5 + (rd - 0.05) * 4
        
    else:
        tmp = (1 - 4*rd - UT1) / (1 - UT1)
        aprime = 1 - tmp * tmp

    if m.num_measures == 1:    # only reflection is known
        g = r.default_g
        a = aprime / (1 - g + aprime * g)
        b = HUGE_VAL
        return a, b, g
    
    if m.num_measures == 2:  # R and T are known
        # Estimate |bprime|
        if rd < 0.01:
            bprime = What_Is_B(r.slab, UT1)
        elif UT1 <= 0:
            bprime = HUGE_VAL
        elif UT1 > 0.1:
            bprime = 2 * exp(5 * (rd - UT1) * log(2.0))
        else:
            alpha = 1 / log(0.05 / 1.0)
            beta = log(1.0) / log(0.05 / 1.0)
            logr = log(UR1)
            bprime = log(UT1) - beta * log(0.05) + beta * logr
            bprime /= alpha * log(0.05) - alpha * logr - 1

        g = r.default_g
        a = aprime / (1 - g + aprime * g)
        b = bprime / (1 - a * g)
        return a, b, g
    
    # Guess when all three measurements are known

    if r.search == 'find_a':  # Guess when finding albedo
        g = r.default_g
        a = aprime / (1 - g + aprime * g)
        b = What_Is_B(r.slab, m.m_u)
        return a, b, g

    if r.search == 'find_b':  # Guess when finding optical depth
        g = r.default_g
        a = 0.0
        b = What_Is_B(r.slab, m.m_u)
        return a, b, g

    if r.search == 'find_ab':  # Guess when finding albedo and optical depth
        g = r.default_g

        if g == 1:
            a = 0.0
        else:
            a = aprime / (1 - g + aprime * g)
    
        if bprime == HUGE_VAL or a * g == 1:
            b = HUGE_VAL
        else:
            b = bprime / (1 - a * g)
        return a, b, g

    # Guess when finding anisotropy and albedo@>=
    b = What_Is_B(r.slab, m.m_u)
    if b == HUGE_VAL or b == 0:
        a = aprime
        g = r.default_g
    else:
        if bprime == HUGE_VAL or a * g == 1:
            b = HUGE_VAL
        else:
            b = bprime / (1 - a * g)

        a = 1 + bprime * (aprime - 1) / b
        if a < 0.1:
            g = 0.0
        else:
            g = (1 - bprime / b) / a

    if a < 0:
        a = 0.0
    if g < 0:
        g = 0.0
    elif g >= 1:
        g = 0.5

    return a, b, g


def Print_Invert_Type(r):
    s = "\n"
    s += "default  a=%10.5f   b=%10.5f    g=%10.5f\n" % (r.default_a, r.default_b,r.default_g)
    s += "slab     a=%10.5f   b=%10.5f    g=%10.5f\n" % (r.slab.a, r.slab.b,r.slab.g)
    s += "n      top=%10.5f mid=%10.5f  bot=%10.5f\n" % (r.slab.n_top_slide, r.slab.n_slab,r.slab.n_bottom_slide)
    s += "thick  top=%10.5f cos=%10.5f  bot=%10.5f\n" % (r.slab.b_top_slide, r.slab.cos_angle,r.slab.b_bottom_slide)
    s += "search = %d quadrature points = %d\n" % (r.search,r.method.quad_pts )
    return s

def Print_Measure_Type(m):
    s = "\n"
    s += "#                        Beam diameter = %7.1f mm\n" % (m.d_beam)
    s += "#                     Sample thickness = %7.1f mm\n" % (m.slab_thickness )
    s += "#                  Top slide thickness = %7.1f mm\n" % (m.slab_top_slide_thickness )
    s += "#               Bottom slide thickness = %7.1f mm\n" % (m.slab_bottom_slide_thickness )
    s += "#           Sample index of refraction = %7.3f\n" % (m.slab_index )
    s += "#        Top slide index of refraction = %7.3f\n" % (m.slab_top_slide_index )
    s += "#     Bottom slide index of refraction = %7.3f\n" % (m.slab_bottom_slide_index )
    s += "#    Fraction unscattered light in M_R = %7.1f %%\n" % (m.fraction_of_rc_in_mr*100)
    s += "#    Fraction unscattered light in M_T = %7.1f %%\n" % (m.fraction_of_tc_in_mt*100)
    s += "# \n"
    s += "# Reflection sphere\n"
    s += "#                      sphere diameter = %7.1f mm\n" % (m.d_sphere_r )
    s += "#                 sample port diameter = %7.1f mm\n" % (2*m.d_sphere_r*sqrt(m.as_r) )
    s += "#               entrance port diameter = %7.1f mm\n" % (2*m.d_sphere_r*sqrt(m.ae_r) )
    s += "#               detector port diameter = %7.1f mm\n" % (2*m.d_sphere_r*sqrt(m.ad_r) )
    s += "#                     wall reflectance = %7.1f %%\n" % (m.rw_r*100 )
    s += "#                 standard reflectance = %7.1f %%\n" % (m.rstd_r*100 )
    s += "#                 detector reflectance = %7.1f %%\n" % (m.rd_r*100 )
    s += "#                              spheres = %7d\n" % (m.num_spheres )
    s += "#                             measures = %7d\n" % (m.num_measures )
    s += "#                               method = %7d\n" % (m.method )
    s += "area_r as=%10.5f  ad=%10.5f    ae=%10.5f  aw=%10.5f\n" % (m.as_r, m.ad_r, m.ae_r, m.aw_r)
    s += "refls  rd=%10.5f  rw=%10.5f  rstd=%10.5f   f=%10.5f\n" % (m.rd_r, m.rw_r, m.rstd_r, m.f_r)
    s += "area_t as=%10.5f  ad=%10.5f    ae=%10.5f  aw=%10.5f\n" % (m.as_t, m.ad_t, m.ae_t, m.aw_t)
    s += "refls  rd=%10.5f  rw=%10.5f  rstd=%10.5f   f=%10.5f\n" % (m.rd_t, m.rw_t, m.rstd_t, m.f_t)
    s += "lost  ur1=%10.5f ut1=%10.5f   uru=%10.5f  utu=%10.5f\n" % (m.ur1_lost, m.ut1_lost, m.utu_lost, m.utu_lost)
    return s
