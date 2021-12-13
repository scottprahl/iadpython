# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string

"""
Functions to figure out how close guess is to correct answer.

import iadpython as iad

s = iad.Sample(a=0.95, b=3)
x = iad.Experiment(sample=s, m_r=0.5, m_t=0.1)
delta = corrected_distance(x)
"""

import numpy as np
import iadpython as iad


def corrected_distance(x):
    """
    Calculate distance including corrections.  

    The measured values are corrected by sphere parameters and light that
    was not collected during the experiment.
    
    These corrected values are used with the sphere formulas to convert the
    modified |ur1| and |ut1| to values for |*M_R| and |*M_T|.
    """
    s = x.sample  
    ur1, ut1, uru, utu = s.rt()
    
    # find the unscattered reflection and transmission
    nu_inside = iad.cos_snell(1, s.nu_0, s.n)
    r_un, t_un = iad.specular_rt(s.n_above, s.n, s.n_below, s.b, nu_inside)

    # correct for lost light
    R_diffuse = uru - x.uru_lost
    T_diffuse = utu - x.utu_lost
    R_direct = ur1 - x.ur1_lost
    T_direct = ut1 - x.ut1_lost

    # correct for fraction not collected
    R_direct -= (1.0 - x.fraction_of_rc_in_mr) * r_un
    T_direct -= (1.0 - x.fraction_of_tc_in_mt) * t_un

    # If no spheres were used in the measurement, then presumably the
    # measured values are the reflection and transmission.  Consequently,
    # we just acertain what the irradiance was and whether the
    # specular reflection ports were blocked and proceed accordingly.
    # Note that blocking the ports does not have much meaning unless
    # the light is collimated, and therefore the reflection and
    # transmission is only modified for collimated irradiance.
    if exp.num_spheres == 0:
        M_R = R_direct
        M_T = T_direct

    if exp.num_spheres in [-2, 1]:
        if MM.method == COMPARISON:
            # The dual beam case is different because the sphere efficiency
            # is equivalent for measurement of light hitting the sample first or
            # hitting the reference standard first.  The dual beam measurement
            # should report the ratio of these two reflectance measurements, thereby
            # eliminating the need to calculate the gain completely.   The same
            # holds when no sample is present.
            # 
            # The normalized reflectance measurement (the difference between
            # dual beam measurement for a port with the sample and with nothing)
            # is
            # $$
            # M_R = r_\std\cdot(1-f)\rdirect  + f r_w\over (1-f')r_\std -f' r_w
            #     - r_\std\cdot(1-f) (0)  + f r_w\over (1-f')r_\std -f' r_w
            # $$
            # or
            # $$
            # M_R = (1-f)\rdirect \over (1-f') -f' r_w/r_\std
            # $$
            # When $f=f'=1$, then $M_R=1$ no matter what the reflectance is.
            # (Leave it in this form to avoid division by zero when $f=1$.)
            # 
            # The normalized transmittance is simply $\tdirect$.
            # 
            # When $f=0$ then this result is essentially the same as the
            # no spheres result (because no sphere corrections are needed).
            # However if the number of spheres is zero, then no lost light
            # calculations are made and therefore that is a potential error.
            # 
            *M_R     = (1-MM.f_r) * R_direct/((1-MM.f_r) + MM.f_r*MM.rw_r/MM.rstd_r)
            *M_T     = T_direct
        else:
            # The direct incident power is $(1-f) P$. The reflected power will 
            # be $(1-f)\rdirect P$.  Since baffles ensure that the light cannot
            # reach the detector, we must bounce the light off the sphere walls to
            # use to above gain formulas.  The contribution will then be 
            # $(1-f) \rdirect (1-a_e) r_w P$.  The measured power will be 
            # $$
            # P_d = a_d (1-a_e) r_w [(1-f) \rdirect + f r_w] P \cdot G(r_s)
            # $$
            # Similarly the power falling on the detector measuring transmitted light is
            # $$
            # P_d'= a_d' \tdirect r_w'  (1-a_e')  P \cdot G'(r_s)
            # $$
            # when the `entrance' port in the transmission sphere is closed, $a_e'=0$.
            # 
            # The normalized sphere measurements are
            # $$
            # M_R = r_\std\cdotR(\rdirect,r_s)-R(0,0) \over R(r_\std,r_\std)-R(0,0)
            # $$
            # and
            # $$
            # M_T = t_\std\cdotT(\tdirect,r_s)-T(0,0) \over T(t_\std,r_\std)-T(0,0)
            # $$
            # 
            # @<Calc |M_R| and |M_T| for single beam sphere@>=

            G_0      = Gain(REFLECTION_SPHERE, MM, 0.0)
            G        = Gain(REFLECTION_SPHERE, MM, R_diffuse)
            G_std    = Gain(REFLECTION_SPHERE, MM, MM.rstd_r)
    
            P_d      = G     * (R_direct  * (1-MM.f_r) + MM.f_r*MM.rw_r)
            P_std    = G_std * (MM.rstd_r * (1-MM.f_r) + MM.f_r*MM.rw_r)
            P_0      = G_0   * (                         MM.f_r*MM.rw_r)   
            *M_R     = MM.rstd_r * (P_d - P_0)/(P_std - P_0)

            GP       = Gain(TRANSMISSION_SPHERE, MM, R_diffuse)
            GP_std   = Gain(TRANSMISSION_SPHERE, MM, 0.0)
            *M_T     = T_direct * GP / GP_std

    if exp.num_spheres == 2:
        # """When two integrating spheres are present then the
        # double integrating sphere formulas are slightly more complicated.
        # 
        # I am not sure what it means when |rstd_t| is not unity.
        # 
        # The normalized sphere measurements for two spheres are
        # 
        # $$
        # M_R = R(\rdirect,\rdiffuse,\tdirect,\tdiffuse) - R(0,0,0,0)
        # \over R(r_\std,r_\std,0,0) - R(0,0,0,0)
        # $$
        # and
        # $$
        # M_T = T(\rdirect,\rdiffuse,\tdirect,\tdiffuse) - T(0,0,0,0)
        #                    \over T(0,0,1,1) - T(0,0,0,0)
        # $$
        # 
        # Note that |R_0| and |T_0| will be zero unless one has explicitly
        # set the fraction |m.f_r| ore |m.f_t| to be non-zero.
        # 
        # @<Calc |M_R| and |M_T| for two spheres@>=
        # """
        # double R_0, T_0
        # R_0 = Two_Sphere_R(MM, 0, 0, 0, 0)
        # T_0 = Two_Sphere_T(MM, 0, 0, 0, 0)

        *M_R = MM.rstd_r * (Two_Sphere_R(MM, R_direct, R_diffuse, T_direct, T_diffuse) - R_0)/
                           (Two_Sphere_R(MM, MM.rstd_r, MM.rstd_r, 0, 0) - R_0)
        *M_T =  (Two_Sphere_T(MM, R_direct, R_diffuse, T_direct, T_diffuse) - T_0)/
                           (Two_Sphere_T(MM, 0, 0, 1, 1) - T_0)
    
    """There are at least three things that need to be considered here.
    First, the number of measurements.  Second, is the metric is relative or absolute.  
    And third, is the albedo fixed at zero which means that the transmission
    measurement should be used instead of the reflection measurement.

    @<Calculate the deviation@>=
    """
    if (RR.search==FIND_A  || RR.search==FIND_G || RR.search==FIND_B || 
        RR.search==FIND_Bs || RR.search == FIND_Ba) 
        """This part was slightly tricky.  The crux of the problem was to
        decide if the transmission or the reflection was trustworthy.  After
        looking a bunches of measurements, I decided that the transmission
        measurement was almost always more reliable.  So when there is just
        a single measurement known, then use the total transmission if it 
        exists.

        @<One parameter deviation@>=
        """
        if  MM.m_t > 0:
            if RR.metric == RELATIVE: 
                distance = abs(MM.m_t - *M_T) / (MM.m_t + ABIT)
            else: 
                distance = abs(MM.m_t - *M_T) 
         else: 
            if RR.metric == RELATIVE: 
                distance = abs(MM.m_r - *M_R) / (MM.m_r + ABIT)
            else: 
                distance = abs(MM.m_r - *M_R) 
     else: 
#         This stuff happens when we are doing two parameter searches.
#         In these cases there should be information in both R and T.
#         The distance should be calculated using the deviation from
#         both.  The albedo stuff might be able to be take out.  We'll see.
        if RR.metric == RELATIVE: 
            distance = 0
            if MM.m_t > ABIT:
                distance = T_TRUST_FACTOR* abs(MM.m_t - *M_T) / (MM.m_t + ABIT)
            if  RR.default_a != 0:
                distance += abs(MM.m_r - *M_R) / (MM.m_r + ABIT)          
         else: 
            distance = T_TRUST_FACTOR * abs(MM.m_t - *M_T)
            if  RR.default_a != 0:
                distance += abs(MM.m_r - *M_R)
    return distance
