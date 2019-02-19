
# coding: utf-8

# In[1]:


import numpy as np

from astropy.time import Time
from astropy import units as u
from scipy import ndimage
from scipy.interpolate import InterpolatedUnivariateSpline

from PRAO import *


# In[ ]:


def move_profile(head, main_pulse, shift):
    """
    Function for shifting pulse in window of observation
    Input: header, profile of pulse and needed shift for pulse
    Output: header and profile of pulse
    """
    
    main_pulse = np.roll(main_pulse, shift)
    
    day, month, year = head['date'].split('.')
    hour, minute, second = head['time'].split(':')
    isot_time = (year + '-' + month + '-' + day + 'T' +
                 hour + ':' + minute + ':' + second
                ) 
    t = Time(isot_time, format='isot', scale='utc', precision=7)
    t -= shift * np.float64(head['tay'])*u.millisecond
    
    head['date'] = str(t.datetime.day) + '.' + str(t.datetime.month) + '.' + str(t.datetime.year)
    head['time'] = str(t.datetime.hour) + ':' + str(t.datetime.minute) + ':' + t.value[17:]
    
    return head, main_pulse


# In[2]:


def max_spl_toa(head, main_pulse, zone):
    """
    Defenition TOA by maximum of spline of pulse profile.
    Return TOA as MJD and error in millisecond. In addition, function return maximum of spline
    """
    
    NUM_POINT = 10000
    spl = InterpolatedUnivariateSpline(range(len(main_pulse)), main_pulse)
    xs = np.linspace(0, len(main_pulse)-1, NUM_POINT)

    max_spl = np.argmax(spl(xs))
    max_spl = max_spl * ((len(main_pulse)-1)/NUM_POINT)
    
    error = 0.5*(float(head['tay'])*u.millisecond)
    error = error.to(u.microsecond)
    error = error.round(1)
    
    day, month, year = head['date'].split('.')
    hour, minute, second = head['time'].split(':')
    isot_time = (year + '-' + month + '-' + day + 'T' +
                 hour + ':' + minute + ':' + second 
                ) 
    t = Time(isot_time, format='isot', scale='utc', precision=7)
    t = t - int(zone)*u.hour #Перевод местного времени в UTC.
    t = t + max_spl * float(head['tay'])*u.millisecond
    
    return t.mjd, error.value, max_spl


# In[3]:


def max_corrf_toa(head, main_pulse, pattern, zone):
    """
    Defenition TOA by maximum of correlation pulse profile and pattern.
    Return TOA as MJD and error in millisecond. In addition, function return tuple 
    which contains maximum of cor func, left and right edges 
    """
    
    cor_A_B = np.correlate(main_pulse, pattern, 'full')
    l_frame = np.argmax(cor_A_B, axis=0) - len(pattern)
    r_frame = np.argmax(cor_A_B, axis=0)

    max_cor = np.argmax(cor_A_B, axis=0) - 0.5*len(pattern)
    
    if l_frame < 0:
        l_frame = 0
    elif r_frame > len(main_pulse):
        r_frame = len(main_pulse)
    else:
        pass
    
    norm_res = (max_cor, l_frame, r_frame,)
    
    day, month, year = head['date'].split('.')
    hour, minute, second = head['time'].split(':')
    isot_time = (year + '-' + month + '-' + day + 'T' +
                 hour + ':' + minute + ':' + second 
                ) 
    t = Time(isot_time, format='isot', scale='utc', precision=7)
    t = t - int(zone)*u.hour #Перевод местного времени в UTC.
    t = t + max_cor * float(head['tay'])*u.millisecond
    """
    Определение погрешности
    """
    snr = SNR(main_pulse, l_frame, r_frame)
    wight10, l, r = width_of_pulse(main_pulse, 0.1)
    error = 0.3*np.sqrt((wight10 - 1)*(float(head['tay'])*u.millisecond)**2)/snr
    error = error.to(u.microsecond)
    error = error.round(1)
    
    return t.mjd, error.value, norm_res


# In[6]:


def max_corr_spls_toa(head, main_pulse, pattern, zone):
    """
    Defenition TOA by maximum of correlation spline of pulse profile and spline of pattern.
    Return TOA as MJD and error in millisecond. In addition, function return tuples 
    which contain maximum of cor func, left and right edges, 
    for normal correlation and splines correlation respectively. 
    """
    
    NUM_POINT = 100
    
    # вводится для определения погрешности определения МПИ
    cor_norm = np.correlate(main_pulse, pattern, 'full')
    l_frame_norm = np.argmax(cor_norm, axis=0) - len(pattern)
    r_frame_norm = np.argmax(cor_norm, axis=0)
    max_cor_norm = np.argmax(cor_norm, axis=0) - 0.5*len(pattern)
    
    if l_frame_norm < 0:
        l_frame_norm = 0
    elif r_frame_norm > len(main_pulse):
        r_frame_norm = len(main_pulse)
    else:
        pass
    
    norm_res = (max_cor_norm, l_frame_norm, r_frame_norm,)
    
    splMP = InterpolatedUnivariateSpline(range(len(main_pulse)), main_pulse)
    xsMP = np.linspace(0, len(main_pulse)-1, len(main_pulse)*NUM_POINT)
    splPAT = InterpolatedUnivariateSpline(range(len(pattern)), pattern)
    xsPAT = np.linspace(0, len(pattern)-1, len(pattern)*NUM_POINT)
    cor_splines = np.correlate(splMP(xsMP), splPAT(xsPAT), 'full')
                    
    l_frame_spline = np.argmax(cor_splines, axis=0) - len(splPAT(xsPAT))
    r_frame_spline = np.argmax(cor_splines, axis=0)
    max_cor_spline = np.argmax(cor_splines, axis=0) - 0.5*len(splPAT(xsPAT))            
    
    spl_res = (max_cor_spline, l_frame_spline, r_frame_spline)
    """
    Определение погрешности
    """
    snr = SNR(main_pulse, l_frame_norm, r_frame_norm)
    wight10, l, r = width_of_pulse(main_pulse, 0.1)
            
    error = 0.3*np.sqrt((wight10 - 1)*(float(head['tay'])*u.millisecond)**2)/snr
    error = error.to(u.microsecond)
    error = error.round(1)
    """
    Окончание определения погрешности
    """
    day, month, year = head['date'].split('.')
    hour, minute, second = head['time'].split(':')
    isot_time = (year + '-' + month + '-' + day + 'T' +
                 hour + ':' + minute + ':' + second 
                ) 
    t = Time(isot_time, format='isot', scale='utc', precision=7)
    t = t - int(zone)*u.hour #Перевод местного времени в UTC.
    t = t + (max_cor_spline/NUM_POINT) * float(head['tay'])*u.millisecond
    
    return t.mjd, error.value, norm_res, spl_res 


# In[2]:


def max_str_pls_toa(head, main_pulse, data_pulses, pattern, zone):
    """
    Defenition TOA by maximum of strongest pulse.
    Return TOA as MJD and error in millisecond, strong pulse ind index of strong pulse. 
    """
    # определяем границы среднего профиля
    l_edg, r_edg = edgesOprofile(main_pulse, pattern)
    # Определяем сильнейший импульс в записи
    pls = [float('-inf')]
    idx = None
    for i, pulse in enumerate(data_pulses):
        if (np.max(pulse) > np.max(pls)) and (l_edg <= np.argmax(pulse) <= r_edg):
            pls = pulse
            idx = i
        
    """
    Определение погрешности
    """
    snr = SNR(pls, l_edg, r_edg)
    wight10, l, r = width_of_pulse(pls, 0.1)
            
    error = 0.3*np.sqrt((wight10 - 1)*(float(head['tay'])*u.millisecond)**2)/snr
    error = error.to(u.microsecond)
    error = error.round(1)
    """
    Окончание определения погрешности
    """
    day, month, year = head['date'].split('.')
    hour, minute, second = head['time'].split(':')
    isot_time = (year + '-' + month + '-' + day + 'T' +
                 hour + ':' + minute + ':' + second 
                ) 
    t = Time(isot_time, format='isot', scale='utc', precision=7)
    t = t - int(zone)*u.hour #Перевод местного времени в UTC.
    t = t + idx*(float(head['period'])*u.second) + (np.argmax(pls)) * float(head['tay'])*u.millisecond
    
    return t.mjd, error.value, pls, idx 


# In[ ]:


def toa_center_mass(head, main_pulse, pattern, zone):
    """
    Defenition TOA by center mass of average profile.
    Return TOA as MJD and error in millisecond, TOA as point. 
    """
    # определяем границы среднего профиля
    l_edg, r_edg = edgesOprofile(main_pulse, pattern)
    phase = l_edg + ndimage.measurements.center_of_mass(main_profile[l_edg:r_edg])
    """
    Определение погрешности
    """
    snr = SNR(main_pulse, l_edg, r_edg)
    wight10, l, r = width_of_pulse(main_pulse, 0.1)
            
    error = 0.3*np.sqrt((wight10 - 1)*(float(head['tay'])*u.millisecond)**2)/snr
    error = error.to(u.microsecond)
    error = error.round(1)
    """
    Окончание определения погрешности
    """
    day, month, year = head['date'].split('.')
    hour, minute, second = head['time'].split(':')
    isot_time = (year + '-' + month + '-' + day + 'T' +
                 hour + ':' + minute + ':' + second 
                ) 
    t = Time(isot_time, format='isot', scale='utc', precision=7)
    t = t - int(zone)*u.hour #Перевод местного времени в UTC.
    t = t + phase * float(head['tay'])*u.millisecond
    
    return t.mjd, error.value, phase


# In[9]:


def write_tempo(head, toa_list, errs_list, lablel):
    """
    Save tim file format tempo1
    Input: header, array with toa, array with errors of toa and lablel which means metod of determing toa 
    Output: None
    """
    freq = '111.879'
    observ = 'p'
    big_space = ' '*10
    midle_space = ' '*3
    small_space = ' '*2
    space4mjd = 23
    #accuracy = str(float(head['tay'])*10**3)
    with open(head['name'][:-3] + lablel + '_t1' '.tim', 'w') as file:
        for date, er in zip(toa_list, errs_list):
            try:
                accuracy = str(er)
                integer, fractional = accuracy.split('.')
            except ValueError:
                accuracy = str(float(head['tay'])*10**3)
                integer, fractional = accuracy.split('.')
            string = (' ' + 'B' + head['name'] + big_space + '01' + small_space + 
                      '0' + small_space + freq + '   ' + str(date) + 
                      ' '*(space4mjd-len(str(date))) + '0.00' + 
                      ' '*(6 - len(integer)) + integer + '.' + fractional +
                      ' '*(9 - len(fractional)) + observ)
        
            file.write(string)
            file.write('\n')
    
    return None


# In[10]:


def write_tempo2(head, toa_list, errs_list, lablel):
    """
    Save tim file format tempo2
    Input: header, array with toa, array with errors of toa and lablel which means metod of determing toa 
    Output: None
    """
    freq = '111.87900'
    observ = 'p'
    big_space = ' '*15
    midle_space = ' '*3
    small_space = ' '*2
    space4mjd = 20
    #accuracy = str(float(head['tay'])*10**3)
    with open(head['name'][:-3] + lablel + '_t2' '.tim', 'w') as file:
        file.write('FORMAT 1')
        file.write('\n'*2)
        for date, er in zip(toa_list, errs_list):
            try:
                accuracy = str(er)
                integer, fractional = accuracy.split('.')
            except ValueError:
                accuracy = str(float(head['tay'])*10**3)
                integer, fractional = accuracy.split('.')
            string = (head['name'] + big_space + freq + '   ' + str(date) + 
                      ' '*(space4mjd-len(str(date))) + 
                      ' '*(6 - len(integer)) + integer + '.' + fractional[0] +
                      ' ' + observ)
        
            file.write(string)
            file.write('\n')
    
    return None

