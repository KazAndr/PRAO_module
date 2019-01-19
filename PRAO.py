
# coding: utf-8

# In[2]:


from copy import copy 

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


# In[1]:


def read_prf(filename):
    with open(filename, 'r') as file:
        header = {}
        for i in range(12):
            *a, b = file.readline().split()
            header[a[0]] = b
    
    main_pulse = np.loadtxt(filename, skiprows=14).T
    main_pulse = main_pulse[1]
    
    background = np.median(main_pulse)
    main_pulse -= background
    
    return header, main_pulse, background


# In[ ]:


def read_dat(filename):
    """
    Return header, main pulse, frequency response and background from file "filename".
    """
        
    with open(filename, 'r') as file:
        first_line_numpar, first_line_value = file.readline().split()
        header = {first_line_numpar : first_line_value}
        for i in range(int(first_line_value)-1):
            a, *b = file.readline().split()
            try:
                header[a] = b[0] + '.' + b[1]
            except IndexError:
                header[a] = b[0].replace(',', '.')
    
    main_pulse = np.genfromtxt(
        filename,  skip_header = 1 + int(header['numpar']),
        skip_footer=1 + 511) #511 кол-во каналов
    background = np.median(main_pulse)
    main_pulse = main_pulse - background 
    
    fr = np.genfromtxt(
        filename, dtype=str,
        skip_header = 2 + int(header['numpar']) + int(header['numpointwin'])).T
    
    b = [np.float(a.replace(',','.')) for a in fr[0]]
    c = [np.float(a.replace(',','.')) for a in fr[1]]
    fr = [b, c]
    
    return header, main_pulse, fr, background


# In[1]:


def read_srez(filename):
    """
    Return header, main pulse and background from file "filename".
    """
        
    with open(filename, 'r') as file:
        first_line_numpar, first_line_value = file.readline().split()
        header = {first_line_numpar : first_line_value}
        for i in range(int(first_line_value)-1):
            a, *b = file.readline().split()
            try:
                header[a] = b[0] + '.' + b[1]
            except IndexError:
                header[a] = b[0].replace(',', '.')
    
    main_pulse = np.loadtxt(filename,  skiprows=int(header['numpar'])).T
    main_pulse = main_pulse[1]
    background = np.median(main_pulse)
    main_pulse = main_pulse - background
    
    
    return header, main_pulse, background


# In[2]:


def read_profiles(filename):
    """
    read_profiles(filename)
    Return header, main pulse, array of pulses and background from file "filename".
    """
    
    with open(filename, 'r') as file:
        first_line_numpar, first_line_value = file.readline().split()
        header = {first_line_numpar : first_line_value}
        for i in range(int(first_line_value)-1):
            a, *b = file.readline().split()
            try:
                header[a] = b[0] + '.' + b[1]
            except IndexError:
                header[a] = b[0].replace(',', '.')
    
        data_pulses = []
        for i in range(int(header["numpuls"])):
            file.readline()
            data_pulses.append([])
            for j in range(int(header["l_point_win"]) + 1):
                a, b = file.readline().split()
                data_pulses[i].append(float(b.replace(',', '.')))
        data_pulses = np.asarray(data_pulses)
    
    #удаление подложки отдельно для каждого импульса и получение среднего профиля
    main_pulse = 0
    background = []
    for item in data_pulses:
        back = np.median(item)
        item -= back
        main_pulse += item
        background.append(back)
    main_pulse = main_pulse/int(header['numpuls'])
    
    return header, main_pulse, data_pulses, background


# In[ ]:


def read_profiles_MD(filename):
    """
    read_profiles_MD(filename)
    Return header, main pulse, array of pulses and background from file "filename".
    Modifited program for reading date from PulseViewer with active option "Использовать МД"
    """
    
    with open(filename, 'r') as file:
        first_line_numpar, first_line_value = file.readline().split()
        header = {first_line_numpar : first_line_value}
        for i in range(int(first_line_value)-1):
            a, *b = file.readline().split()
            try:
                header[a] = b[0] + '.' + b[1]
            except IndexError:
                header[a] = b[0].replace(',', '.')
    
        data_pulses = []
        for i in range(int(header["numpuls"])-1):
            file.readline()
            data_pulses.append([])
            for j in range(int(header["l_point_win"]) + 1):
                a, b = file.readline().split()
                data_pulses[i].append(float(b.replace(',', '.')))
        data_pulses = np.asarray(data_pulses)
    
    #удаление подложки отдельно для каждого импульса и получение среднего профиля
    main_pulse = 0
    background = []
    for item in data_pulses:
        back = np.median(item)
        item -= back
        main_pulse += item
        background.append(back)
    main_pulse = main_pulse/int(header['numpuls'])
    
    return header, main_pulse, data_pulses, background


# In[1]:


def read_profiles_w_skip(filename):
    """
    read_profiles_w_skip(filename)
    Return header, main pulse, array of pulses and background from file "filename".
    Modifited program for reading date from PulseViewer if some pulses were skiped
    """
    
    with open(filename, 'r') as file:
        first_line_numpar, first_line_value = file.readline().split()
        header = {first_line_numpar : first_line_value}
        for i in range(int(first_line_value)-1):
            a, *b = file.readline().split()
            try:
                header[a] = b[0] + '.' + b[1]
            except IndexError:
                header[a] = b[0].replace(',', '.')
    
        data_pulses = []
        for i in range(int(header["numpuls"])-1):
            file.readline()
            data_pulses.append([])
            for j in range(int(header["l_point_win"]) + 1):
                try:
                    a, b = file.readline().split()
                except ValueError:
                    data_pulses.remove([])
                    break
                data_pulses[i].append(float(b.replace(',', '.')))
        data_pulses = np.asarray(data_pulses)
    
    #удаление подложки отдельно для каждого импульса и получение среднего профиля
    main_pulse = 0
    background = []
    for item in data_pulses:
        back = np.median(item)
        item -= back
        main_pulse += item
        background.append(back)
    main_pulse = main_pulse/int(header['numpuls'])
    
    return header, main_pulse, data_pulses, background


# In[3]:


def edgesOprofile(profile, pattern):
    """
    edgesOprofile(profile, pattern)
    Return left and right edges of average profile which were determed by cross-correlation
    with pattern
    """
    cor_A_B = np.correlate(profile, pattern, mode='full')
    l_frame = np.argmax(cor_A_B, axis=0) - len(pattern)
    r_frame = np.argmax(cor_A_B, axis=0)
    
    return l_frame, r_frame


# In[2]:


# Функция определения границ по определенному уровню
def edge_by_level(spline, level):
    """
    edge_by_level(spline, level)
    """
    max_arg_spline = np.argmax(spline)
    left_arg = None
    right_arg = None
    #left edge
    for i in range(len(spline[:max_arg_spline+1])):
        try:
            if spline[max_arg_spline - i] < level:
                left_arg = max_arg_spline - i
                break
            else:
                continue
        except IndexError:
            left_arg = 0
            break
        
    
    #Right edge
    for i in range(len(spline[max_arg_spline-1:])):
        try:
            if spline[max_arg_spline + i] < level:
                right_arg = max_arg_spline + i
                break
            else:
                continue
        except IndexError:
                right_arg = len(spline)
                break
                
                
    if left_arg == None: left_arg = 0
    if right_arg == None: right_arg = len(spline)
    
    return left_arg, right_arg  


# In[1]:


def width_of_pulse(pulse, level):
    """
    width_of_pulse(pulse, level)
    Function calculate width of profile or pulse in certain level (10%, 50% or other).
    Width of pulses determ by spline interpolation.
    There are problem with multicomponents average profile
    """
    inter_point = 20000
    
    point_for_pulse = range(len(pulse))
    max_point = np.argmax(pulse)
    
    spl = InterpolatedUnivariateSpline(point_for_pulse, pulse)
    xs = np.linspace(0, len(point_for_pulse)-1, inter_point) # подумать об увеличении количества точек для интерполяции
    
    spline = spl(xs)
    
    #  Расчет на определенном уровне%
    val_level = level*max(spline)
    left_spl, right_spl = edge_by_level(spline, val_level)
    left_spl = left_spl * ((len(point_for_pulse)-1)/inter_point)
    right_spl = right_spl * ((len(point_for_pulse)-1)/inter_point)
    
    # расчет ширины
    w_level = (right_spl - left_spl)
    
    return w_level, left_spl, right_spl


# In[ ]:


def width_of_pulse_up(pulse, level, l_edge, r_edge):
    """
    width_of_pulse_up(pulse, level, l_edge, r_edge)
    """
    inter_point = 20000
    
    point_for_pulse = range(len(pulse))
    
    spl = InterpolatedUnivariateSpline(point_for_pulse, pulse)
    xs = np.linspace(0, len(point_for_pulse)-1, inter_point) # подумать об увеличении количества точек для интерполяции
    
    spline = spl(xs)
    
    #  Расчет на определенном уровне%
    val_level = level * max(pulse)
    left_arg = int(l_edge * (inter_point/len(pulse)))
    right_arg = int(r_edge * (inter_point/len(pulse)))
    #left edge
    
    for i in range(len(spline[left_arg:right_arg])):
        try:
            if spline[left_arg + i] > val_level :
                left_arg = left_arg + i
                break
            else:
                continue
        except IndexError:
            left_arg = 0
            break
        
    
    #Right edge
    for i in range(len(spline[left_arg:right_arg])):
        try:
            if spline[right_arg - i] > val_level :
                right_arg = right_arg - i
                break
            else:
                continue
        except IndexError:
                right_arg = len(spline)
                break
    
    left_arg = left_arg * ((len(point_for_pulse)-1)/inter_point)
    right_arg = right_arg * ((len(point_for_pulse)-1)/inter_point)
    
    # расчет ширины
    w_level = (right_arg - left_arg)
    
    return w_level, left_arg, right_arg


# In[7]:


def SNR(array, l_edge, r_edge):
    """
    Return signal-noise ratio which was calculated as ratio maximum of average pulse 
    and standart devision of noise.
    """
    return max(array[l_edge:r_edge])/np.std(np.append(array[:l_edge], array[r_edge:]), 
                                            dtype=np.float64)


# In[1]:


def quality_data(profile, l_edge, r_edge):
    """
    Function for checking average profile for qood or bad data. 
    Input: average profile, left edge, rigtht edge
    Output: True if average profile is good or False if bad 
    
    """
    prf = copy(profile)   
    prf -= np.median(prf)
    #main_profile = np.roll(main_profile, int(int(header['numpointwin'])/2) - np.argmax(main_profile))
    prf /= np.max(prf)
    
    noise = np.append(prf[:l_edge], prf[r_edge:])
        
    intersection = np.argwhere(np.diff(np.sign(prf - 0.5*np.max(prf)))).flatten()
    pulses = str(len(intersection)/2)
        
    fp, residuals, rank, sv, rcond = np.polyfit(range(len(noise)), noise, 1, full=True)
    f = np.poly1d(fp)
        
    cond_1 = float(pulses) <=10
    cond_2 = abs(round(np.median(noise) - np.mean(noise), 4)) <= 0.0055
    cond_3 = round(SNR(prf, l_edge, r_edge), 1) >= 3.5
    cond_4 = abs(round(f[0], 5)) < 0.05
        
    if cond_1 and cond_2 and cond_3 and cond_4:
        return True
    else:
        return False


# In[2]:


data = [[1, 2, 3], [4, 5, 6], []]


# In[4]:


data.remove([])


# In[5]:


data

