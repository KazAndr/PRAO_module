
# coding: utf-8

# In[2]:


from copy import copy 

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


# In[1]:


def read_prf(filename):
    """
    Help on function read_prf in module PRAO:
    
    read_prf(filename):
        return header, main_pulse, background
    
    Discription
    ----------
    The function reads a file "filename" in format *.prf.
    The function gets a name of file or path to file as input data
    and return header information, average profile of pulsar and 
    median value of noise.
    
    Parameters
    ----------
    filename : str
        Input data. Name of file in current directory or path to file.
    
    Returns
    -------
    header : dict
            Dictionary with header information, such as name of pulsar,
            resolution, numbers of points in the observation, 
            time of start of the observation and other.
    main_pulse : ndarray
            Array of intensities by time which is an average profile of pulsar.
    background : float
            Median value of noise.
    
    Examples
    --------
    It will be add in future.
    """
    
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
    Help on function read_dat in module PRAO:
    
    read_dat(filename):
        return header, main_pulse, fr, background
    
    Discription
    ----------
    The function reads a file "filename" in format *.dat.
    The function gets a name of file or path to file as input data
    and return header information, average profile of pulsar, frequency response
    and median value of noise.
    
    Parameters
    ----------
    filename : str
        Input data. Name of file in current directory or path to file.
    
    Returns
    -------
    header : dict
            Dictionary with header information, such as name of pulsar,
            resolution, numbers of points in the observation, 
            time of start of the observation and other.
    main_pulse : ndarray
            Array of intensities by time which is an average profile of pulsar.
    fr : list
            Frequency response of the observation.
    background : float
            Median value of noise.
    
    Examples
    --------
    It will be add in future.
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
    Help on function read_srez in module PRAO:
    
    read_srez(filename):
        return header, main_pulse, background
    
    Discription
    ----------
    The function reads a file "filename" in format *.srez.
    Files in format *.srez are output files from PulseViewer program 
    and are records of dedispersed average pulse of pulsar observed at 111 MHz
    (bandwight - 2.5 MHz).
    The function gets a name of file or path to file as input data
    and return header information, average profile of pulsar
    and median value of noise.
    
    Parameters
    ----------
    filename : str
        Input data. Name of file in current directory or path to file.
    
    Returns
    -------
    header : dict
            Dictionary with header information, such as name of pulsar,
            resolution, numbers of points in the observation, 
            time of start of the observation and other.
    main_pulse : ndarray
            Array of intensities by time which is an average profile of pulsar.
    background : float
            Median value of noise.
    
    Examples
    --------
    It will be add in future.
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
    Help on function read_profiles in module PRAO:
    
    read_profiles(filename):
        return header, main_pulse, data_pulses, background
    
    Discription
    ----------
    The function reads a file "filename" in format *_profiles.txt.
    Files in format *_profiles.txt are output files from PulseViewer program
    and are records of sequence of dedispersed individual pulses of pulsar
    observed at 111 MHz (bandwight - 2.5 MHz).
    The function gets a name of file or path to file as input data
    and return header information, average profile of pulsar, 
    array of individual pulses and array of values of madian noise for each pulse.
    
    Parameters
    ----------
    filename : str
        Input data. Name of file in current directory or path to file.
    
    Returns
    -------
    header : dict
            Dictionary with header information, such as name of pulsar,
            resolution, numbers of points in the observation, 
            time of start of the observation and other.
    main_pulse : ndarray
            Array of intensities by time which is an average profile of pulsar.
    data_pulses : ndarray
            Array of individual pulses.
    background : list
            List of vedian value of noise for each individual pulse.
    
    Examples
    --------
    It will be add in future.
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
    Help on function read_profiles_MD in module PRAO:
    
    read_profiles_MD(filename):
        return header, main_pulse, data_pulses, background
    
    Discription
    ----------
    The function reads a file "filename" in format *_profiles.txt.
    Files in format *_profiles.txt are output files from PulseViewer program
    and are records of sequence of dedispersed individual pulses of pulsar
    observed at 111 MHz (bandwight - 2.5 MHz). This function is used for processing
    files produced by "Использовать МД" mode. 
    The function gets a name of file or path to file as input data
    and return header information, average profile of pulsar, 
    array of individual pulses and array of values of madian noise for each pulse.
    
    Parameters
    ----------
    filename : str
        Input data. Name of file in current directory or path to file.
    
    Returns
    -------
    header : dict
            Dictionary with header information, such as name of pulsar,
            resolution, numbers of points in the observation, 
            time of start of the observation and other.
    main_pulse : ndarray
            Array of intensities by time which is an average profile of pulsar.
    data_pulses : ndarray
            Array of individual pulses.
    background : list
            List of vedian value of noise for each individual pulse.
    
    Examples
    --------
    It will be add in future.
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
    Help on function read_profiles_MD in module PRAO:
    
    read_profiles_MD(filename):
        return header, main_pulse, data_pulses, background
    
    Discription
    ----------
    The function reads a file "filename" in format *_profiles.txt.
    Files in format *_profiles.txt are output files from PulseViewer program
    and are records of sequence of dedispersed individual pulses of pulsar
    observed at 111 MHz (bandwight - 2.5 MHz). This function is used for processing
    files if some pulses were sckiped during processing (a short record, noise cleansing). 
    The function gets a name of file or path to file as input data
    and return header information, average profile of pulsar, 
    array of individual pulses and array of values of madian noise for each pulse.
    
    Parameters
    ----------
    filename : str
        Input data. Name of file in current directory or path to file.
    
    Returns
    -------
    header : dict
            Dictionary with header information, such as name of pulsar,
            resolution, numbers of points in the observation, 
            time of start of the observation and other.
    main_pulse : ndarray
            Array of intensities by time which is an average profile of pulsar.
    data_pulses : ndarray
            Array of individual pulses.
    background : list
            List of vedian value of noise for each individual pulse.
    
    Examples
    --------
    It will be add in future.
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
    Help on function edgesOprofile in module PRAO:
    
    edgesOprofile(profile, pattern):
        return l_frame, r_frame
    
    Discription
    ----------
    The function returns left and right edges of 
    average profile(or individual pulses) which were determed
    by cross-correlation with pattern of average profile(or individual pulses).
    
    Parameters
    ----------
    profile : list, ndarray
        Input data. Array of intensities by time which is an average profile of pulsar
        or individual pulse.
    pattern : list, ndarray
        Input data. Array of intensities by time which is an pattern of 
        average profile of pulsar or individual pulse.
    
    Returns
    -------
    l_frame : float
            The left edge of average profile(or individual pulses)
            which were determed by cross-correlation with pattern 
            of average profile(or individual pulses).
    r_frame : float
            The right edge of average profile(or individual pulses)
            which were determed by cross-correlation with pattern 
            of average profile(or individual pulses).
    
    Examples
    --------
    It will be add in future.
    """
    
    cor_A_B = np.correlate(profile, pattern, mode='full')
    l_frame = np.argmax(cor_A_B, axis=0) - len(pattern)
    r_frame = np.argmax(cor_A_B, axis=0)
    
    return l_frame, r_frame


# In[2]:


# Функция определения границ по определенному уровню
def edge_by_level(spline, level):
    """
    Help on function edge_by_level in module PRAO:
    
    edge_by_level(spline, level):
        return left_arg, right_arg
    
    Discription
    ----------
    This is a support function
    which determines borders  of spline at the certain level by gradual downhill.
    
    Parameters
    ----------
    spline : list, ndarray
        Input data. Array of intensities by time which is an average profile of pulsar
        or individual pulse.
    level : int, float
        Input data. The desired level for determing borders.
    
    Returns
    -------
    
    left_arg : float
            The left border of spline.
    right_arg : float
            The right border of spline.
    
    Examples
    --------
    It will be add in future.
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
    Help on function width_of_pulse in module PRAO:
    
    width_of_pulse(pulse, level):
        return w_level, left_spl, right_spl
    
    Discription
    ----------
    Function calculates width of profile or pulse 
    in certain level (10%, 50% or other). 
    Width of pulses is determined by spline interpolation.
    There is problem with multicomponents average profiles.
    
    Parameters
    ----------
    pulse : list, ndarray
        Input data. Array of intensities by time which is an average profile of pulsar
        or individual pulse.
    level : int, float
        Input data. The desired level for determing borders.
    
    Returns
    -------
    w_level : float
            The width of profile or pulse.
    left_spl : float
            The left border of spline.
    right_spl : float
            The right border of spline.
    
    Examples
    --------
    It will be add in future.
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
    Help on function width_of_pulse_up in module PRAO:
    
    width_of_pulse_up(pulse, level, l_edge, r_edge):
        return w_level, left_arg, right_arg
    
    Discription
    ----------
    Function calculates width of profile or pulse 
    in certain level (10%, 50% or other). 
    Width of pulses is determined by spline interpolation.
    The difference from width_of_pulse is that the algorithm
    for determining the width is based on calculating the required level
    from the boundaries of the profile. 
    It solves the problem with multicomponents average profiles.
    
    Parameters
    ----------
    pulse : list, ndarray
        Input data. Array of intensities by time which is an average profile of pulsar
        or individual pulse.
    level : int, float
        Input data. The desired level for determing borders.
    l_edge : int, float
        Input data. The left border of profile.
    r_edge : int, float
        Input data. The right border of profile.
        
    Returns
    -------
    w_level : float
            The width of profile or pulse.
    left_arg : float
            The left border of spline.
    right_arg : float
            The right border of spline.
    
    Examples
    --------
    It will be add in future.
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
            if spline[left_arg + i] > val_level:
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
    Help on function SNR in module PRAO:
    
    SNR(array, l_edge, r_edge):
        return snr
    
    Discription
    ----------
    Function calculates signal-noise ratio of pulse or profile. 
    The signal-noise ratio was calculated as ratio maximum of average pulse 
    and standart devision of noise.
    
    Parameters
    ----------
    array : list, ndarray
        Input data. Array of intensities by time which is an average profile of pulsar
        or individual pulse.
    l_edge : int, float
        Input data. The left border of profile.
    r_edge : int, float
        Input data. The rigth border of profile.
        
    Returns
    -------
    snr : float
            The signal-noise ratio of pulse or profile of pulsar.
    
    Examples
    --------
    It will be add in future.
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


# In[ ]:


def my_chi(profile, pattern):
    return sum([abs(pat - prof) for pat, prof in zip(pattern, profile)])

