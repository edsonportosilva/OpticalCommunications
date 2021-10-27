import numpy as np
from cupyx.scipy.fft import fft, ifft, fftfreq
from cupy.random import normal
import scipy.constants as const
from tqdm.notebook import tqdm
from numba import njit, jit
import cupy as cp


def edfa(Ei, Fs, G=20, NF=4.5, Fc=193.1e12):
    """
    Simple EDFA model

    :param Ei: input signal field [nparray]
    :param Fs: sampling frequency [Hz][scalar]
    :param G: gain [dB][scalar, default: 20 dB]
    :param NF: EDFA noise figure [dB][scalar, default: 4.5 dB]
    :param Fc: optical center frequency [Hz][scalar, default: 193.1e12 Hz]    

    :return: amplified noisy optical signal [nparray]
    """
    assert G > 0, 'EDFA gain should be a positive scalar'
    assert NF >= 3, 'The minimal EDFA noise figure is 3 dB'
    
    NF_lin   = 10**(NF/10)
    G_lin    = 10**(G/10)
    nsp      = (G_lin*NF_lin - 1)/(2*(G_lin - 1))
    N_ase    = (G_lin - 1)*nsp*const.h*Fc
    p_noise  = N_ase*Fs    
    noise    = normal(0, cp.sqrt(p_noise), Ei.shape) + 1j*normal(0, cp.sqrt(p_noise), Ei.shape)
    return Ei*cp.sqrt(G_lin) + noise

def manakovSSF(Ei, Fs, paramCh):      
    """
    Manakov model split-step Fourier (symmetric, dual-pol.)

    :param Ei: input signal
    :param Fs: sampling frequency of Ei [Hz]
    :param paramCh: object with physical parameters of the optical channel
    
    :paramCh.Ltotal: total fiber length [km][default: 400 km]
    :paramCh.Lspan: span length [km][default: 80 km]
    :paramCh.hz: step-size for the split-step Fourier method [km][default: 0.5 km]
    :paramCh.alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
    :paramCh.D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
    :paramCh.gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
    :paramCh.Fc: carrier frequency [Hz] [default: 193.1e12 Hz]
    :paramCh.amp: 'edfa', 'ideal', or 'None. [default:'edfa']
    :paramCh.NF: edfa noise figure [dB] [default: 4.5 dB]    
    
    :return Ech: propagated signal
    """
    # check input parameters
    paramCh.Ltotal = getattr(paramCh, 'Ltotal', 400)
    paramCh.Lspan  = getattr(paramCh, 'Lspan', 80)
    paramCh.hz     = getattr(paramCh, 'hz', 0.5)
    paramCh.alpha  = getattr(paramCh, 'alpha', 0.2)
    paramCh.D      = getattr(paramCh, 'D', 16)
    paramCh.gamma  = getattr(paramCh, 'gamma', 1.3)
    paramCh.Fc     = getattr(paramCh, 'Fc', 193.1e12)
    paramCh.amp    = getattr(paramCh, 'amp', 'edfa')
    paramCh.NF     = getattr(paramCh, 'NF', 4.5)   

    Ltotal = paramCh.Ltotal 
    Lspan  = paramCh.Lspan
    hz     = paramCh.hz
    alpha  = paramCh.alpha  
    D      = paramCh.D      
    gamma  = paramCh.gamma 
    Fc     = paramCh.Fc     
    amp    = paramCh.amp   
    NF     = paramCh.NF

    # channel parameters  
    c_kms = const.c/1e3 # speed of light (vacuum) in km/s
    λ  = c_kms/Fc
    α  = alpha/(10*np.log10(np.exp(1)))
    β2 = -(D*λ**2)/(2*np.pi*c_kms)
    γ  = gamma

    c_kms = cp.asarray(c_kms) # speed of light (vacuum) in km/s
    λ  = cp.asarray(λ)
    α  = cp.asarray(α)
    β2 = cp.asarray(β2)
    γ  = cp.asarray(γ)
    hz = cp.asarray(hz)
    
    # generate frequency axis 
    Nfft = len(Ei)
    ω = 2*np.pi*Fs*fftfreq(Nfft)
    
    Nspans = int(np.floor(Ltotal/Lspan))
    Nsteps = int(np.floor(Lspan/hz))

    Ei = cp.asarray(Ei)
    
    Ech_x = Ei[:,0].reshape(len(Ei),)
    Ech_y = Ei[:,1].reshape(len(Ei),)

    # define linear operator
    linOperator = cp.array(cp.exp(-(α/2)*(hz/2) + 1j*(β2/2)*(ω**2)*(hz/2)))
    
    for spanN in tqdm(range(1, Nspans+1)):   
        Ech_x = fft(Ech_x) #polarization x field
        Ech_y = fft(Ech_y) #polarization y field
        
        # fiber propagation step
        for stepN in range(1, Nsteps+1):            
            # First linear step (frequency domain)
            Ech_x = Ech_x*linOperator
            Ech_y = Ech_y*linOperator

            # Nonlinear step (time domain)
            Ex = ifft(Ech_x)
            Ey = ifft(Ech_y)
            Ech_x = Ex*cp.exp(1j*(8/9)*γ*(Ex*cp.conj(Ex)+Ey*cp.conj(Ey))*hz)
            Ech_y = Ey*cp.exp(1j*(8/9)*γ*(Ex*cp.conj(Ex)+Ey*cp.conj(Ey))*hz)

            # Second linear step (frequency domain)
            Ech_x = fft(Ech_x)
            Ech_y = fft(Ech_y)
            
            Ech_x = Ech_x*linOperator
            Ech_y = Ech_y*linOperator
            
        # amplification step
        Ech_x = ifft(Ech_x)
        Ech_y = ifft(Ech_y)
        
        if amp =='edfa':
            Ech_x = edfa(Ech_x, Fs, alpha*Lspan, NF, Fc)
            Ech_y = edfa(Ech_y, Fs, alpha*Lspan, NF, Fc)
        elif amp =='ideal':
            Ech_x = Ech_x*cp.exp(α/2*Nsteps*hz)
            Ech_y = Ech_y*cp.exp(α/2*Nsteps*hz)
        elif amp == None:
            Ech_x = Ech_x*cp.exp(0);
            Ech_y = Ech_y*cp.exp(0);

    Ech_x = cp.asnumpy(Ech_x)
    Ech_y = cp.asnumpy(Ech_y)
    
    Ech = np.array([Ech_x.reshape(len(Ei),),
                    Ech_y.reshape(len(Ei),)]).T
    
    return Ech, paramCh