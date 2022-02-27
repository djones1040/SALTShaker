#!/usr/bin/env python
# D. Jones - 11/20/20

import numpy as np
import pylab as plt
plt.ion()
import os
import spectral_analysis

# low mass: Si = -11.29 +/- 0.51
# low mass: H&K = -16.36 +/- 2.01
# high mass: Si = -11.39 +/- 0.39
# high mass: H&K = -13.18 +/- 1.7
# 1.2-sigma difference

def main():
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)
    ax2.tick_params(top="on",bottom="on",left="on",right="off",direction="inout",length=8, width=1.5)


    phase_data,wave_data,flux_data = np.loadtxt('SALT3.HighMass/salt3_template_0.dat',unpack=True)
    #phase_data,wave_data,flux_data = np.loadtxt('SALT3.LowMass/salt3_template_0.dat',unpack=True)
    wave_data,flux_data = wave_data[phase_data == 0],flux_data[phase_data == 0]
    phase_data,wave_data,fluxerr = np.loadtxt('SALT3.HighMass/salt3_lc_variance_0.dat',unpack=True)
    fluxerr = np.sqrt(fluxerr[phase_data == 0])
    wave_data = wave_data[phase_data == 0]
    #wave_data,flux_data,fluxerr = np.loadtxt('sim_spec/offset0/nugentsnia_p=+15_z=1.8_F200W=25.7_SNR=20_prism.csv',unpack=True,skiprows=1,delimiter=',',usecols=[0,1,2])
    #fluxobs = flux_data + fluxerr*np.random.randn(np.shape(flux_data)[0])
    #wave_data *= 10000
    #wave_data,fluxobs,fluxerr,flux_data = np.loadtxt('tmpspec.txt',unpack=True)
    #wave_data *= 2.8
    
    # measure Ca H\&K and SiII velocities
    # have to plot these also
    vexp_auto,SNR = spectral_analysis.find_vexp(wave_data,flux_data)
    sample_v_si, sample_si_min_wave, err_si = spectral_analysis.measure_velocity(
        wave_data,flux_data, 5900., 6300.,
        clip=False, vexp=vexp_auto, plot=False, error=False)

    v_si_samp = []
    for i in range(50):
        fluxobstmp = flux_data + fluxerr*np.random.randn(np.shape(flux_data)[0])
        sample_v_si_tmp, _, _ = spectral_analysis.measure_velocity(
            wave_data,fluxobstmp, 5900., 6300.,
            clip=False, vexp=vexp_auto, plot=False, error=False)
        v_si_samp += [sample_v_si_tmp]
    v_si_err = np.std(v_si_samp)
        
    sample_v_hk, sample_hk_min_wave, err_hk = spectral_analysis.measure_velocity(
        wave_data,flux_data, 3500., 4000.,
        clip=False, vexp=vexp_auto, plot=False, error=True, rest_wave=3950)
    v_hk_samp = []
    for i in range(50):
        fluxobstmp = flux_data + fluxerr*np.random.randn(np.shape(flux_data)[0])
        sample_v_hk_tmp, sample_hk_min_wave, err_hk = spectral_analysis.measure_velocity(
            wave_data,fluxobstmp, 3500., 4000.,
            clip=False, vexp=vexp_auto, plot=False, error=True, rest_wave=3950)
        v_hk_samp += [sample_v_hk_tmp]
    v_hk_err = np.std(v_hk_samp)

    print(f'Si vel: {sample_v_si} +/- {v_si_err}')
    print(f'H&K vel: {sample_v_hk} +/- {v_hk_err}')
    
    import pdb; pdb.set_trace()
    
if __name__ == "__main__":
    main()
