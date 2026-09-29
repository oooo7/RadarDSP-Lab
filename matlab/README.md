# MATLAB Reference Implementations

This directory will contain independent MATLAB/GNU Octave baseline scripts used for:
- Independent validation of Python signal generation and FFT resolution algorithms.
- FMCW Radar range/Doppler map verification against MATLAB Phased Array System Toolbox.
- Cross-verification of CA-CFAR detection threshold mathematical bounds.

## Directory Layout (Planned)
- `dsp_reference/` : MATLAB scripts for Nyquist sampling, FIR design, and STFT.
- `radar_reference/` : MATLAB scripts for FMCW beat generation, 2D FFT, and CA-CFAR.
- `validation_harness/` : Exported CSV verification datasets to cross-compare Python vs MATLAB numerical outputs.
