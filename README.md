# OpticalCommunications

Simulation code and educational notebooks to support the lectures of the **Optical Communications** course at the Electrical Engineering Department of the Federal University of Campina Grande (UFCG).

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edsonportosilva/OpticalCommunications/HEAD?urlpath=lab)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## Overview

This repository provides interactive Jupyter notebooks that combine theory, mathematical derivations, and Python simulations for studying optical communication systems. The material covers topics from basic optical transmitters and receivers through to advanced coherent systems and digital signal processing (DSP) techniques.

The notebooks are available in **English** and **Portuguese** and are designed to be explored interactively — either directly in the cloud via Binder (no local installation required) or in a local Python environment.

---

## How to Use

### ▶ Run in the Cloud (recommended)

Click the **Binder** badge above to launch a fully interactive JupyterLab environment in your browser — no installation needed.

### 💻 Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/edsonportosilva/OpticalCommunications.git
   cd OpticalCommunications
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch JupyterLab:**
   ```bash
   jupyter lab
   ```

4. Open the notebooks in the `jupyter notebooks/English/` or `jupyter notebooks/Portuguese/` folder.

---

## Dependencies

The following Python packages are required (see [`requirements.txt`](./requirements.txt)):

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays and signal processing |
| `scipy` | Scientific computing and signal analysis |
| `matplotlib` | Plotting and visualization |
| `sympy` | Symbolic mathematics and equation display |
| `pandas` | Data manipulation and analysis |
| `scikit-commpy` | Digital communications utilities |
| `opticommpy` | Optical communications simulation toolkit (transmitters, receivers, fiber, DSP) |
| `tqdm` | Progress bars for simulations |
| `numba` | JIT compilation for performance-critical code |

---

## Repository Structure

```
OpticalCommunications/
├── requirements.txt                  # Python dependencies
├── jupyter notebooks/
│   ├── English/                      # Notebooks in English
│   │   ├── 0. Optical transmitters.ipynb
│   │   ├── 1-Optical transmitters.ipynb
│   │   ├── 2. Optical receivers and noise.ipynb
│   │   ├── 3. Optical fiber.ipynb
│   │   ├── 4. Optical fiber_dispersion.ipynb
│   │   ├── 5. Optical fiber - dispersion and loss.ipynb
│   │   ├── 6. Optical fiber - SSFM.ipynb
│   │   ├── 7. Optical amplification.ipynb
│   │   ├── 8. Introduction to coherent optical communications.ipynb
│   │   ├── 9. Equalization in coherent optical systems.ipynb
│   │   ├── Coherent WDM systems.ipynb
│   │   ├── MZM.ipynb
│   │   ├── Split-Step Fourier Method (SSFM).ipynb
│   │   └── utils.py                  # Utility functions for the notebooks
│   ├── Portuguese/                   # Notebooks in Portuguese
│   │   └── [equivalent notebooks in Portuguese]
│   ├── data/                         # Data files used by notebooks
│   └── figuras/                      # Figures and diagrams
```

---

## Notebook Contents

The notebooks are organized in a numbered sequence that follows the course curriculum. Below is a description of each notebook.

### 0. Optical Transmitters (introductory)
An introductory notebook covering the fundamentals of optical transmitters. Topics include the Mach-Zehnder Modulator (MZM) transfer function, operating point biasing, on-off keying (OOK) modulation, extinction ratio, and frequency limitations.

### 1. Optical Transmitters (advanced)
A deeper look at optical transmitter design. Covers the IQ modulator (IQM) and how complex-valued optical fields are generated for advanced modulation formats such as QPSK and QAM. Includes numerical simulations of modulated optical signals.

### 2. Optical Receivers and Noise
Covers the theory and simulation of optical receivers. Topics include:
- Intensity Modulation / Direct Detection (IMDD) receivers
- Sources of noise: shot noise (with power spectral density analysis), thermal noise
- Signal-to-noise ratio (SNR) calculations
- p-i-n and APD-based photodetectors
- Receiver sensitivity, matched filtering, and bit error probability

### 3. Optical Fiber
Introduces the physical properties of optical fiber. Topics include:
- The Sellmeier equation for wavelength-dependent refractive index
- Fiber modes and Bessel function solutions to the characteristic equation
- Field intensity distribution and spot size

### 4. Optical Fiber – Dispersion
Focuses on chromatic dispersion and its effect on pulse propagation. Covers group velocity, group velocity dispersion (GVD), pulse broadening in the time domain, and spectral effects.

### 5. Optical Fiber – Dispersion and Loss
Extends the dispersion analysis to include fiber attenuation. Simulates end-to-end 10G OOK transmission over a fiber span with both dispersion and loss, and evaluates signal degradation.

### 6. Optical Fiber – Split-Step Fourier Method (SSFM)
Introduces the nonlinear Schrödinger equation (NLSE) that governs pulse propagation in fiber, and implements the **Split-Step Fourier Method (SSFM)** to solve it numerically. Covers the symmetric and asymmetric SSFM schemes and simulates nonlinear effects.

### 7. Optical Amplification
Covers Erbium-Doped Fiber Amplifiers (EDFAs). Topics include:
- EDFA energy levels and population inversion
- Gain and noise figure characterization
- Forward, backward, and bidirectional pumping topologies
- Rate and propagation equations
- Simulation of amplified 10G OOK transmission over 40 km

### 8. Introduction to Coherent Optical Communications
An introduction to coherent detection and advanced modulation. Topics include:
- Motivation and sensitivity advantages of coherent systems
- Coherent modulation formats: BPSK, QPSK, QAM
- Coherent receiver architecture with 90° hybrid
- Phase noise modeling (Lorentzian laser linewidth)
- Comparison of OOK vs. BPSK/QPSK/QAM system performance
- Polarization-division multiplexing (PDM)

### 9. Equalization in Coherent Optical Systems
Covers digital signal processing (DSP) techniques for mitigating impairments in coherent systems:
- Chromatic dispersion (CD) compensation in frequency and time domain
- Polarization Mode Dispersion (PMD) and differential group delay (DGD)
- Adaptive linear equalization with FIR filters
- Least Mean Squares (LMS) algorithm with training sequences
- Constant Modulus Algorithm (CMA) for blind equalization

### Coherent WDM Systems
End-to-end simulation of a Wavelength Division Multiplexing (WDM) system using coherent detection. Covers multichannel transmission, channel spacing, and DSP-based demultiplexing.

### MZM
A focused notebook on the Mach-Zehnder Modulator (MZM): electro-optic effect, driving voltage, chirp, and small-signal frequency response.

### Split-Step Fourier Method (SSFM)
A standalone, detailed walkthrough of the SSFM algorithm used to simulate nonlinear fiber propagation, including derivation steps and convergence analysis.

---

## Utility Functions (`utils.py`)

Each language folder contains a `utils.py` module with helper functions used across the notebooks:

| Function | Description |
|---|---|
| `symdisp(expr, var, unit)` | Display SymPy expressions in LaTeX format |
| `round_expr(expr, numDig)` | Round numerical values in symbolic expressions |
| `symplot(t, F, interval, ...)` | Plot SymPy symbolic functions |
| `genGIF(x, y, figName, ...)` | Generate and save animated GIFs from plot data |
| `genSignalGIF(x, y, windowSize, ...)` | Animated signal plots with a sliding window |
| `genConvGIF(x, h, t, totalTime, ...)` | Animate convolution operations step by step |

---

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](./LICENSE) file for details.
