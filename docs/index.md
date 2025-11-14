---
layout: default

---

# LINK: Long-Term Intracortical Neural Activity and Kinematics

**Long-term Intracortical Neural activity and Kinematics (LINK)** is a chronic, large-scale, high-resolution intracortical dataset collected from a nonhuman primate performing dexterous finger movements over more than four years.
LINK serves as a useful dataset for exploring **non-stationarity in brain-machine interfaces, computational neuroscience, and in machine learning generally.**

---

## Experimental Setup
<img src="experimental_setup.png" alt="Experimental Setup" style="float: left; width: 300px; margin-right: 20px;">

- **Subject:** Male rhesus macaque (*Macaca mulatta*), recorded from 7-11 years of age.
- **Behavioral task:** Two-degree-of-freedom, trial-based individuated finger flexion task ("center-out" and "random" target styles). Two task variations:
    1. *Center-out*: fixed sequence of target positions.
    2. *Random target*: randomized order and position each trial.
- **Implants:** Two 64-channel Utah microelectrode arrays (Blackrock Neurotech) placed in the hand area of the right precentral gyrus.
- **Neural Features:** 96-channel threshold crossings (TCR) and spiking-band power (SBP), binned into 20ms timepoints.
   - **Threshold Crossings (TCR):** spike count features obtained using -4.5x RMS thresholding at 30 kHz sampling, then summed into 20 ms bins.
   - **Spiking-Band Power (SBP):** RMS power of bandpass-filtered (300-1,000 Hz) data, sampled at 2 kHz and averaged into 20 ms bins.
- **Kinematics:** Finger joint angles measured continuously and synchronized with neural recordings, binned into 20ms timepoints. Position was recorded during experiments, and velocities were calculated *post hoc*

---

## Data Format and Organization

Data is organized by 'sessions' - 375 (mostly) contiguous trials per session. Each task variation is one of the two specified above. In total, there are 312 sessions, recorded on 303 days, spanning 1,242 days (~3.5 years).

All files are hosted on the [DANDI Archive](https://dandiarchive.org/dandiset/001201) and follow the **Neurodata Without Borders (NWB)** format. 

Demo notebooks are available to illustrate data loading and structure – see the github readme for details.

---

## Access
- [**Dataset**](https://dandiarchive.org/dandiset/001201/0.251023.2336)
- [**Code and documentation**](https://github.com/chesteklab/LINK_dataset)
-  [**Paper - Currently OpenReview**](https://openreview.net/forum?id=TAdeh1dLzu)

## Citation

If you use this dataset, please cite both the LINK paper and the DANDI repository.
