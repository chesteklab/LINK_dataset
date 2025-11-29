---
layout: default

---

[**Dataset**](https://dandiarchive.org/dandiset/001201/0.251023.2336) | [**Code**](https://github.com/chesteklab/LINK_dataset) | [**Paper**](https://openreview.net/forum?id=TAdeh1dLzu)

# LINK: Long-Term Intracortical Neural Activity and Kinematics

**Long-term Intracortical Neural activity and Kinematics (LINK)** is a chronic intracortical dataset collected from a single nonhuman primate performing a dexterous finger movement task over ~3.5 years.
LINK serves as a useful dataset for exploring **non-stationarity in brain-machine interfaces, computational neuroscience, and in machine learning generally.**

___

## Data Format and Organization

Data is organized by 'sessions', with 375 (mostly) contiguous trials per session. Each session will be one of the two task variations described below. In total, there are 312 sessions, recorded on 303 days, spanning 1,242 days (~3.5 years).

All files are hosted on the [DANDI Archive](https://dandiarchive.org/dandiset/001201) and follow the **Neurodata Without Borders (NWB)** format. 

Demo notebooks are available to illustrate data loading and structure – see the github readme for details.

___

## Experimental Setup

![Experimental Setup](experimental_setup.png)

**Subject**: Male rhesus macaque (*Macaca mulatta*), recorded from 7-11 years of age. Recordings were captured from day 349 post-implant to 1591 days post-implant.

**Behavioral task:** Two degree-of-freedom, trial-based individuated finger flexion task ("center-out" and "random" target styles). Each session is one of two task variations, described in more detail in the paper:

1. *Center-out*: Starting with a 'rest' position, trials alternate between reaches out in set directions and returns to rest. 
2. *Random target*: Trials are pseudo-randomly selected each trial.
    
**Implants:** Two 64-channel Utah microelectrode arrays (Blackrock Neurotech) placed in the hand area of the right precentral gyrus.

**Neural Features:** 96-channel threshold crossings (TCR) and spiking-band power (SBP), binned into 20-ms timepoints.

- **Threshold Crossings (TCR):** Spike were counted using a -4.5*RMS threshold on each channel at 30 kHz sampling, then summed into 20-ms bins.
- **Spiking-Band Power (SBP):** Raw neural data was bandpass-filtered from 300-1,000 Hz and downsampled to 2 kHz. The resulting signal power was then averaged into 20-ms bins.

**Kinematics:** Finger joint angles measured continuously and synchronized with neural recordings, binned into 20ms timepoints. Position was recorded during experiments, and velocities were calculated *post hoc*.

___

## Citation

If you use this dataset, please cite both the LINK paper and the DANDI repository.
