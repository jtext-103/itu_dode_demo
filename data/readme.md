# An introduction to tags and meta of data from J-TEXT

## Tags

What is a tag?

Tag is a name that you use to refer a signal in a HDF5 shot file.

Each tag corresponds to a channel of raw or processed diagnostic data, typically presented in the form of a time series.

Below is descriptions of tag names and its meaning.

### Raw diagnostic data

- AXUV_CA/CB/CE/CF (16): The AXUV stands for "Absolute eXtended Ultra Violet", which is measurement for radiation emission from the plasma in J-TEXT. The letter postfix is the name of an AXUV array. Each array has many channels represented with the trailing number. Each channel is a measuring chord, since each array is a pinhole camera. Detailed description of the AXUV array is presented in Figure 1:

![Figure 1](./figures/AXUV.png)
Figure 1 The geometry of the AXUV array in J-TEXT.

- MA_POL/TOR1 (12 poloidal / 2 toroidal): Mirnov probes installed poloidally/toroidally to monitor magnetic fluctuations within the plasma in J-TEXT. The poloidal/toroidal Mirnov probes (MA_POL/TOR1) measure magnetic perturbation around the poloidal/toroidal section of plasma with a spatial resolution of 15/22.5 degrees. Detailed description of the Mirnov array is presented in Figure 2:

![Figure 2](./figures/Mirnov.png)
Figure 2 The geometry of the Mirnov array in J-TEXT.

- SXR array (38): The Soft X-Ray radiation measurement system in J-TEXT. It is similar as AXUV. It's also an pin hole camera. Detailed description of the SXR array is presented in Figure 3:

![Figure 3](./figures/SXR.png)
Figure 3 The geometry of the SXR array in J-TEXT.

- polaris_den_v01: The line integral density of the high field side
- polaris_den_v09: The line integral density of the center chord
- polaris_den_v17: The line integral density of the low field side
- ne0: The line averaged density of the center chrod
- Ihfp: The current of horizontal field.
- Iohp: The current of the ohmic field.
- Ivfp: The current of the vertical field.
- bt: The toroidal magnetic field.
- dx: The horizontal displacement.
- dy: The vertical displacement.
- ip: The plasma current.
- vl: The loop voltage.
- vs_c3_aa018: The C3 radiation.
- exsad1/4/7/10 (_J-TEXT only_): The locked mode detector placed at toroidal position of 67.5/157.5/247.5/337.5 degrees.

### Extracted features

- P_in: P*in (\_J-TEXT only*), The total input power.
- P_rad: P*rad (\_J-TEXT only*), The total radiated power.
- ip_error: The difference between plasma current and programmed plasma current.
- n=1 amplitude: The amplitude of the locked mode.
- qa_proxy: The edge safety factor.
- radiation_proxy: $\frac{P*{rad}}{P\_{in}}$.
- rotating_mode_proxy: The standard deviation of the rotating-mode-proxy magnetic field.
- ne_nG: A fraction that measures how close the plasma is approaching the Greenwald density limit.

## Meta

Meta represents the metadata to describe a discharge. If you are using JDDB lib, than you can get meta from the shot file. The meta is stored in the a groud in the HDF5 file called "meta".

Those are key-value pairs that contains useful information about the shot.

Below are important meta for disurption prediction. Note that data from different machine may have many other meta, you can just ignore them and focus on the following 3:

- IsDisrupt: Whether the discharge is disruptive or not. `1` or `True` stands for disruptive.
- DownTime: The time of disruption for a disruptive discharge, while the end of the plasma current flat-top time for a non-disruptive discharge.
- StartTime: The starting time of the flattop phase of the plasma current.

## Attribute

Each dataset has attributes which describe the data. By default, "StartTime" and "SampleRate" are provided to help build a time axis of the signal. To access the attributes, use jddb.file_repo.read_attributes() method.

## Other very useful material

- the baseline demo code for disruption prediction
  https://github.com/jtext-103/itu_dode_demo
  this repo contains a step by step demo of how to process the data and than build and evaluate a disruption prediction model.

- the JDDB library repo
  this is the JDDB repo. You may need to install it to run the demo or acess the data conviniently.
  https://github.com/jtext-103/jddb
