# Spectral Network Generator
The Spectral Network Generator is a Python program that generates networks based on spectral similarity. The network data can be visualized using the Spectral Network Visualizer (https://github.com/kanakohirata/spectral-network-visualizer).

[Matchms](https://github.com/matchms/matchms) (Huber et al., 2020) is used for calculating spectral similarity.

Huber F. et al.  (2020) matchms - processing and similarity evaluation of mass spectrometry data. J. Open Source Softw., 5, 2411.

## Preparation of Spectral, Compound Data
The Spectral Network Generator calculate networks among sample spectra, between sample and reference spectra, and among reference spectra. Inputting a blank spectra allows you to remove blank peaks from the sample spectra.

Also, by inputting a compound table, you can add compound information (compound class, pathway) to the reference spectra, and use this to group or filter the reference spectra.

### Acceptable Spectrum Files
You can input spectrum files in **MSP**, **MG**F, or **JSON** format.


- **MSP**: You can download an MSP file from the MassBank of North America repository (MoNA) at https://mona.fiehnlab.ucdavis.edu/downloads
- **MGF**: You can download an MGF file from GNPS at https://gnps-external.ucsd.edu/gnpslibrary
- **JSON**: You can download a JSON file from the MoNA at https://mona.fiehnlab.ucdavis.edu/downloads and GNPS at https://gnps-external.ucsd.edu/gnpslibrary

### Acceptable Compound Tables
You can input a tab-delimited (TSV) file.


**Required column: InChIKey**

Acceptable columns:
- **CAS RN**: CAS Registry Number
- **DBID**: ID in the source database
- **InChI**
- **MOLECULAR FORMULA**
- **NAME**
- **SUPERCLASS**: Superclass is one of the hierarchical levels used to categorize chemical compounds in the ClassyFire (http://classyfire.wishartlab.com/) classification system.
- **CLASS**: a hierarchical level below Superclass
- **SUBCLASS**: a hierarchical level below Class

## How to Use the Spectral Network Generator
### Installation of Docker and Downloading the GitHub Repository
- Installation of Docker

  https://docs.docker.com/engine/install/
- Downloading the GitHub Repository

  Click "**Code**" then "**Download ZIP**" [in this page](https://github.com/kanakohirata/spectral-network-generator/tree/release), and unzip it in any directory.

### Creating and accessing a Docker Container
1. Open the command prompt, move it to the docker folder directly under the unzipped repository ("spectral-network-generator/docker").

   If you want to move to "D:/spectral-network-generator/docker", execute ```d:``` and ```cd D:/spectral-network-generator/docker```

3. Creating a container

   ```
   docker compose -p sng-pub up -d --build
   ```
   
   The project name "sng-pub" can be changed to any name you prefer.

4. Starting the container
   ```
   docker compose -p sng-pub start python
   ```

5. Accessing the container

   ```
   docker compose -p sng-pub exec python /bin/bash
   ```

### Running the spectral-network-generator
1. Prepare spectral data and a compound table.
2. Edit the configuration file **config.ini** ("spectral-network-generator/spectral_network_generator/config.ini").
   Description of each parameter can be found in the config.ini file.
4. Execute ```python main.py```

### Stopping the Container
```
exit
docker compose -p sng-pub stop python
```

## Options
You can set options by changing the config.ini file.

### \[filter] section
Filtering spectra
- **authors**  
  Collect reference spectra created by specified authors.
- **name_key_characters_to_remove**  
  Remove reference spectra whose name (mostly compound name) contains name_key_characters_to_remove.
- **instrument_type**  
  Collect reference spectra obtained by specified instrument type.
- **ion_mod**  
  Collect reference spectra of specified ion mode.
- **precursor_type**  
  Collect reference spectra of specified precursor type.
- **ionization**  
  Collect reference spectra of specified ionization mode: ESI, APCI, etc.
- **fragmentation_type**  
  Collect reference spectra obtained by specified fragmentation type (CID or HCD).
- **min_number_of_peaks**  
  Remove reference spectra if the number of peaks is less than min_number_of_peaks.
- **path_of_compound_dat_for_filter**  
  Collect reference spectra of compounds contained in the specified MetaCyc dat files.
- **remove_spec_wo_prec_mz**  
  Remove spectra with no precursor m/z
- **filename_avoid_filter**  
  Files that you do not want to apply filers.
- **num_top_X_peak_rich**  
  Collect the top N spectra with the highest number of product ions.

### \[spectrum processing] section
- **mz_tol_to_remove_blan**  
  m/z tolerance to remove blank spectra from sample spectra.
- **rt_tol_to_remove_blank**  
  retention time tolerance in seconds to remove blank spectra from sample spectra.
- **remove_low_intensity_peaks**  
  Lower threshold for relative peak intensity. Maximum is 1.
- **deisotope_int_ratio**  
  Intensity ratio to remove isotope.
  If it is 3, isotope whose intensity is less than 1/3 of monoisotope will be removed.
- **deisotope_mz_tol**  
  m/z tolerance to remove isotope.
- **topN_binned_ranges_topN_number, topN_binned_ranges_bin_size**  
  If you want to keep 2 most intense peaks with bins of 100 m/z (0 - 100 m/z, 100 - 200 m/z, ...), set topN_binned_ranges_topN_number = 2, topN_binned_ranges_bin_size = 100
- **intensity_convert_mode**  
  Convert peak intensity after normalization of intensity.
  - 0 : do nothing
  - 2: log (1+x) (preventing negative value). intensity 0.1 will be  0.095310, intensity 1 will be 0.69314
  - 3 : square root
