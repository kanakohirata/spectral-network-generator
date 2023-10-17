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
- **CLASS**: One hierarchical level below Superclass
- **SUBCLASS**: One hierarchical level below Class

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

5. Accessing a container

   ```
   docker compose -p sng-pub exec python /bin/bash
   ```

### Running the spectral-network-generator
1. Prepare spectral data and a compound table.
2. Edit the configuration file **config.ini** ("spectral-network-generator/docker/spectral_network_generator/config.ini") as necessary.
3. Execute ```python main.py```

### Stopping the Container
```
exit
docker compose -p sng-pub stop python
```
 
