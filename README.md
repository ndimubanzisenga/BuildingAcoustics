# BuildingAcoustics


# Python packages to install to run code:
* numpy
* scipy
* matplotlib
* wavio

# Non python packages to install to run code
* sox

# Testing the software:
The software can be tested in two ways:
## Using the DASYLab scripts
* It can be used in Real Time tests. The implementation works either with the PC sound card and microphone, or with a DT9832A data acquisition device fitted with a measurement microphone.
* It provides a GUI to initialize the hardware, and run the different building acoustics tests to measure building acoustics parameters and descriptors, and to detect violations of building acoustics regulations.
* The DASYLab scripts are found under the '/src/dasylab/' folder
## Using Python in a standalone mode
* Run '/src/scripts/test/standalone_test.py'

# Reference dataset
* Datasets for three test tests were created: for Sound Pressure Level calculation, for Reverberation Time calculation and for defects diagnosis.
* The datasets were created for different environments, and for different sound sources.
* Datasets can be found under the '/data' folder
