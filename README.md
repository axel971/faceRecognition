#Isotropic non linear diffusion
This programm implement the eigenFace method for face recognition. <br>

NB : Before apply the algorithm of face recognition you need to create a text file with all the path directory of images.
You can do that in compiling the application inside the directory named path <br>

You need to install the following libraries for this project : <br>
OPENCV <br>
VTK <br>

Instructions for compilation (linux) : <br>

Compilation to creare the txt file : <br>
cd csv
cmake -H. -Bbuild/ <br>
cd build <br>
make <br>
./exec pathImageDirectory <br> 


Compilation for the eigenFace module : <br>
cd eigenFace
cmake -H. -Bbuild/ <br>
cd build <br>
make <br>
./exec <br>
