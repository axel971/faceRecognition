#Face recognition algorithm
This programm implement the eigenFace method for face recognition. <br>

You need to install the following libraries for this project : <br>
OPENCV <br>
VTK <br>

You must download the att_faces.zip file here : http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html <br>
After that you need to create a text file holding all the path directory coresponding of att_faces images. 
For doing that, you must compile the application inside the directory called path <br>

Instructions for compilation (linux) : <br>

Compilation to create the txt file : <br>
cd path
cmake -H. -Bbuild/ <br>
cd build <br>
make <br>
./exec pathDirectory <br> 


Compilation for the eigenFace module : <br>
cd eigenFace
cmake -H. -Bbuild/ <br>
cd build <br>
make <br>
./exec <br>
