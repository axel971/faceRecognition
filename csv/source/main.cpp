
#include <iostream>
#include <fstream>
#include "boost/filesystem.hpp"

using namespace std;
using namespace boost::filesystem;

//examp"../../data/att_faces"

int main(int argc, char *argv[])
{

  if(argc < 2)
    cerr << "You must give a path for the image directory" << endl;
  else
    {  
      path pDirectory(argv[1]);
      directory_iterator end_itrDirectory;
      
      int nDirectory = 0; //Number of sub directory
  
      ofstream outputFile("../result/pathFace.txt"); //File to save the several path
  
      //Cycle through the directory
      for(directory_iterator itrDirectory(pDirectory); itrDirectory != end_itrDirectory; ++itrDirectory)
	{
	  //Get the path of the current sub directory
	  path p = complete(itrDirectory->path());
	  directory_iterator end_itr;
	  
	  if(is_directory(p))
	    {
	      // cycle through the directory
	      for (directory_iterator itr(p); itr != end_itr; ++itr)
		{
		  // If it's not a directory, list it. If you want to list directories too, just remove this check.
		  if (is_regular_file(itr->path()))
		    {
		      //assign current file name to current_file and echo it out to the console.
		      string current_file = itr->path().string();
		      outputFile << current_file << "\t" << nDirectory <<  endl;
		    }
		}
	      
	      nDirectory++;      
	    }
	}
      
      outputFile.close();
    }
}
