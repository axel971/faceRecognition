#include "EigenFace.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
  EigenFace eigenFace;

  string  path = "../../csv/result/pathFace.txt";

  //Get the all the paths and labels of the images
  eigenFace.readFileTxt(path);
  eigenFace.buildTrainingTestSet();
  eigenFace.eigenSpace();
  eigenFace.projectTrainingSet();
  eigenFace.projectTestSet();
  eigenFace.meanClasses();
  eigenFace.predictTestSet();
  eigenFace.errorTestSet();

  return 1;
}
