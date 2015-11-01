
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <cassert>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <fstream>
#include <sstream>

#include <vtkVersion.h>
#include <vtkSmartPointer.h>
 
#include <vtkChartXY.h>
#include <vtkContextScene.h>
#include <vtkContextView.h>
#include <vtkFloatArray.h>
#include <vtkPlotPoints.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTable.h>

class EigenFace
{
private :

  std::vector<int> m_labels, m_labelsTraining, m_labelsTest;
  std::vector< cv::Mat> m_greyImgs;
  std::vector< std::string > m_pathNames;
  int m_nClasses, m_nIndivClass, m_nIndivClassTraining, m_nIndivClassTest; 
  std::vector<bool> m_isTestData;
  cv::Mat m_testSet, m_trainingSet;
  cv::Mat m_eigenVectors, m_eigenValues;
  cv::Mat m_trainingSetEig, m_testSetEig;
  std::vector<cv::Mat> m_means;
  std::vector<int> m_predTestSet, m_predTrainingSet;

public :

  EigenFace();

  void readFileTxt(std::string path);
  void randomChooseTestSet();
  void buildTrainingTestSet();
  void eigenSpace();
  void projectTrainingSet();
  void projectTestSet();
  void showEigenFace(int iAxe1, int iAxe2);
  void meanClasses();
  int predict_aux(cv::Mat data);
  void predictTestSet();
  void predictTrainingSet();
  double mahalanobis(cv::Mat v1, cv::Mat v2, cv::Mat eigenValues);
  double errorTestSet();

};
