#include "EigenFace.hpp"

using namespace std;
using namespace cv;

EigenFace::EigenFace()
{}

void EigenFace::readFileTxt(string path)
{
  /*
    NB : the label of the first class must be zero
   */

  ifstream file(path, ios::in);  //Open the file

  if(file) 
    {   
      //Read the line 
      string lineString;

      while(getline(file, lineString))
	{
	  istringstream line(lineString); //Convert the string lineString into the flux line
	  string pathName;
	  int label;

	  while(line >> pathName >> label)
	    {
	      m_pathNames.push_back(pathName);
	      m_labels.push_back(label);
	      m_greyImgs.push_back(imread(pathName, CV_LOAD_IMAGE_GRAYSCALE));
	    }
	}

      //Close the file
      file.close();
    }

  else
    cerr << "It's impossible to open this file !" << endl;

  //Initialize
  auto it = max_element(std::begin(m_labels), std::end(m_labels));
  m_nClasses = (*it) + 1;
  m_nIndivClass = m_labels.size() / m_nClasses;
  
 }


void EigenFace::randomChooseTestSet()
{
  //Initialize isTestData
  vector<bool> isTestData(m_greyImgs.size(), false); 
  
  srand(time(0)); //Initialise the seed

  for(int i = 0; i < m_nClasses; ++i) 
    {
      int indice = (i * m_nIndivClass) + (rand() % m_nIndivClass);
      isTestData[indice] = true;
    }

  m_isTestData = isTestData;

}


void EigenFace::buildTrainingTestSet()
{

  randomChooseTestSet();

  for(int i = 0; i < m_greyImgs.size(); ++i) 
    {
      Mat vec = m_greyImgs[i].reshape(0, 1).t(); //Transform the image into vector

      if(!m_isTestData[i])
	{ 
	  if(m_trainingSet.empty())
	    m_trainingSet = vec;
	  else
	    hconcat(m_trainingSet, vec, m_trainingSet);  

	  m_labelsTraining.push_back(m_labels[i]);
	}
      else
	{
	  if(m_testSet.empty())
	    m_testSet = vec;
	  else
	    hconcat(m_testSet, vec, m_testSet);  
	
	  m_labelsTest.push_back(m_labels[i]);
	 
	} 
    }
  
  //Convert the data of trainingSet and testSet in double precision
  m_trainingSet.convertTo(m_trainingSet, CV_64FC1);
  m_testSet.convertTo(m_testSet, CV_64FC1);

  m_nIndivClassTraining = m_trainingSet.cols / m_nClasses;
  m_nIndivClassTest = m_testSet.cols / m_nClasses;

}

void EigenFace::eigenSpace()
{
  Mat eigenVectorsC, eigenVectors;

 //Compute the covariance matrix
  Mat c  = m_trainingSet.t() * m_trainingSet; 
 
  //Compute eigen space
  eigen(c, m_eigenValues, eigenVectorsC);

  //Compute the  eigenVectors
  m_eigenVectors = m_trainingSet * eigenVectorsC.t(); 
}

void EigenFace::projectTrainingSet()
{
  m_trainingSetEig = m_trainingSet.t() * m_eigenVectors;
}

void EigenFace::projectTestSet()
{
  m_testSetEig = m_testSet.t() * m_eigenVectors;
}

void EigenFace::showEigenFace(int iAxe1, int iAxe2)
{
  // Set up a 2D scene, add an XY chart to it
  vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
  view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);
  view->GetRenderWindow()->SetSize(400, 300);
 
  vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
  view->GetScene()->AddItem(chart);
  chart->SetShowLegend(false);
  
  // Create a table with some points in it...
  vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
  
  vtkSmartPointer<vtkFloatArray> arrX =   vtkSmartPointer<vtkFloatArray>::New();
  arrX->SetName("X Axis");
  table->AddColumn(arrX);
  
  vtkSmartPointer<vtkFloatArray> arrY =  vtkSmartPointer<vtkFloatArray>::New();
  arrY->SetName("Y Axis");
  table->AddColumn(arrY);
  
  // Test charting with a few more points...
  table->SetNumberOfRows(m_trainingSetEig.rows);

  for (int i = 0; i < m_trainingSetEig.rows; ++i)
    {
      table->SetValue(i, 0, m_trainingSetEig.at<double>(i, iAxe1));
      table->SetValue(i, 1, m_trainingSetEig.at<double>(i, iAxe2));
    }
 
  // Add multiple scatter plots, setting the colors etc
  vtkPlot *points = chart->AddPlot(vtkChart::POINTS);
#if VTK_MAJOR_VERSION <= 5
  points->SetInput(table, 0, 1);
#else
  points->SetInputData(table, 0, 1);
#endif
  points->SetColor(0, 0, 0, 255);
  points->SetWidth(1.0);
  vtkPlotPoints::SafeDownCast(points)->SetMarkerStyle(vtkPlotPoints::CROSS);
   
  //Finally render the scene
  view->GetRenderWindow()->SetMultiSamples(0);
  view->GetInteractor()->Initialize();
  view->GetInteractor()->Start();
}



void EigenFace::meanClasses()
{
 
  for(int i = 0; i < m_trainingSetEig.rows; i+= m_nIndivClassTraining)
    {
       Mat tmp = m_trainingSetEig(Rect(0, i, m_trainingSetEig.cols, m_nIndivClassTraining)) ; 
       Mat mean;
       reduce(tmp, mean, 0, CV_REDUCE_AVG);
       m_means.push_back(mean);      
    }

}

double EigenFace::mahalanobis(Mat v1, Mat v2, Mat eigenValues)
{
  double dist = 0;

  for(int i = 0; i < v1.cols ; ++i)
    {
      dist += pow(v1.at<double>(i) - v2.at<double>(i), 2) / eigenValues.at<double>(i);
    }

  return dist;
}

int EigenFace::predict_aux(Mat data)
{
 
 //To DO :   MeanClasses must be range by ascending order of their label classe changed that
 
  double dist, infDist;
  int pred;

  for(int i = 0; i < m_means.size(); ++i)
    {
  dist = mahalanobis(data, m_means[i], m_eigenValues); 
  
    
  if(i == 0)
	{
	  infDist = dist;
	  pred = i;
	}
      
      else if(dist < infDist)
	{
	  pred = i;
	  infDist = dist;
	}
    }

  return pred;
}

void EigenFace::predictTestSet()
{
  vector<int> res;

  for(int i = 0; i < m_testSetEig.rows; ++i)
    {
      Mat data  = m_testSetEig(Rect(0, i, m_testSetEig.cols, 1)) ;
      res.push_back(predict_aux(data));
    }
  
  m_predTestSet = res;
}

void EigenFace::predictTrainingSet()
{
  vector<int> res;

  for(int i = 0; i < m_trainingSetEig.rows; ++i)
    {
      Mat data  = m_trainingSetEig(Rect(0, i, m_trainingSetEig.cols, 1)) ;
      res.push_back(predict_aux(data));
    }

  m_predTrainingSet = res;
}


double EigenFace::errorTestSet()
{

  assert(!m_predTestSet.empty());
  assert(!m_labelsTest.empty());
  assert(m_predTestSet.size() == m_labelsTest.size());

  double k = 0;

  for(int i = 0; i < m_labelsTest.size(); ++i)
    {
      if(m_predTestSet[i] != m_labelsTest[i])
	 k++;
    }

      k /= m_predTestSet.size();
      
      cout << "The taux of error is " << k << endl; 
}
