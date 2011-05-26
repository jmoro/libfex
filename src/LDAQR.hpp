/***************************************************************************
 *  Copyright (c) 2011 Javier Moro Sotelo.
 *  
 *  This file is part of libfex.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *   
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *   
 *  Contributors:
 *      Javier Moro Sotelo - initial API and implementation
 ***************************************************************************/

#ifndef LDAQR_HPP_
#define LDAQR_HPP_

// TODO: Check really needed header files, including all OpenCV headers
// is way too much
#include "opencv2/opencv.hpp"
#include "Classifier.hpp"
#include "MathHelpers.hpp"
#include <map>

namespace fex
{

using namespace cv;

template <typename _Tp> class LDAQR : public Classifier<_Tp>
{
public:

    LDAQR();
    virtual ~LDAQR();

    void clear();
    bool train(Mat_<_Tp> trainingSet, Mat_<int> classes);
    Mat_<int> predict(Mat_<_Tp> observations);

private:
    int numClasses;
    _Tp priorProb;
    map<int, int> classFrequency;
    map<int, Mat_<_Tp> > groupMeans;
    map<int, int> classLabels;
    Mat_<_Tp> R;
    _Tp logSigma;
};

template <typename _Tp>
LDAQR<_Tp>::LDAQR()
{
}

template <typename _Tp>
LDAQR<_Tp>::~LDAQR()
{
}

template <typename _Tp>
void LDAQR<_Tp>::clear()
{
    this->numClasses = 0;
    this->priorProb = 0;
    this->classFrequency.clear();
    this->classLabels.clear();
    this->groupMeans.clear();
    this->logSigma = 0;
    ((Mat)this->R).release();
    logSigma = 0.0;
}

template <typename _Tp>
bool LDAQR<_Tp>::train(Mat_<_Tp> trainingSet, Mat_<int> classes)
{
    int trCols = ((Mat)trainingSet).cols;
    int trRows = ((Mat)trainingSet).rows;
    int numObs = ((Mat)classes).rows;
    int clCols = ((Mat)classes).cols;

    CV_Assert((trRows == numObs) && (trRows >= trCols) && (numObs > 1) &&
            (clCols == 1));

    MatConstIterator_<int> itMat =
            ((Mat)classes).begin<int>(), itMat_end = ((Mat)classes).end<int>();

    for(; itMat < itMat_end; ++itMat)
    {
        ++this->classFrequency[*itMat];
    }

    map<int, int>::iterator itMap = this->classFrequency.begin(),
            itMap_end = this->classFrequency.end();
    int i=1;
    for(; itMap != itMap_end; ++itMap)
    {
        this->classLabels[i] = (*itMap).first;
        this->groupMeans[(*itMap).first] = Mat_<_Tp>::zeros(1,trCols);
        i++;
    }

    this->numClasses = this->classFrequency.size();

    // We should allow to pass a prior probability matrix as parameter, or a
    // map with prior probability for each class.
    this->priorProb = (1.0/this->numClasses);

    Mat_<_Tp> tempRow(1,trCols);

    // We get the mean for each group along all the dimmensions
    for(int i=0; i<numObs; i++)
    {
        tempRow = trainingSet.row(i);
        this->groupMeans[classes(i,0)] = this->groupMeans[classes(i,0)] +
                (tempRow * (1.0 / this->classFrequency[classes(i,0)]));
    }


    for(int i=0; i<numObs; i++)
    {
        trainingSet.row(i) = trainingSet.row(i) -
                this->groupMeans[classes(i,0)];
    }


    Mat_<_Tp> Q;

    MathHelpers::QR(trainingSet, Q, this->R);

    this->R = this->R/sqrt(numObs - this->numClasses);

    SVD s(this->R,SVD::NO_UV);

    Mat_<double> S(s.w);

    Mat_<_Tp> SLog;
    log(S,SLog);

    // TODO: Check if any element in s.w is <= than max(n,d) * eps(max(s)),
    // which would indicate we've got a negative covariance matrix. Throw
    // an exception in that case or return false

    this->logSigma = 2*(MathHelpers::sum1D(SLog));

    return true;
}

template <typename _Tp>
Mat_<int> LDAQR<_Tp>::predict(Mat_<_Tp> observations)
{
    int numObs = ((Mat)observations).rows;
    Mat_<int> predictions(numObs, 1);
    Mat_<_Tp> D = Mat_<int>::zeros(numObs, numClasses);
    Mat_<_Tp> A;

    map<int, int>::iterator itMap = this->classFrequency.begin(),
            itMap_end = this->classFrequency.end();

    invert(R,R);
    int j=0;
    for(; itMap != itMap_end; ++itMap)
    {

        A = MathHelpers::meanSubstraction(observations,
                groupMeans[(*itMap).first])*R;

        multiply(A,A,A);

        D.col(j) = log(priorProb)- (0.5)*(
                MathHelpers::sum2D(A,MathHelpers::MATH_BY_ROWS) +
                logSigma);
        j++;
    }

    predictions = MathHelpers::maxIndex(D, this->classLabels);

    return predictions;
}

}

#endif /* LDAQR_HPP_ */
