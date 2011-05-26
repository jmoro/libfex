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

#ifndef MATHHELPERS_HPP_
#define MATHHELPERS_HPP_

// TODO: Check really needed header files, including all OpenCV headers
// is way too much
#include <cmath>
#include <map>

namespace fex
{

using namespace cv;

class MathHelpers
{
public:

    const static int MATH_BY_ROWS=0;
    const static int MATH_BY_COLS=1;
    const static bool MATH_ECON_MODE_ON=true;
    const static bool MATH_ECON_MODE_OFF=false;

    template<typename _Tp>
    static Mat_<_Tp> givensRotation(const Mat_<_Tp>& mat, int i, int j);

    /*
     * QR Factorization. A=Q*R.
     *
     * A is a square matrix (size NxN)
     * R is upper-triangular
     * Q is Ortogonal, hence Q'*Q=1.
     *
     * We can use QR decomposition to solve A*x = b by doing:  R*x = Q'*b.
     */
    template<typename _Tp>
    static void QR(const Mat_<_Tp>& A, Mat_<_Tp>& Q, Mat_<_Tp>& R,
            bool econ = MATH_ECON_MODE_ON);

    template<typename _Tp>
    static void cumSum(const Mat_<_Tp>& mat, Mat_<_Tp>& sum);

    template<typename _Tp>
    static _Tp sum1D(Mat_<_Tp> mat);

    template<typename _Tp>
    static Mat_<_Tp> sum2D(Mat_<_Tp> mat, int type = MATH_BY_ROWS);

    template<typename _Tp>
    static Mat_<int> maxIndex(Mat_<_Tp> mat,
            map<int, int> labels = map<int, int>(), int type = MATH_BY_ROWS);

    // TODO: Mow we only accept a mean matrix as a mean matrix for column
    // dimmension, we sould consider also this method for row dimmension.
    template<typename _Tp>
    static Mat_<_Tp> meanSubstraction(const Mat_<_Tp>& mat,
            const Mat& mean);

    template<typename _Tp>
    static void getZeroMeanScoresFromPCA(const Mat_<_Tp>& mat,
    		const PCA& pca, Mat_<_Tp>& zeroMeanScores);

    template<typename _Tp>
    static void pcaReduceData(const Mat_<_Tp>& mat, const _Tp variability,
            Mat_<_Tp>& reducedData, Mat_<_Tp>& coefficients);
};

template<typename _Tp>
Mat_<_Tp> MathHelpers::givensRotation(const Mat_<_Tp>& mat, int i, int j)
{
    int rows = ((Mat)mat).rows;
    //CV_ASSERT((i>0) && (j>=0) && (i<rows) && (j<cols));
    Mat_<_Tp> G = Mat_<_Tp>::eye(rows, rows);

    _Tp a = mat(i-1,j);
    _Tp b = mat(i,j);
    _Tp r = sqrt(pow(a,2) + pow(b,2));
    _Tp c = a/r;
    _Tp s = -(b/r);

    G(i,i) = c;
    G(i-1,i) = -s;
    G(i,i-1) = s;
    G(i-1,i-1) = c;

    return G;
}


// TODO: Add flags to avoid Q calculation if not needed, and to work in
// econ mode.
// FIXME: get rid of near 0 values
template<typename _Tp>
void MathHelpers::QR(const Mat_<_Tp>& A, Mat_<_Tp>& Q, Mat_<_Tp>& R,
        bool econ)
{
    int i, j;
    int rows = (Mat(A)).rows;
    int cols = (Mat(A)).cols;
    Mat_<_Tp> Rtmp = A.clone();
    Q = Mat_<_Tp>::eye(rows, rows);
    Mat_<_Tp> tempGivensRot;

    for(j=0; j<cols; j++)
    {
        for(i=rows-1; i>j; i--)
        {
            tempGivensRot = givensRotation(Rtmp, i, j);
            Q=tempGivensRot*Q;
            Rtmp=tempGivensRot*Rtmp;
        }
    }
    if(econ)
    {
        ((Mat)Rtmp(Range(0, cols), Range(0,cols))).copyTo(R);

    }
    else {
        ((Mat)Rtmp).copyTo(R);
    }
    Q=((Mat)Q).t();
}

template<typename _Tp>
void MathHelpers::cumSum(const Mat_<_Tp>& mat, Mat_<_Tp>& sum)
{
    sum.create(1,((Mat)mat).cols);
    sum(0,0) = mat(0,0);

    for(int i=1; i<((Mat)mat).cols; i++)
    {
        sum(0,i) = sum(0,i-1) + mat(0,i);
    }
}

template<typename _Tp>
_Tp MathHelpers::sum1D(Mat_<_Tp> mat)
{
    _Tp sumResult = 0.0;
    int rows = ((Mat)mat).rows;
    int cols = ((Mat)mat).cols;

    CV_Assert((rows == 1) || (cols == 1));

    if(rows==1)
    {
        for(int i=0; i<cols; i++)
        {
            sumResult = sumResult + mat(0,i);
        }
    }
    else
    {
        for(int i=0; i<rows; i++)
        {
            sumResult = sumResult + mat(i,0);
        }
    }

    return sumResult;
}

template<typename _Tp>
Mat_<_Tp> MathHelpers::sum2D(Mat_<_Tp> mat, int type)
{
    Mat_<_Tp> sumResult;
    int rows = ((Mat)mat).rows;
    int cols = ((Mat)mat).cols;

    CV_Assert((rows > 1) && (cols > 1));

    if(type==MATH_BY_ROWS)
    {
        sumResult = Mat_<_Tp>::zeros(rows,1);
        for(int i=0; i<rows; i++)
        {
            sumResult(i,0) = sum1D(mat.row(i));
        }
    }
    else
    {
        sumResult = Mat_<_Tp>::zeros(1,cols);
        for(int i=0; i<cols; i++)
        {
            sumResult(0,i) = sum1D(mat.col(i));
        }
    }

    return sumResult;
}

template<typename _Tp>
Mat_<int> MathHelpers::maxIndex(Mat_<_Tp> mat, map<int, int> labels,
        int type)
{
    Mat_<int> maxResult;
    int rows = ((Mat)mat).rows;
    int cols = ((Mat)mat).cols;
    bool hasLabels = !labels.empty();
    _Tp maxValue;

    CV_Assert((rows > 1) && (cols > 1));

    if(type==MATH_BY_ROWS)
    {
        maxResult = Mat_<int>::zeros(rows,1);
        for(int i=0; i<rows; i++)
        {
            maxValue = mat(i,0);
            maxResult(i,0) = hasLabels ? labels[1]: 1;
            for(int j=1; j<cols; j++)
            {
                if(mat(i,j) > maxValue)
                {
                    maxValue = mat(i,j);
                    maxResult(i,0) = hasLabels ? labels[j+1]: j+1;
                }
            }
        }
    }
    else
    {
        maxResult = Mat_<int>::zeros(1,cols);
        for(int j=0; j<cols; j++)
        {
            maxValue = mat(0,j);
            maxResult(0,j) = hasLabels ? labels[1]: 1;
            for(int i=1; i<rows; i++)
            {
                if(mat(i,j) > maxValue)
                {
                    maxValue = mat(i,j);
                    maxResult(0,j) = hasLabels ? labels[i+1]: i+1;
                }
            }
        }
    }

    return maxResult;
}


template<typename _Tp>
Mat_<_Tp> MathHelpers::meanSubstraction(const Mat_<_Tp>& mat,
        const Mat& mean)
{
    Mat_<_Tp> matCopy(mat.size());
    mat.copyTo(matCopy);
    Mat_<_Tp> meanCopy(mean);
    for (int i=0; i<matCopy.rows; i++)
    {
        matCopy.row(i) = matCopy.row(i) - meanCopy;
    }

    return matCopy;
}

template<typename _Tp>
void MathHelpers::getZeroMeanScoresFromPCA(const Mat_<_Tp>& mat,
		const PCA& pca, Mat_<_Tp>& zeroMeanScores)
{
    zeroMeanScores.create(((Mat)mat).rows, ((Mat)mat).cols);
    mat.copyTo(zeroMeanScores);
    Mat_<_Tp> mean(pca.mean);
    for (int i=0; i<((Mat)mat).rows; i++)
    {
        zeroMeanScores.row(i) = zeroMeanScores.row(i) - mean;
    }

    Mat_<_Tp> coeffs(pca.eigenvectors);
    zeroMeanScores = zeroMeanScores * coeffs.t();
}

template<typename _Tp>
void MathHelpers::pcaReduceData(const Mat_<_Tp>& mat,
		const _Tp variability, Mat_<_Tp>& reducedData,
		Mat_<_Tp>& coefficients)
{
    // We need mat.rows -1 dimmensions as much.
    PCA components(mat, Mat(), CV_PCA_DATA_AS_ROW, ((Mat)mat).rows-1);

    Mat_<_Tp> tmpReducedData;
    Mat_<_Tp> tmpCoefficients;

    getZeroMeanScoresFromPCA(mat, components, tmpReducedData);

    components.eigenvectors.copyTo(tmpCoefficients);
    //TODO: Calculate number of dimmensions needed to achieve certain level of
    // variability.

    Mat_<_Tp> cSum;
    Mat_<_Tp> eigenValues = ((Mat)components.eigenvalues).t();
    MathHelpers::cumSum(eigenValues, cSum);
    _Tp sSum = sum1D(eigenValues);
    Mat_<_Tp> cumVar = cSum / sSum;


    int numDimm = ((Mat)mat).rows-1;

    // Break after finding first value
    for (int i=0; i<((Mat)cumVar).cols; i++)
    {
    	if(cumVar(0,i) >= variability)
    	{
    		numDimm = i+1;
    		break;
    	}
    }

    ((Mat)tmpReducedData(Range(0, ((Mat)tmpReducedData).rows),
            Range(0, numDimm))).copyTo(reducedData);
    ((Mat)tmpCoefficients(Range(0, numDimm),
                Range(0, ((Mat)tmpCoefficients).cols))).copyTo(coefficients);
    coefficients = ((Mat)coefficients).t();
}

}
#endif /* MATHHELPERS_HPP_ */
