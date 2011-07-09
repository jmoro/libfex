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

#include "opencv2/opencv.hpp"
// TODO: Check really needed header files, including all OpenCV headers
// is way too much
#include <cmath>
#include <map>
#include <armadillo>
#include "DebugHelpers.hpp"

namespace fex
{

using namespace cv;

class MathHelpers
{
public:

    const static int MATH_BY_ROWS = 2;
    const static int MATH_BY_COLS = 4;
    const static bool MATH_ECON_MODE_ON = true;
    const static bool MATH_ECON_MODE_OFF = false;

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
    static void stdMean(const Mat_<_Tp> mat, _Tp& std, _Tp& mean);

    template<typename _Tp>
	static void sum1D(const Mat_<_Tp>& mat, _Tp& dst);

	template<typename _Tp>
	static void sum2D(const Mat_<_Tp>& mat, _Tp& dst);

    template<typename _Tp>
	static void maxIndex(const Mat_<_Tp>& mat,
    		map<int, int>& labels, Mat_<int>& dst,
    		int type = MATH_BY_ROWS);

    // TODO: Mow we only accept a mean matrix as a mean matrix for column
    // dimmension, we sould consider also this method for rows.
    template<typename _Tp>
    static void meanSubstraction(const Mat_<_Tp>& mat,
            const Mat_<_Tp>& mean, Mat_<_Tp>& dst, int type = MATH_BY_ROWS);

    template<typename _Tp>
    static void meanNormalize(const Mat_<_Tp>& mat,
            Mat_<_Tp>& meanNormalizedMat, int flags);

    template<typename _Tp>
    static void getZeroMeanScoresFromPCA(const Mat_<_Tp>& mat,
    		const PCA& pca, Mat_<_Tp>& zeroMeanScores);

    template<typename _Tp>
    static void pcaReduceData(const Mat_<_Tp>& mat, const _Tp variability,
            Mat_<_Tp>& reducedData, Mat_<_Tp>& coefficients);

    /*
    template<typename _Tp>
	static void pcaReduceDataArma(const Mat_<_Tp>& mat,
			const _Tp variability,	Mat_<_Tp>& reducedData,
			Mat_<_Tp>& coefficients);
	*/
};

// TODO: Add flags to avoid Q calculation if not needed, and to work in
// econ mode.
// FIXME: get rid of near 0 values
template<typename _Tp>
void MathHelpers::QR(const Mat_<_Tp>& A, Mat_<_Tp>& Q, Mat_<_Tp>& R,
        bool econ)
{
    int rows = (Mat(A)).rows;
    int cols = (Mat(A)).cols;
    Mat_<_Tp> At = A.t();

    arma::mat AArma(((cv::Mat)At).ptr<_Tp>(0), rows, cols, false);

    arma::mat QArma, RArma;

    arma::qr(QArma, RArma, AArma);
    Q = Mat_<_Tp>(QArma.n_cols, QArma.n_rows, QArma.memptr()).t();
    Mat_<_Tp> Rtmp = Mat_<_Tp>(RArma.n_cols, RArma.n_rows, RArma.begin()).t();

    if(econ)
    {
        ((Mat)Rtmp(Range(0, cols), Range(0, cols))).copyTo(R);

    }
    else {
        ((Mat)Rtmp).copyTo(R);
    }
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
void MathHelpers::stdMean(const Mat_<_Tp> mat, _Tp& std, _Tp& mean)
{
    int rows = ((Mat)mat).rows;
    int cols = ((Mat)mat).cols;
    int elems = ((Mat)mat).rows * ((Mat)mat).cols;

    _Tp* data = ((Mat)mat).ptr<_Tp>(0);

    for(int index=0; index<elems; index++)
    {
        mean += *data++;
    }

    mean /= elems;

    data = ((Mat)mat).ptr<_Tp>(0);

    for(int index=0; index<elems; index++)
    {
        std += pow((mean - *data++), 2);
    }

    std = sqrt(std/(elems - 1));
}

template<typename _Tp>
void MathHelpers::sum1D(const Mat_<_Tp>& mat, _Tp& dst)
{
	CV_Assert((mat.cols == 1) || (mat.rows == 1));

	Mat_<_Tp> tmpDst;

	int type = (mat.cols == 1) ? 0 : 1;

	reduce(mat, tmpDst, type, CV_REDUCE_SUM);

	dst = tmpDst(0,0);
}

template<typename _Tp>
void MathHelpers::sum2D(const Mat_<_Tp>& mat, _Tp& dst)
{
	Mat_<_Tp> tmpDst1;
	Mat_<_Tp> tmpDst2;

	reduce(mat, tmpDst1, 0, CV_REDUCE_SUM);
	reduce(tmpDst1, tmpDst2, 1, CV_REDUCE_SUM);

	dst = tmpDst2(0,0);
}

template<typename _Tp>
void MathHelpers::maxIndex(const Mat_<_Tp>& mat,
		map<int, int>& labels, Mat_<int>& dst, int type)
{
	int rows = ((Mat)mat).rows;
	int cols = ((Mat)mat).cols;
	bool hasLabels = !labels.empty();
	_Tp maxValue;

	CV_Assert((rows > 1) && (cols > 1));

	if(type==MATH_BY_ROWS)
	{
		dst = Mat_<int>::zeros(rows,1);

		for(int i=0; i<rows; i++)
		{
			maxValue = mat(i,0);
			dst(i,0) = hasLabels ? labels[1]: 1;
			for(int j=1; j<cols; j++)
			{
				if(mat(i,j) > maxValue)
				{
					maxValue = mat(i,j);
					dst(i,0) = hasLabels ? labels[j+1]: j+1;
				}
			}
		}
	}
	else
	{
		dst = Mat_<int>::zeros(1,cols);
		for(int j=0; j<cols; j++)
		{
			maxValue = mat(0,j);
			dst(0,j) = hasLabels ? labels[1]: 1;
			for(int i=1; i<rows; i++)
			{
				if(mat(i,j) > maxValue)
				{
					maxValue = mat(i,j);
					dst(0,j) = hasLabels ? labels[i+1]: i+1;
				}
			}
		}
	}

}


template<typename _Tp>
void MathHelpers::meanSubstraction(const Mat_<_Tp>& mat,
        const Mat_<_Tp>& mean, Mat_<_Tp>& dst, int type)
{
    mat.copyTo(dst);
    if(type==MATH_BY_ROWS)
	{
    	CV_Assert(mean.rows == 1);

		for (int i=0; i<dst.rows; i++)
		{
			dst.row(i) = dst.row(i) - mean;
		}
	}
    else
    {
    	CV_Assert(mean.cols == 1);

    	for (int i=0; i<dst.rows; i++)
		{
			dst.col(i) = dst.col(i) - mean;
		}
    }
}

template<typename _Tp>
void MathHelpers::meanNormalize(const Mat_<_Tp>& mat,
        Mat_<_Tp>& meanNormalizedMat, int flags)
{
    int takeRows =  (flags & MATH_BY_ROWS) != 0;
    Mat_<_Tp> mean;
    reduce(mat, mean, takeRows ? 0 : 1, CV_REDUCE_AVG, mat.type());

    meanSubstraction(mat, mean, meanNormalizedMat);
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

    Mat_<_Tp> scores;
    Mat_<_Tp> tmpCoefficients;

    getZeroMeanScoresFromPCA(mat, components, scores);

    components.eigenvectors.copyTo(tmpCoefficients);

    //TODO: Calculate number of dimensions needed to achieve certain level of
    // variability.
    Mat_<_Tp> cSum;
    Mat_<_Tp> eigenValues = ((Mat)components.eigenvalues).t();
    MathHelpers::cumSum(eigenValues, cSum);

    _Tp sSum;
    sum1D(eigenValues, sSum);

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

    ((Mat)scores(Range(0, ((Mat)scores).rows),
            Range(0, numDimm))).copyTo(reducedData);
    ((Mat)tmpCoefficients(Range(0, numDimm),
            Range(0, ((Mat)tmpCoefficients).cols))).copyTo(coefficients);
    coefficients = ((Mat)coefficients).t();
}

/*
template<typename _Tp>
void MathHelpers::pcaReduceDataArma(const Mat_<_Tp>& mat,
		const _Tp variability, Mat_<_Tp>& reducedData,
		Mat_<_Tp>& coefficients)
{
    // We need mat.rows -1 dimmensions as much.

	Mat_<_Tp> scores;
	Mat_<_Tp> tmpCoefficients;
	Mat_<_Tp> eigenValues;
	Mat_<_Tp> cSum;

	double durationStep;

	cout << "Data conversion: ";
	durationStep = static_cast<double>(cv::getTickCount());

	int rows = (Mat(mat)).rows;
	int cols = (Mat(mat)).cols;
	Mat_<_Tp> matT = mat.t();

	arma::mat matArma(((cv::Mat)matT).ptr<_Tp>(0), rows, cols, false);

	durationStep = static_cast<double>(cv::getTickCount()) - durationStep;
	durationStep /= cv::getTickFrequency();
	cout << durationStep << " seconds." << endl;

	arma::mat score;
	arma::mat coeff;
	arma::vec latent;

	cout << "PCA: ";
	durationStep = static_cast<double>(cv::getTickCount());

	arma::princomp(coeff, score, latent, matArma);

    durationStep = static_cast<double>(cv::getTickCount()) - durationStep;
	durationStep /= cv::getTickFrequency();
	cout << durationStep << " seconds." << endl;

	cout << "Data conversion: ";
	durationStep = static_cast<double>(cv::getTickCount());

	scores = Mat_<_Tp>(score.n_cols, score.n_rows, score.memptr()).t();
	tmpCoefficients = Mat_<_Tp>(coeff.n_rows, coeff.n_cols, coeff.memptr());
	eigenValues = Mat_<_Tp>(latent.n_cols, coeff.n_rows, latent.memptr()).t();

	durationStep = static_cast<double>(cv::getTickCount()) - durationStep;
	durationStep /= cv::getTickFrequency();
	cout << durationStep << " seconds." << endl;

    cout << "Reduction	: ";
	durationStep = static_cast<double>(cv::getTickCount());

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

    ((Mat)scores(Range(0, ((Mat)scores).rows),
            Range(0, numDimm))).copyTo(reducedData);
    ((Mat)tmpCoefficients(Range(0, numDimm),
            Range(0, ((Mat)tmpCoefficients).cols))).copyTo(coefficients);
    coefficients = ((Mat)coefficients).t();
    durationStep = static_cast<double>(cv::getTickCount()) - durationStep;
	durationStep /= cv::getTickFrequency();
	cout << durationStep << " seconds." << endl;
}
*/

}
#endif /* MATHHELPERS_HPP_ */
