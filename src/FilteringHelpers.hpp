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

#ifndef FILTERINGHELPERS_HPP_
#define FILTERINGHELPERS_HPP_

// TODO: Check really needed header files, including all OpenCV headers
// is way too much
#include "opencv2/opencv.hpp"
#include "opencv2/core/internal.hpp"
#include "ImageHelpers.hpp"
#include "GaborSet.hpp"
#include "GaborFilter.hpp"
#include <vector>
#include <map>

namespace fex
{

class FilteringHelpers
{
public:

    template<typename _Tp>
    static void imageApplyGaborSetToMatVector(
            vector<Mat_<_Tp> >& mat, const GaborSet<_Tp> filterSet,
            Mat_<_Tp>& features, bool needZMUNormalization,
            bool needDownSampling, _Tp downSamplingRatio = 1.0f);

    template<typename _Tp>
    static void imageApplyGaborSet(Mat_<_Tp> image,
            GaborSet<_Tp> filterSet, Mat_<_Tp>& dst,
            bool needZMUNorm, bool needDownSampl, _Tp ratio=1.0f);

};

/*
 ==============================================================================
 ==============================================================================
 ==                           ApplyFilterSetBody                             ==
 ==============================================================================
 ==============================================================================
 */

/*
 * Template class for parallel filtering
 */
template<typename _Tp>
class ApplyFilterSetBody
{
public:

	/*
	 * Constructor
	 */
	ApplyFilterSetBody(int _numFilters, int _rowFilteredImageSize,
			GaborSet<_Tp> _filterSet, bool _needZMUNormalization,
			bool _needDownSampling, _Tp _downSamplingRatio,
			vector<Mat_<_Tp> > _input, Mat_<_Tp> _output);

	/*
	 * TBB operator
	 */
	void operator() (const BlockedRange& range ) const;

private:

	/*
	 * Input and output arguments
	 */
	vector<Mat_<_Tp> > input;
	Mat_<_Tp> output;

	/*
	 * Arguments needed for computation
	 */
	int mNumFilters;
	int mRowFilteredImageSize;
	GaborSet<_Tp> mFilterSet;
	bool mNeedZMUNormalization;
	bool mNeedDownSampling;
	_Tp mDownSamplingRatio;
};

/******************************************************************************
 ******************************************************************************
 **                          CLASS IMPLEMENTATION                            **
 ******************************************************************************
 ******************************************************************************/

/*************
 * Constructor
 *************/
template<typename _Tp>
ApplyFilterSetBody<_Tp>::ApplyFilterSetBody(int _numFilters,
		int _rowFilteredImageSize,	GaborSet<_Tp> _filterSet,
		bool _needZMUNormalization,	bool _needDownSampling,
		_Tp _downSamplingRatio,	vector<Mat_<_Tp> > _input, Mat_<_Tp> _output) :
		mNumFilters(_numFilters), mRowFilteredImageSize(_rowFilteredImageSize),
		mFilterSet(_filterSet), mNeedZMUNormalization(_needZMUNormalization),
		mNeedDownSampling(_needDownSampling),
		mDownSamplingRatio(_downSamplingRatio), input(_input),
		output(_output) {}

/**************
 * TBB Operator
 **************/
template<typename _Tp>
void ApplyFilterSetBody<_Tp>::operator() (
		const BlockedRange& range ) const
{
	Mat_<double> tmpResult;

	for( int index=range.begin(); index!=range.end( ); ++index )
	{
		FilteringHelpers::imageApplyGaborSet(input[index] , mFilterSet,
			   tmpResult, mNeedZMUNormalization,
			   mNeedDownSampling, mDownSamplingRatio);

	   Mat_<_Tp> tmp = output.row(index);

	   ((Mat)tmpResult.reshape(1)).copyTo(tmp);
	}

}

template<typename _Tp>
void FilteringHelpers::imageApplyGaborSetToMatVector(
        vector<Mat_<_Tp> >& mat, const GaborSet<_Tp> filterSet,
        Mat_<_Tp>& features, bool needZMUNormalization,
        bool needDownSampling, _Tp downSamplingRatio)
{

    int numFilters = filterSet.getScales() * filterSet.getOrientations();

    int numImages = mat.size();

    int rowFilteredImageSize = ((Mat)mat.front()).cols *
           ((Mat)mat.front()).rows * numFilters *
           pow(downSamplingRatio,2);

    features.create(numImages, rowFilteredImageSize);

    ApplyFilterSetBody<_Tp> applyFilterSetBody(numFilters,
    		rowFilteredImageSize, filterSet, needZMUNormalization,
    		needDownSampling, downSamplingRatio, mat, features);

    parallel_for(BlockedRange(0, numImages), applyFilterSetBody);
}

template<typename _Tp>
void FilteringHelpers::imageApplyGaborSet(Mat_<_Tp> image,
        GaborSet<_Tp> filterSet, Mat_<_Tp>& dst,
        bool needZMUNorm, bool needDownSampl, _Tp ratio)
{

    GaborFilter<_Tp>* filters = filterSet.getGaborSet();

    int numFilters = filterSet.getScales() * filterSet.getOrientations();
    int rowFilteredImageSize = (((Mat)image).rows)*
            (((Mat)image).cols);
    if(needDownSampl)
    {
        rowFilteredImageSize *= pow(ratio,2);
    }

    dst.create(numFilters, rowFilteredImageSize);

    GaborFilter<_Tp> filter;
    Mat_<Vec<_Tp, 2> > imageFFT;
    ImageHelpers::complexDFT(image, imageFFT);
    Mat_<Vec<_Tp, 2> > fFFT;
    Mat_<Vec<_Tp, 2> > tmpResult;
    Mat_<Vec<_Tp, 2> > normalizedImage;
    Mat_<_Tp> features;
    for(int i=0; i< numFilters; i++)
    {
        filter = filters[i];
        fFFT = filter.getFilterFFT();
        ImageHelpers::convolutionComplexFilter(imageFFT, fFFT, tmpResult);
        if(needDownSampl)
        {
            ImageHelpers::downSample(tmpResult, tmpResult, ratio);
        }
        if(needZMUNorm)
        {
            ImageHelpers::zmuNormalization(tmpResult, normalizedImage);
            ImageHelpers::magnitudeComplexImage(normalizedImage, features);
        }
        else {
            ImageHelpers::magnitudeComplexImage(tmpResult, features);
        }
        Mat_<_Tp> tmp = dst.row(i);

        ((Mat)features.reshape(1)).copyTo(tmp);
    }

}


}

#endif /* FILTERINGHELPERS_HPP_ */
