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

#ifndef GABORFEATURESET_HPP_
#define GABORFEATURESET_HPP_

// TODO: Check really needed header files, including all OpenCV headers
// is way too much
#include "opencv2/opencv.hpp"
#include "FeatureSet.hpp"

namespace fex {

/*
 * Template class for a Feature Set.
 * Using long types (such as long, long float or long double will fail
 * compiling, since OpenCV mat's structure does not support them.
 *
 * Using int type is not allowed, and will cause a runtime error.
 */
template <typename _Tp> class GaborFeatureSet : public FeatureSet<_Tp> {
public:
	/*
	 * Typedefs
	 */
	typedef _Tp value_type;

	/*
	 * Constructors
	 */
	GaborFeatureSet();
	GaborFeatureSet(FilterSet<_Tp> _filterSet, _Tp _variabilityRate,
	        bool _needZMUNormalization,	bool _needDownSampling,
	        _Tp _downsamplingRatio=1.0f);
	virtual ~GaborFeatureSet();

	/*
	 * Methods
	 */
	void generateFeatureSet(vector<Mat_<_Tp> >& mat);

	/*
	 * Attribute getters
	 */
	Mat_<_Tp> getCoefficients() const;
	Mat_<_Tp> getTrainingData() const;

private:
    /*
     * Attributes
     */
	FilterSet<_Tp> mFilterSet;
	_Tp mVariabilityRate;
	bool mNeedZMUNormalization;
	bool mNeedDownSampling;
	_Tp mDownSamplingRatio;
	Mat_<_Tp> mCoefficients;
	Mat_<_Tp> mTrainingData;

	/*
	 * Methods
	 */
	void init(FilterSet<_Tp> filterSet, _Tp variabilityRate,
			bool needZMUNormalization, bool needDownSampling,
			_Tp downsamplingRatio);

};

template <typename _Tp>
GaborFeatureSet<_Tp>::GaborFeatureSet()
{
}

template <typename _Tp>
GaborFeatureSet<_Tp>::GaborFeatureSet(FilterSet<_Tp> _filterSet,
        _Tp _variabilityRate, bool _needZMUNormalization,
        bool _needDownSampling, _Tp _downsamplingRatio)
{
	init(_filterSet, _needZMUNormalization, _variabilityRate, _needDownSampling,
			_downsamplingRatio);
}


template <typename _Tp>
GaborFeatureSet<_Tp>::~GaborFeatureSet()
{
}

template<typename _Tp>
inline Mat_<_Tp> GaborFeatureSet<_Tp>::getCoefficients() const
{
    return mCoefficients;
}

template<typename _Tp>
inline Mat_<_Tp> GaborFeatureSet<_Tp>::getTrainingData() const
{
    return mTrainingData;
}

template <typename _Tp>
void GaborFeatureSet<_Tp>::generateFeatureSet(vector<Mat_<_Tp> >& mat)
{
    typedef typename vector<Mat_<_Tp> >::iterator vectorMatrix;

	int numFilters = this->mFilterSet.getFilterSet().size();

	int numImages = mat.size();
	int rowFilteredImageSize = ((Mat)mat.front()).cols *
			((Mat)mat.front()).rows * numFilters *
			pow(this->mDownSamplingRatio,2);

	Mat_<_Tp> features(numImages, rowFilteredImageSize);

	vectorMatrix itVec =
			mat.begin(), itVec_end = mat.end();

	Mat_<double> tmpResult;
	int i=0;
	// Iteration over the images to generate the features representing the image
	for(; itVec != itVec_end; ++itVec)
	{
		ImageHelpers::imageApplyFilterSet((*itVec) , this->mFilterSet,
		        tmpResult, this->mNeedZMUNormalization,
		        this->mNeedDownSampling, this->mDownSamplingRatio);

		Mat_<_Tp> tmp = features.row(i);

		((Mat)tmpResult.reshape(1)).copyTo(tmp);

		i++;
	}

	MathHelpers::pcaReduceData(tmpResult, this->mVariabilityRate,
	        this->mTrainingData, this->mCoefficients);
}

template <typename _Tp>
void GaborFeatureSet<_Tp>::init(FilterSet<_Tp> filterSet,
        _Tp variabilityRate, bool needZMUNormalization,
        bool needDownSampling, _Tp downsamplingRatio)
{
	mFilterSet = filterSet;
	mVariabilityRate = variabilityRate;
	mNeedZMUNormalization = needZMUNormalization;
	mNeedDownSampling = needDownSampling;
	mDownSamplingRatio = downsamplingRatio;
}

}

#endif /* GABORFEATURESET_HPP_ */
