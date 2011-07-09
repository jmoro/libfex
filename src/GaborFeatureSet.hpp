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
#include "FilteringHelpers.hpp"
#include <armadillo>

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
	GaborFeatureSet(GaborSet<_Tp> _filterSet, _Tp _variabilityRate,
	        bool _needZMUNormalization,	bool _needDownSampling,
	        bool _storeRawFeatures=false, _Tp _downsamplingRatio=1.0f);
	virtual ~GaborFeatureSet();

	/*
	 * Methods
	 */
	void generateFeatureSet(vector<Mat_<_Tp> >& mat);
	void projectData(vector<Mat_<_Tp> >& mat, Mat_<_Tp>& dst);
	void reduceRawFeatureSet(double variabilityRate);

	/*
	 * Attribute getters
	 */
	Mat_<_Tp> getCoefficients() const;
	Mat_<_Tp> getTrainingData() const;
	Mat_<_Tp> getFeatures() const;

private:
    /*
     * Attributes
     */
	GaborSet<_Tp> mGaborSet;
	_Tp mVariabilityRate;
	bool mNeedZMUNormalization;
	bool mNeedDownSampling;
	bool mStoreRawFeatures;
	_Tp mDownSamplingRatio;
	Mat_<_Tp> mFeatures;
	Mat_<_Tp> mCoefficients;
	Mat_<_Tp> mTrainingData;

	/*
	 * Methods
	 */
	void init(GaborSet<_Tp> filterSet, _Tp variabilityRate,
			bool needZMUNormalization, bool needDownSampling,
			bool storeRawFeatures, _Tp downsamplingRatio);

};

template <typename _Tp>
GaborFeatureSet<_Tp>::GaborFeatureSet()
{
}

template <typename _Tp>
GaborFeatureSet<_Tp>::GaborFeatureSet(GaborSet<_Tp> _filterSet,
        _Tp _variabilityRate, bool _needZMUNormalization,
        bool _needDownSampling, bool _storeRawFeatures,
        _Tp _downsamplingRatio)
{
	init(_filterSet, _variabilityRate, _needZMUNormalization, _needDownSampling,
			_storeRawFeatures, _downsamplingRatio);
}


template <typename _Tp>
GaborFeatureSet<_Tp>::~GaborFeatureSet()
{
}

template<typename _Tp>
inline Mat_<_Tp> GaborFeatureSet<_Tp>::getCoefficients() const
{
    return (mCoefficients);
}

template<typename _Tp>
inline Mat_<_Tp> GaborFeatureSet<_Tp>::getTrainingData() const
{
    return (mTrainingData);
}

template<typename _Tp>
inline Mat_<_Tp> GaborFeatureSet<_Tp>::getFeatures() const
{
    if(this->mStoreRawFeatures)
    {
        return (mFeatures);
    }
    return (Mat_<_Tp>());
}

template <typename _Tp>
void GaborFeatureSet<_Tp>::generateFeatureSet(
         vector<Mat_<_Tp> >& mat)
{
    Mat_<_Tp> features;

    FilteringHelpers::imageApplyGaborSetToMatVector(mat, this->mGaborSet,
            features, this->mNeedZMUNormalization, this->mNeedDownSampling,
            this->mDownSamplingRatio);

	if(mStoreRawFeatures)
	{
	    ((Mat)features).copyTo(this->mFeatures);
	}

    MathHelpers::pcaReduceData(features, this->mVariabilityRate,
	        this->mTrainingData, this->mCoefficients);
}

template <typename _Tp>
void GaborFeatureSet<_Tp>::projectData(vector<Mat_<_Tp> >& mat,
        Mat_<_Tp>& dst)
{
    Mat_<_Tp> features;

    FilteringHelpers::imageApplyGaborSetToMatVector(mat, this->mGaborSet,
            features, this->mNeedZMUNormalization, this->mNeedDownSampling,
            this->mDownSamplingRatio);

    dst = features * mCoefficients;
}

template <typename _Tp>
void GaborFeatureSet<_Tp>::reduceRawFeatureSet(double variabilityRate)
{
    CV_Assert(this->mStoreRawFeatures);

    this->mVariabilityRate = variabilityRate;

    MathHelpers::pcaReduceData(this->mFeatures, this->mVariabilityRate,
            this->mTrainingData, this->mCoefficients);
}

template <typename _Tp>
void GaborFeatureSet<_Tp>::init(GaborSet<_Tp> filterSet,
        _Tp variabilityRate, bool needZMUNormalization,
        bool needDownSampling, bool storeRawFeatures,
        _Tp downsamplingRatio)
{
	mGaborSet = filterSet;
	mVariabilityRate = variabilityRate;
	mNeedZMUNormalization = needZMUNormalization;
	mNeedDownSampling = needDownSampling;
	mStoreRawFeatures = storeRawFeatures;
	if(!mNeedDownSampling)
	{
	    mDownSamplingRatio = 1;
	}
	else {
	    mDownSamplingRatio = downsamplingRatio;
	}
}

}

#endif /* GABORFEATURESET_HPP_ */
