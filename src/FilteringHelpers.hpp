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

template<typename _Tp>
void FilteringHelpers::imageApplyGaborSetToMatVector(
        vector<Mat_<_Tp> >& mat, const GaborSet<_Tp> filterSet,
        Mat_<_Tp>& features, bool needZMUNormalization,
        bool needDownSampling, _Tp downSamplingRatio)
{
    typedef typename vector<Mat_<_Tp> >::iterator vectorMatrix;

    int numFilters = filterSet.getGaborSet().size();

    int numImages = mat.size();

    int rowFilteredImageSize = ((Mat)mat.front()).cols *
           ((Mat)mat.front()).rows * numFilters *
           pow(downSamplingRatio,2);

    features.create(numImages, rowFilteredImageSize);

    vectorMatrix itVec =
           mat.begin(), itVec_end = mat.end();

    Mat_<double> tmpResult;

    int i=0;
    // Iteration over the images to generate the features representing the image
    for(; itVec != itVec_end; ++itVec)
    {

       FilteringHelpers::imageApplyGaborSet((*itVec) , filterSet,
               tmpResult, needZMUNormalization,
               needDownSampling, downSamplingRatio);

       Mat_<_Tp> tmp = features.row(i);

       ((Mat)tmpResult.reshape(1)).copyTo(tmp);

       i++;
    }
}

template<typename _Tp>
void FilteringHelpers::imageApplyGaborSet(Mat_<_Tp> image,
        GaborSet<_Tp> filterSet, Mat_<_Tp>& dst,
        bool needZMUNorm, bool needDownSampl, _Tp ratio)
{

    // Move this typedef somewhere else?
    typedef typename map<pair<int, int>, GaborFilter<_Tp> >::iterator
            mapPairGaborIter;

    map<pair<int, int>, GaborFilter<_Tp> > filters =
            filterSet.getGaborSet();

    int numFilters = filters.size();
    int rowFilteredImageSize = (((Mat)image).rows)*
            (((Mat)image).cols);
    if(needDownSampl)
    {
        rowFilteredImageSize *= pow(ratio,2);
    }


    dst.create(numFilters, rowFilteredImageSize);

    mapPairGaborIter itMap = filters.begin(), itMap_end = filters.end();

    int i=0;
    GaborFilter<_Tp> filter;
    Mat_<Vec<_Tp, 2> > imageFFT;
    ImageHelpers::complexDFT(image, imageFFT);
    Mat_<Vec<_Tp, 2> > fFFT;
    Mat_<Vec<_Tp, 2> > tmpResult;
    Mat_<Vec<_Tp, 2> > normalizedImage;
    Mat_<_Tp> features;
    for(; itMap != itMap_end; ++itMap)
    {
        filter = (*itMap).second;
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
        i++;
    }

}


}

#endif /* FILTERINGHELPERS_HPP_ */
