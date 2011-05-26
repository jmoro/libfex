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

#ifndef IMAGEHELPERS_HPP_
#define IMAGEHELPERS_HPP_

// TODO: Check really needed header files, including all OpenCV headers
// is way too much
#include "opencv2/opencv.hpp"
#include "FilterSet.hpp"
#include "GaborFilter.hpp"
#include <map>

namespace fex
{

using namespace cv;

class ImageHelpers
{
public:

    template<typename _Tp>
    static void complexDFT(Mat_<_Tp> image, Mat_<Vec<_Tp, 2> >& dst);

    template<typename _Tp>
    static void complexDFT(Mat_<complex<_Tp> > image,
    		Mat_<Vec<_Tp, 2> >& dst);

    template<typename _Tp>
    static void magnitudeComplexImage(Mat_<Vec<_Tp, 2> > image,
    		Mat_<_Tp>& dst);

    template<typename _Tp>
    static void convolutionComplexFilter(Mat_<_Tp> image,
    		Mat_<complex<_Tp> > filter, Mat_<Vec<_Tp, 2> >& dst);

    template<typename _Tp>
	static void downSample(Mat_<Vec<_Tp, 2> > image,
			Mat_<Vec<_Tp, 2> >& dst, double ratio, int method=INTER_LANCZOS4);

    template<typename _Tp>
    static void zmuNormalization(Mat_<Vec<_Tp, 2> > image,
    		Mat_<Vec<_Tp, 2> >& dst);

    template<typename _Tp>
    static void imageApplyFilterSet(Mat_<_Tp> image,
    		FilterSet<_Tp> filterSet, Mat_<_Tp>& dst,
    		bool needZMUNorm, bool needDownSampl, _Tp ratio=1.0f);

};

template<typename _Tp>
void ImageHelpers::complexDFT(Mat_<_Tp> image, Mat_<Vec<_Tp, 2> >& dst)
{
    int M = getOptimalDFTSize(image.rows);
    int N = getOptimalDFTSize(image.cols);

    Mat_<_Tp> padded;
    copyMakeBorder(image, padded, 0, M - image.rows, 0, N - image.cols,
            BORDER_CONSTANT, Scalar::all(0));

    Mat_<_Tp> planes[] = {padded, Mat::zeros(padded.size(), padded.type())};
    Mat_<Vec<_Tp, 2> > complexImg(M,N);

    merge(planes, 2, complexImg);
    dft(complexImg, dst, DFT_COMPLEX_OUTPUT);
}

template<typename _Tp>
void ImageHelpers::complexDFT(Mat_<complex<_Tp> > image,
		Mat_<Vec<_Tp, 2> >& dst)
{
    int M = getOptimalDFTSize(image.rows);
    int N = getOptimalDFTSize(image.cols);

    Mat_<complex<_Tp> > padded;

    copyMakeBorder(image, padded, 0, M - image.rows, 0, N - image.cols,
            BORDER_CONSTANT, Scalar::all(0));

    dft(padded, dst, DFT_COMPLEX_OUTPUT);
}

template<typename _Tp>
void ImageHelpers::magnitudeComplexImage(Mat_<Vec<_Tp, 2> > image,
		Mat_<_Tp>& dst)
{
    Mat_<_Tp> real;
    Mat_<_Tp> imaginary;
    Mat_<_Tp> planes[] = {real, imaginary};
    split(image, planes);
    magnitude(planes[0], planes[1], dst);
}

template<typename _Tp>
void ImageHelpers::convolutionComplexFilter(Mat_<_Tp> image,
		Mat_<complex<_Tp> > filter, Mat_<Vec<_Tp, 2> >& dst)
{
	Mat_<Vec<_Tp, 2> > complexDFTImage;
	Mat_<Vec<_Tp, 2> > complexDFTFilter;
    complexDFT(image, complexDFTImage);
    complexDFT(filter, complexDFTFilter);

    Mat_<Vec<_Tp, 2> >
    spectrum(complexDFTImage.size(), complexDFTImage.type());
    cv::mulSpectrums(complexDFTImage, complexDFTFilter, spectrum,
    		DFT_COMPLEX_OUTPUT);

    idft(spectrum, dst, DFT_COMPLEX_OUTPUT + DFT_SCALE);
}

template<typename _Tp>
void ImageHelpers::downSample(Mat_<Vec<_Tp, 2> >image,
		Mat_<Vec<_Tp, 2> >& dst, double ratio, int method)
{
	Mat_<_Tp> real;
	Mat_<_Tp> imaginary;
	Mat_<_Tp> planes[] = {real, imaginary};
	split(image, planes);

	Mat_<_Tp> realResized;
	Mat_<_Tp> imaginaryResized;


	resize(planes[0], realResized, Size(), ratio, ratio);
	resize(planes[1], imaginaryResized, Size(), ratio, ratio);

	Mat_<_Tp> planesResized[] = {realResized, imaginaryResized};

	merge(planesResized, 2, dst);
}

template<typename _Tp>
void ImageHelpers::zmuNormalization(Mat_<Vec<_Tp, 2> > image,
		Mat_<Vec<_Tp, 2> >& dst)
{
    Scalar_<_Tp> mean;
    Scalar_<_Tp> std;

    meanStdDev(image, mean, std);

    Mat meanMatrix(image.size(), image.type(), mean);
    Mat stdMatrix(image.size(), image.type(), std);

    // FIXME: Not working, fix needed.
    dst = (image - meanMatrix)/stdMatrix;
}


template<typename _Tp>
void ImageHelpers::imageApplyFilterSet(Mat_<_Tp> image,
		FilterSet<_Tp> filterSet, Mat_<_Tp>& dst,
		bool needZMUNorm, bool needDownSampl, _Tp ratio)
{

	// Move this typedef somewhere else?
	typedef typename map<pair<int, int>, GaborFilter_<_Tp> >::iterator
			mapPairGaborIter;

	map<pair<int, int>, GaborFilter_<_Tp> > filters =
			filterSet.getFilterSet();

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
	GaborFilter_<_Tp> filter;
	Mat_<complex<_Tp> > f;
	Mat_<Vec<_Tp, 2> > tmpResult;
	Mat_<Vec<_Tp, 2> > normalizedImage;
	Mat_<_Tp> features;
	for(; itMap != itMap_end; ++itMap)
	{
		filter = (*itMap).second;
		f = filter.getFilter();
		convolutionComplexFilter(image, f, tmpResult);
		if(needDownSampl)
		{
			downSample(tmpResult, tmpResult, ratio);
		}
		if(needZMUNorm)
		{
			zmuNormalization(tmpResult, normalizedImage);
			magnitudeComplexImage(normalizedImage, features);
		}
		else {
			magnitudeComplexImage(tmpResult, features);
		}
		Mat_<_Tp> tmp = dst.row(i);

		((Mat)features.reshape(1)).copyTo(tmp);
		i++;
	}

}

}

#endif /* IMAHEHELPERS_HPP_ */
