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
    static void convolutionComplexFilter(
            Mat_<Vec<_Tp, 2> > complexDFTImage,
            Mat_<Vec<_Tp, 2> > complexDFTFilter, Mat_<Vec<_Tp, 2> >& dst);

    template<typename _Tp>
	static void downSample(Mat_<Vec<_Tp, 2> > image,
			Mat_<Vec<_Tp, 2> >& dst, double ratio, int method=INTER_LANCZOS4);

    template<typename _Tp>
    static void zmuNormalization(Mat_<Vec<_Tp, 2> > image,
    		Mat_<Vec<_Tp, 2> >& dst);
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
void ImageHelpers::convolutionComplexFilter(
        Mat_<Vec<_Tp, 2> > complexDFTImage,
        Mat_<Vec<_Tp, 2> > complexDFTFilter, Mat_<Vec<_Tp, 2> >& dst)
{
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

}

#endif /* IMAHEHELPERS_HPP_ */
