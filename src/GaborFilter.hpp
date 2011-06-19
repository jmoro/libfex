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

#ifndef GABORFILTER_HPP_
#define GABORFILTER_HPP_

// TODO: Check really needed header files, including all OpenCV headers
// is way too much
#include "ImageHelpers.hpp"
#include "opencv2/opencv.hpp"
// TODO: Remove this dependence with typeinfo here, create an utility function.
#include <typeinfo>

namespace fex
{

using namespace cv;

/*
 * Template class for Gabor Filter.
 * Using long types (such as long, long float or long double will fail
 * compiling, since OpenCV mat's structure does not support them.
 *
 * Using int type is not allowed, and will cause a runtime error.
 */
template<typename _Tp> class GaborFilter
{
public:

    /*
     * Typedefs
     */
    typedef _Tp value_type;

    /*
     * Constructors
     */
    GaborFilter();
    GaborFilter(int _scale, int _orientation, int _filterSize,
            _Tp _kMax, _Tp _sigma);
    GaborFilter(int _scale, int _orientation, int _filterSizeX,
            int _filterSizeY, _Tp _kMax, _Tp _sigma);
    // TODO: New types of constructors, allow to create a filter from a
    // Mat object, a IPLImage, etc...
    virtual ~GaborFilter();

    /*
     * Attribute getters
     */
    int getScale() const;
    int getOrientation() const;
    int getFilterSizeX() const;
    int getFilterSizeY() const;
    _Tp getKMax() const;
    _Tp getSigma() const;
    Mat_<complex<_Tp> > getFilter() const;
    Mat_<Vec<_Tp, 2> > getFilterFFT() const;

private:
    /*
     * Attributes
     */
    int mScale;
    int mOrientation;
    int mFilterSizeX;
    int mFilterSizeY;
    _Tp mKMax;
    _Tp mSigma;
    Mat_<complex<_Tp> > mFilter;
    Mat_<Vec<_Tp, 2> > mFilterFFT;


    /*
     * Private functions
     */
    inline void checkForFloatingPoint() {
        double doubleType;
        float floatType;
        value_type myType;
        CV_Assert((typeid(floatType) == typeid(myType)) ||
                (typeid(doubleType) == typeid(myType)));
    }

    // Since nested constructor calling is not allowed in C++, we define an
    // init function to refactor code.
    void init(Mat_<complex<_Tp> >& result,
            Mat_<Vec<_Tp, 2> >& filterFFT, int scale, int orientation,
            int filterSizeX, int filterSizeY, _Tp kMax, _Tp sigma);

    void generateFilter(Mat_<complex<_Tp> >& result, int scale,
            int orientation, int filterSizeX, int filterSizeY, _Tp kMax,
            _Tp sigma);

};

/******************************************************************************
 ******************************************************************************
 **                          CLASS IMPLEMENTATION                            **
 ******************************************************************************
 ******************************************************************************/

/*
 * Since we are using templates, we need to write the implementation on the
 * header file
 */

/**************
 * Constructors
 **************/
template<typename _Tp> GaborFilter<_Tp>::GaborFilter()
{
    GaborFilter<_Tp>::checkForFloatingPoint();
}

template<typename _Tp> GaborFilter<_Tp>::GaborFilter(int _scale,
        int _orientation, int _filterSize, _Tp _kMax, _Tp _sigma)
{
    init(mFilter, mFilterFFT, _scale, _orientation, _filterSize, _filterSize,
            _kMax, _sigma);
}

template<typename _Tp> GaborFilter<_Tp>::GaborFilter(int _scale,
        int _orientation, int _filterSizeX, int _filterSizeY, _Tp _kMax,
        _Tp _sigma)
{
    init(mFilter, mFilterFFT, _scale, _orientation, _filterSizeX, _filterSizeY,
        _kMax, _sigma);
}

template<typename _Tp> GaborFilter<_Tp>::~GaborFilter()
{
    // Nothing to see here... keep walking!!!
}

/*******************
 * Attribute getters
 *******************/
template<typename _Tp>
inline int GaborFilter<_Tp>::getScale() const
{
    return mScale;
}

template<typename _Tp>
inline int GaborFilter<_Tp>::getOrientation() const
{
    return mOrientation;
}

template<typename _Tp>
inline int GaborFilter<_Tp>::getFilterSizeX() const
{
    return mFilterSizeX;
}

template<typename _Tp>
inline int GaborFilter<_Tp>::getFilterSizeY() const
{
    return mFilterSizeY;
}

template<typename _Tp>
inline _Tp GaborFilter<_Tp>::getKMax() const
{
    return mKMax;
}

template<typename _Tp>
inline _Tp GaborFilter<_Tp>::getSigma() const
{
    return mSigma;
}

template<typename _Tp> inline Mat_<complex<_Tp> >
GaborFilter<_Tp>::getFilter() const
{
    return mFilter;
}

template<typename _Tp> inline Mat_<Vec<_Tp, 2> >
GaborFilter<_Tp>::getFilterFFT() const
{
    return mFilterFFT;
}

/*******************
 * Private functions
 *******************/
template<typename _Tp>
void GaborFilter<_Tp>::init(Mat_<complex<_Tp> > & filter,
        Mat_<Vec<_Tp, 2> >& filterFFT, int scale, int orientation,
        int filterSizeX, int filterSizeY, _Tp kMax, _Tp sigma)
{
    GaborFilter<_Tp>::checkForFloatingPoint();

    mScale = scale;
    mOrientation = orientation;
    mFilterSizeX = filterSizeX;
    mFilterSizeY = filterSizeY;
    mKMax = kMax;
    mSigma = sigma;
    // We generate the filter
    generateFilter(mFilter, mScale, mOrientation, mFilterSizeX,
            mFilterSizeY, mKMax, mSigma);
    // We generate it's dft
    ImageHelpers::complexDFT(this->mFilter, this->mFilterFFT);
}

template<typename _Tp>
void GaborFilter<_Tp>::generateFilter(Mat_<complex<_Tp> > & result,
        int scale, int orientation, int filterSizeX, int filterSizeY,
        _Tp kMax, _Tp sigma)
{
    const complex<_Tp> i = sqrt (complex<_Tp>(-1));

    _Tp offsetX = filterSizeX/2.0;
    _Tp offsetY = filterSizeY/2.0;
    _Tp psi = ((orientation*M_PI)/8);
    _Tp f = sqrt(2);
    _Tp fV = pow(f, scale);
    _Tp kReal = (kMax/fV)*cos(psi);
    _Tp kImag = (kMax/fV)*sin(psi);
    _Tp kSquare = pow(sqrt(pow(kReal, 2) + pow(kImag, 2)), 2);
    _Tp sSquare = pow(sigma,2);
    _Tp sSquareHalf = -0.5 * sSquare;
    _Tp kS = kSquare/sSquare;
    _Tp kSHalf = -0.5f * kSquare / sSquare;

    _Tp magnitude;
    _Tp commonPart;
    _Tp realPart;
    _Tp imagPart;
    _Tp offsetXVal;
    _Tp offsetYVal;
    complex<_Tp> complexTempResult;
    complex<_Tp> complexResult;

    Mat_<_Tp> realWavelet(filterSizeX, filterSizeY);
    Mat_<_Tp> imagWavelet(filterSizeX, filterSizeY);

    result.create(filterSizeX, filterSizeY);

    int numberColumns = filterSizeY*filterSizeX;
    int numberLines = 1;

    complex<_Tp>* data = ((Mat)result).ptr<complex<_Tp> >(0);

    for (int index=0; index<numberColumns; index++)
    {
        offsetXVal = (index/filterSizeY) - offsetX;
        offsetYVal = (index%filterSizeY) - offsetY;

        magnitude = (offsetXVal)*(offsetXVal) + (offsetYVal)*(offsetYVal);

        commonPart = kS * exp(kSHalf * (magnitude));

        complexTempResult =
                (exp(1.0*i * ((kReal * offsetYVal) + (kImag * offsetXVal)))
                - exp(-0.5 * sSquare));

        complexResult = commonPart * complexTempResult;

        *data++ = complexResult;
    }
}

}

#endif /* GABORFILTER_HPP_ */
