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

#ifndef DEBUGHELPERS_HPP_
#define DEBUGHELPERS_HPP_

#include <complex>
#include "opencv2/opencv.hpp"

namespace fex
{

using namespace cv;
using namespace std;

class DebugHelpers
{
public:

    template <typename _Tp>
    static void printMatrixValues(const Mat_<_Tp>& matrix,
            int sizeX, int sizeY, int x, int y);

    template <typename _Tp>
	static void printMatrixValues(const Mat_<complex<_Tp> >& matrix,
			int sizeX, int sizeY, int x, int y);

    template <typename _Tp>
    static void printMatrixValues(const Mat_<Vec<_Tp, 2> >& matrix,
            int sizeX, int sizeY, int x, int y);

    template <typename _Tp, int _numEl>
    static void printVectorValues(const Vec<_Tp, _numEl>& v, int length,
            int start);

    static void printMatInformation(const Mat& matrix);

    template <typename _Tp, int _numEl>
    static void printVecInformation(const Vec<_Tp, _numEl>& v);

    template <typename _Tp>
    static void showImage(const Mat_<_Tp>& matrix);

    static String opencvTypeAsString(int type);
};

template <typename _Tp>
void DebugHelpers::printMatrixValues(const Mat_<_Tp>& matrix,
        int sizeX, int sizeY, int x, int y)
{

    String sign;
    cout.precision(5);
    _Tp value;

    for (int i=x; i<x + sizeX; i++) {
        for (int j=y; j<y + sizeY; j++ ) {
            value = matrix(i,j);
            if(value >= 0) {
                sign = "+";
            }
            else {
                sign="";
            }
            std::cout << scientific << sign << value << " ";
        }
        std::cout << std::endl;
    }
}

template <typename _Tp>
void DebugHelpers::printMatrixValues(const Mat_<complex<_Tp> >& matrix,
        int sizeX, int sizeY, int x, int y)
{

    cout.precision(5);
    complex<_Tp> rValue;

    for (int i=x; i<x + sizeX; i++) {
        for (int j=y; j<y + sizeY; j++ ) {
            std::cout << scientific << matrix(i,j) << " ";
        }
        std::cout << std::endl;
    }
}

template <typename _Tp>
void DebugHelpers::printMatrixValues(const Mat_<Vec<_Tp, 2> >& matrix,
        int sizeX, int sizeY, int x, int y)
{

    Mat_<_Tp> real;
    Mat_<_Tp> imaginary;
    Mat_<_Tp> planes[] = {real, imaginary};
    split(matrix, planes);
    String signReal;
    String signImag;
    _Tp realPart;
    _Tp imagPart;

    cout.precision(5);

    for (int i=x; i<x + sizeX; i++) {
        for (int j=y; j<y + sizeY; j++ ) {
            realPart = planes[0](i,j);
            imagPart = planes[1](i,j);
            if(realPart >= 0) {
                signReal = "+";
            }
            else {
                signReal = "";
            }
            if(imagPart >= 0) {
                signImag = "+";
            }
            else {
                signImag = "";
            }
            std::cout << scientific << signReal <<  realPart
                    << signImag << imagPart << " ";
        }
        std::cout << std::endl;
    }
}

template <typename _Tp, int _numEl>
void DebugHelpers::printVectorValues(const Vec<_Tp, _numEl>& v,
        int length, int start)
{
    String sign;
    cout.precision(5);
    _Tp value;

    for(int i=0; i<start + length; i++)
    {
        value = v[i];
        if(value >= 0) {
            sign = "+";
        }
        else {
            sign="";
        }
        std::cout << scientific << sign << value << " ";
    }
    std::cout << std::endl;
}

template <typename _Tp, int _numEl>
void DebugHelpers::printVecInformation(const Vec<_Tp, _numEl>& v)
{
    std::cout << "Size: " << v.channels << std::endl;
    std::cout << "Type: " << opencvTypeAsString(v.type) << std::endl;
}

template <typename _Tp>
void DebugHelpers::showImage(const Mat_<_Tp>& matrix)
{
	Mat_<_Tp> dstMatrix;
	normalize(matrix, dstMatrix, 0.0, 1.0, cv::NORM_MINMAX);
	imshow("Image", dstMatrix);
	waitKey(0);
}

}


#endif /* DEBUGHELPER_HPP_ */
