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

#include "../config.h"

#include "MathHelpers.hpp"
#include "DebugHelpers.hpp"
#include "opencv2/opencv.hpp"

using namespace fex;
using namespace cv;
using namespace std;

void pcaReduceDataTest();
Mat_<double> magic5();
void sum2DTest();

int main()
{
	sum2DTest();
    return (0);
}

void pcaReduceDataTest()
{
	Mat_<double> m5 = magic5();
	double var = 1.0;
	Mat_<double> data;
	Mat_<double> coeff;

	MathHelpers::pcaReduceData(m5, var, data, coeff);

	DebugHelpers::printMatInformation(data);
	DebugHelpers::printMatrixValues(data, 5, 4, 0, 0);
	DebugHelpers::printMatInformation(coeff);
	DebugHelpers::printMatrixValues(coeff, 5, 4, 0, 0);

}

void sum2DTest()
{
	Mat_<double> m5 = magic5();
	double result;
	MathHelpers::sum2D(m5, result);

	cout << "Result: " << result << endl;

}

Mat_<double> magic5()
{
    Mat_<double> M(5,5);
    M(0,0) = 17.0;
    M(0,1) = 24.0;
    M(0,2) = 1.0;
    M(0,3) = 8.0;
    M(0,4) = 15.0;
    M(1,0) = 23.0;
    M(1,1) = 5.0;
    M(1,2) = 7.0;
    M(1,3) = 14.0;
    M(1,4) = 16.0;
    M(2,0) = 4.0;
    M(2,1) = 6.0;
    M(2,2) = 13.0;
    M(2,3) = 20.0;
    M(2,4) = 22.0;
    M(3,0) = 10.0;
    M(3,1) = 12.0;
    M(3,2) = 19.0;
    M(3,3) = 21.0;
    M(3,4) = 3.0;
    M(4,0) = 11.0;
    M(4,1) = 18.0;
    M(4,2) = 25.0;
    M(4,3) = 2.0;
    M(4,4) = 9.0;

    return (M);
}

