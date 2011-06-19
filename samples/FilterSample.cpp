/*************************************************************************** 
 *  Copyright (c) 2011 Javier Moro Sotelo.
 *..
 *  This file is part of libfex.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *..
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *...
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *...
 *  Contributors:
 *      Javier Moro Sotelo - initial API and implementation
 ***************************************************************************/

#include "GaborFilter.hpp"
#include "DebugHelpers.hpp"
#include "MathHelpers.hpp"
#include "opencv2/opencv.hpp"

using namespace fex;
using namespace cv;
using namespace std;

void gaborFilterTest();

int main()
{
    double duration;
    duration = static_cast<double>(cv::getTickCount());
    gaborFilterTest();
    duration = static_cast<double>(cv::getTickCount()) - duration;
    duration /= cv::getTickFrequency();
    cout << "Elapsed time: " << duration << " seconds." << endl;
}

void gaborFilterTest()
{

    GaborFilter<double> filter(0, 0, 120, M_PI/2, 2*M_PI);
    Mat_<Vec2d> f = filter.getFilter();

    DebugHelpers::printMatInformation(f);
    DebugHelpers::printMatrixValues(f, 5, 5, 0, 0);
    Mat_<double> real;
    Mat_<double> imaginary;
    Mat_<double> planes[] = {real, imaginary};
    split(f, planes);
    DebugHelpers::showImage(planes[0]);
}
