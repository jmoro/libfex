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

#include "GaborSet.hpp"
#include "GaborFilter.hpp"
#include "DebugHelpers.hpp"
#include "opencv2/opencv.hpp"

using namespace fex;
using namespace cv;
using namespace std;

void gaborFilterSetTest();

int main()
{
    gaborFilterSetTest();
    return (0);
}

void gaborFilterSetTest()
{
	double duration;
	duration = static_cast<double>(cv::getTickCount());
	GaborSet<double> filterSet(5, 8, 120, M_PI/2, 2*M_PI, true);
    duration = static_cast<double>(cv::getTickCount()) - duration;
	duration /= cv::getTickFrequency();
	cout << "Elapsed time: " << duration << " seconds." << endl;
}
