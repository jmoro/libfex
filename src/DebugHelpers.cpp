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

#include "DebugHelpers.hpp"
#include <iostream>

namespace fex
{
using namespace cv;

void DebugHelpers::printMatInformation(const Mat& matrix)
{
    std::cout << "Size: " << matrix.rows << " x " << matrix.cols << std::endl;
    std::cout << "Channels: " << matrix.channels() << std::endl;
    std::cout << "Type: " << opencvTypeAsString(matrix.type()) << std::endl;
}

String DebugHelpers::opencvTypeAsString(int type)
{
    switch(type) {
    case CV_8SC1: return "CV_8SC1";
    case CV_8SC2: return "CV_8SC2";
    case CV_8SC3: return "CV_8SC3";
    case CV_8SC4: return "CV_8SC4";
    case CV_8UC1: return "CV_8UC1";
    case CV_8UC2: return "CV_8UC2";
    case CV_8UC3: return "CV_8UC3";
    case CV_8UC4: return "CV_8UC4";
    case CV_16SC1: return "CV_16SC1";
    case CV_16SC2: return "CV_16SC2";
    case CV_16SC3: return "CV_16SC3";
    case CV_16SC4: return "CV_16SC4";
    case CV_16UC1: return "CV_16UC1";
    case CV_16UC2: return "CV_16UC2";
    case CV_16UC3: return "CV_16UC3";
    case CV_16UC4: return "CV_16UC4";
    case CV_32FC1: return "CV_32FC1";
    case CV_32FC2: return "CV_32FC2";
    case CV_32FC3: return "CV_32FC3";
    case CV_32FC4: return "CV_32FC4";
    case CV_32SC1: return "CV_32SC1";
    case CV_32SC2: return "CV_32SC2";
    case CV_32SC3: return "CV_32SC3";
    case CV_32SC4: return "CV_32SC4";
    case CV_64FC1: return "CV_64FC1";
    case CV_64FC2: return "CV_64FC2";
    case CV_64FC3: return "CV_64FC3";
    case CV_64FC4: return "CV_64FC4";
    default: return "Unknown type";
    }
}

}
