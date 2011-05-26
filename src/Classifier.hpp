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

#ifndef CLASSIFIER_HPP_
#define CLASSIFIER_HPP_

// TODO: Check really needed header files, including all OpenCV headers
// is way too much
#include "opencv2/opencv.hpp"

namespace fex
{

using namespace cv;

template<typename _Tp> class Classifier
{
public:

    virtual void clear() = 0;
    virtual bool train(Mat_<_Tp> trainingSet, Mat_<int> classes) = 0;
    virtual Mat_<int> predict(Mat_<_Tp> observations) = 0;
};

}

#endif /* CLASSIFIER_HPP_ */
