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

#ifndef GABORSET_HPP_
#define GABORSET_HPP_

// TODO: Check really needed header files, including all OpenCV headers
// is way too much
#include "opencv2/opencv.hpp"
#include "GaborFilter.hpp"
#include <vector>

namespace fex {

/*
 * Template class for Gabor Filter set.
 * Using long types (such as long, long float or long double will fail
 * compiling, since OpenCV mat's structure does not support them.
 *
 * Using int type is not allowed, and will cause a runtime error.
 */
template<typename _Tp> class GaborSet
{
public:
	/*
	 * Typedefs
	 */
	typedef _Tp value_type;

	/*
	 * Constructors
	 */
	GaborSet();
	GaborSet(int _scales, int _orientations, int _filterSize,
            _Tp _kMax, _Tp _sigma, bool startAtScaleZero = true);
	GaborSet(int _scales, int _orientations, int _filterSizeX,
			int _filterSizeY, _Tp _kMax, _Tp _sigma,
			bool startAtScaleZero = true);
	virtual ~GaborSet();

	/*
	 * Attribute getters
	 */
	int getScales() const;
	int getOrientations() const;
	int getFilterSizeX() const;
	int getFilterSizeY() const;
	_Tp getKMax() const;
	_Tp getSigma() const;
	bool isStartAtScaleZero() const;
	map <pair<int, int>, GaborFilter<_Tp> > getGaborSet() const;

private:
	/*
	 * Attributes
	 */
	int mScales;
	int mOrientations;
	int mFilterSizeX;
	int mFilterSizeY;
	_Tp mKMax;
	_Tp mSigma;
	bool startAtScaleZero;
	map <pair<int, int>, GaborFilter<_Tp> > mGaborSet;

	/*
	 * Private functions
	 */
	void init(map <pair<int, int>, GaborFilter<_Tp> >& result,
			int scales, int orientations, int filterSizeX, int filterSizeY,
			_Tp kMax, _Tp sigma, bool startAtScaleZero);

	void generateGaborSet(map <pair<int, int>, GaborFilter<_Tp> >& result,
			int scales, int orientations, int filterSizeX, int filterSizeY,
			_Tp kMax, _Tp sigma, bool startAtScaleZero);

};

/******************************************************************************
 ******************************************************************************
 **                          CLASS IMPLEMENTATION                            **
 ******************************************************************************
 ******************************************************************************/

/**************
 * Constructors
 **************/
template<typename _Tp> GaborSet<_Tp>::GaborSet()
{
}

template<typename _Tp> GaborSet<_Tp>::GaborSet(int _scales,
		int _orientations, int _filterSize, _Tp _kMax, _Tp _sigma,
		bool startAtScaleZero)
{
	init(mGaborSet, _scales, _orientations, _filterSize, _filterSize,
			_kMax, _sigma, startAtScaleZero);
}

template<typename _Tp> GaborSet<_Tp>::GaborSet(int _scales,
		int _orientations, int _filterSizeX, int _filterSizeY, _Tp _kMax,
		_Tp _sigma, bool startAtScaleZero)
{
	init(mGaborSet, _scales, _orientations, _filterSizeX, _filterSizeY,
			_kMax, _sigma, startAtScaleZero);
}

template<typename _Tp> GaborSet<_Tp>::~GaborSet()
{
    // Nothing to see here... keep walking!!!
}

/*******************
 * Attribute getters
 *******************/
template<typename _Tp>
inline int GaborSet<_Tp>::getScales() const
{
	return mScales;
}

template<typename _Tp>
inline int GaborSet<_Tp>::getOrientations() const
{
	return mOrientations;
}

template<typename _Tp>
inline int GaborSet<_Tp>::getFilterSizeX() const
{
	return mFilterSizeX;
}

template<typename _Tp>
inline int GaborSet<_Tp>::getFilterSizeY() const
{
	return mFilterSizeY;
}

template<typename _Tp>
inline _Tp GaborSet<_Tp>::getKMax() const
{
    return mKMax;
}

template<typename _Tp>
inline _Tp GaborSet<_Tp>::getSigma() const
{
    return mSigma;
}

template<typename _Tp>
inline bool GaborSet<_Tp>::isStartAtScaleZero() const
{
	return isStartAtScaleZero();
}

template<typename _Tp>
inline map <pair<int, int>, GaborFilter<_Tp> >
GaborSet<_Tp>::getGaborSet() const
{
	return mGaborSet;
}

/*******************
 * Private functions
 *******************/
template<typename _Tp>
void GaborSet<_Tp>::init(map <pair<int, int>, GaborFilter<_Tp> >& result,
		int scales, int orientations, int filterSizeX, int filterSizeY,
		_Tp kMax, _Tp sigma, bool startAtScaleZero)
{

	mScales = scales;
	mOrientations = orientations;
	mFilterSizeX = filterSizeX;
	mFilterSizeY = filterSizeY;
	mKMax = kMax;
	mSigma = sigma;
	generateGaborSet(mGaborSet, scales, orientations, filterSizeX,
			filterSizeY, kMax, sigma, startAtScaleZero);

}

template<typename _Tp>
void GaborSet<_Tp>::generateGaborSet(
		map <pair<int, int>, GaborFilter<_Tp> >& result,
		int scales, int orientations, int filterSizeX, int filterSizeY,
		_Tp kMax, _Tp sigma, bool startAtScaleZero)
{

	int startScale = 0;
	int startOrientation = 0;
	int stopScale = scales-1;
	int stopOrientation = orientations-1;

	if(!startAtScaleZero) {
		startScale = 1;
		stopScale = scales;
	}

	for(int i=startScale; i<=stopScale; i++)
	{
		for(int j=startOrientation; j<=stopOrientation; j++)
		{
			result[pair<int, int>(i,j)] = GaborFilter<_Tp>(i, j, filterSizeX,
					filterSizeY, kMax, sigma);
		}
	}

}

}

#endif /* GABORSET_HPP_ */
