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

#ifndef IOHELPERS_HPP_
#define IOHELPERS_HPP_

#include <boost/filesystem.hpp>
#include <vector>

namespace fex
{

using namespace std;
using namespace boost::filesystem3;

class IOHelpers
{
public:
		static vector<path> getDirContents (const string& dirName);
		static vector<path> getDirContents (const string& dirName,
				const std::string& ext);
};

}

#endif /* IOHELPERS_HPP_ */
