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

#include "IOHelpers.hpp"
#include <boost/algorithm/string/case_conv.hpp>

namespace fex {

using namespace boost::algorithm;

vector<path> IOHelpers::getDirContents(const string& dirName)
{
    vector<path> paths;
    copy(directory_iterator(dirName),
            directory_iterator(),
            inserter(paths, paths.end ()));

    return paths;
}


vector<path> IOHelpers::getDirContents(const string& dirName,
		const std::string& ext)
{
	string extension = "." + ext;
	vector<path> files;
	for (directory_iterator it( dirName );
			it != directory_iterator();
			++it )
	{
		if (is_regular_file( it->status() ) && to_lower_copy(
		        it->path().extension().string()) == extension)
		{
			files.push_back( it->path());
		}
	}

	return files;
}
}
