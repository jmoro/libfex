pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libfex.pc

lib_LTLIBRARIES = libfex.la

include_HEADERS = GaborFilter.hpp \
                  ImageHelpers.hpp \
                  Classifier.hpp \
                  LDAQR.hpp \
                  MathHelpers.hpp \
                  DebugHelpers.hpp \
                  FilterSet.hpp \
                  IOHelpers.hpp

libfex_la_SOURCES = DebugHelpers.cpp IOHelpers.cpp
libfex_la_CPPFLAGS = $(BOOST_CPPFLAGS) $(OPENCV_CFLAGS)
libfex_la_LIBADD = $(BOOST_SYSTEM_LIBS) $(BOOST_FILESYSTEM_LIBS) $(OPENCV_LIBS)
libfex_la_LDFLAGS = -version-info 0:0:0
