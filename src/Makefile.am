pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libfex.pc

lib_LTLIBRARIES = libfex.la

include_HEADERS = GaborFilter.hpp \
                  ImageHelpers.hpp \
                  Classifier.hpp \
                  LDAQR.hpp \
                  MathHelpers.hpp \
                  DebugHelpers.hpp \
                  FilteringHelpers.hpp \
                  GaborSet.hpp

libfex_la_SOURCES = DebugHelpers.cpp
libfex_la_CPPFLAGS = $(OPENCV_CFLAGS) ${TBB_CFLAGS}
libfex_la_LIBADD = $(OPENCV_LIBS) $(ARMADILLO_LIBS) ${TBB_LIBS}
libfex_la_LDFLAGS = -version-info 0:1:0
