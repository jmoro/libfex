FEX_INCLUDE = -I$(top_srcdir)/src
FEX_LTLIB = ../src/libfex.la

AM_CPPFLAGS = $(FEX_INCLUDE) $(OPENCV_CFLAGS) ${TBB_CFLAGS}
LDADD = $(FEX_LTLIB) $(OPENCV_LIBS) $(ARMADILLO_LIBS) ${TBB_LIBS}

bin_PROGRAMS = filter filterSet math

filter_SOURCES = FilterSample.cpp

filterSet_SOURCES = FilterSetSample.cpp

math_SOURCES = MathSample.cpp