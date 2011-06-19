FEX_INCLUDE = -I$(top_srcdir)/src
FEX_LTLIB = ../src/libfex.la

AM_CPPFLAGS = $(FEX_INCLUDE) $(OPENCV_CFLAGS)
LDADD = $(FEX_LTLIB) $(OPENCV_LIBS) $(ARMADILLO_LIBS)

bin_PROGRAMS = filter 

filter_SOURCES = FilterSample.cpp
