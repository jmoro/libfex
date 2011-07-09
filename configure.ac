#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.67])
AC_INIT([libfex], [0.0.2], [javier.moro@udc.es])
AM_INIT_AUTOMAKE([-Wall -Werror])
LT_PREREQ([2.2])
LT_INIT([dlopen])

AC_LANG([C++])

AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([src/GaborFilter.hpp])
AC_CONFIG_HEADERS([config.h])


# Checks for programs.
AC_PROG_CXX
AC_PROG_INSTALL
AM_PROG_CC_C_O

# Checks for libraries.
AC_CHECK_HEADER(armadillo,,
                [AC_MSG_ERROR([ERROR: Armadillo headers required; use '-Iincludedir' in CXXFLAGS for unusual locations.])])
## second check the lib itself
arma_savedlibs="$LIBS"
ARMADILLO_LIBS="-larmadillo"
AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <armadillo>]], 
                [[arma::mat A = arma::eye<arma::mat>(5,5);]])],
                [],
                [AC_MSG_ERROR([ERROR: Armadillo test program did not link, maybe add '-Llibdir' to LDFLAGS.])])
ARMADILLO_LIBS="$arma_savedlibs"
AC_SUBST([ARMADILLO_LIBS],["${ARMADILLO_LIBS} -larmadillo"])

# OpenCV
PKG_CHECK_MODULES(OPENCV, opencv >= 2.2)
AC_SUBST(OPENCV_CFLAGS)
AC_SUBST(OPENCV_LIBS)

# TBB
PKG_CHECK_MODULES([TBB], [tbb >= 3.0], [AC_DEFINE([HAVE_TBB], [1], [Define to use Intel TBB Library])], [AC_MSG_RESULT([Intel TBB not found.])])
AC_SUBST(TBB_CFLAGS)
AC_SUBST(TBB_LIBS)

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE

# Checks for library functions.
AC_CHECK_FUNCS([pow sqrt])

AC_CONFIG_FILES([Makefile
                 samples/Makefile
                 src/Makefile
                 src/libfex.pc])
AC_OUTPUT
