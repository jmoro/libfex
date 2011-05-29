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

# Boost
BOOST_REQUIRE([1.44])
BOOST_SYSTEM()
BOOST_FILESYSTEM()

# OpenCV
PKG_CHECK_MODULES(OPENCV, opencv)
AC_SUBST(OPENCV_CFLAGS)
AC_SUBST(OPENCV_LIBS)

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE

# Checks for library functions.
AC_CHECK_FUNCS([pow sqrt])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 src/libfex.pc])
AC_OUTPUT
