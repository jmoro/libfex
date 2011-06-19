ACLOCAL_AMFLAGS = -I m4

SUBDIRS = src samples

dist-hook:
	rm -rf `find $(distdir) -name .svn`
	rm -rf `find $(distdir) -name .git`
	rm -rf `find $(distdir) -name .gitignore`
