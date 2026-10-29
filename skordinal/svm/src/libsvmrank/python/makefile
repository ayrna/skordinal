.PHONY: clean libsvmrank

libsvmrank: setup.py
	python setup.py build_ext --inplace && rm -f redsvm.cpp && rm -Rf build
clean:
	rm *.so
