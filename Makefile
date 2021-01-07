# standard compile options for the c++ executable
FLAGS = -fPIC

# the python interface through swig
PYTHONI = -I/usr/include/python3.7/
PYTHONL = -Xlinker -export-dynamic

# cpp file
FILE = ./cfr.cpp ./game.cpp ./oracle.cpp ./calculator.cpp
FILE_O = ./cfr.o ./game.o ./oracle.o ./calculator.o
#FILE = ./calculator.cpp
#FILE_O = ./calculator.o
# default super-target
all: 
	rm ./*.o
	g++ -fPIC -c $(FILE)
	swig -c++ -python -o MYCFR_wrap.cxx MYCFR.i 
	g++ $(FLAGS) $(PYTHONI) -c MYCFR_wrap.cxx -o MYCFR_wrap.o
	g++ $(PYTHONL) $(LIBFLAGS) -shared $(FILE_O) MYCFR_wrap.o -o _MYCFR.so