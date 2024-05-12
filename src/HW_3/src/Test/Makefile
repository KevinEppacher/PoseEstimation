# Compiler-Einstellungen
CXX = g++
CXXFLAGS = -Wall -g

# Include Pfade
INCLUDES = -I/usr/local/include/opencv4 \
           -I/usr/local/include \
           -I/home/fhtw_user/catkin_ws/src/HW_3/libraries

# Linker Flags und Bibliotheken
LFLAGS = -L/usr/local/lib -L/usr/local/include/opencv4/opencv2
LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_features2d -lopencv_xfeatures2d -lopencv_videoio -lopencv_calib3d

# Quelldateien und Objektdateien
SRCS = test.cpp
OBJS = $(SRCS:.cpp=.o)

# Hauptausführbare Datei
MAIN = test

.PHONY: clean

# Standard-Build-Prozess
all: $(MAIN)
	@echo Simple compiler named $(MAIN) has been compiled

# Hauptziel
$(MAIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LFLAGS) $(OBJS) -o $(MAIN) $(LIBS)

# Regel für Quelldateien
.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<  -o $@

# Regel für Bereinigung und Kompilierung
clean_all: clean all


clean:
	$(RM) *.o *~ $(MAIN)
