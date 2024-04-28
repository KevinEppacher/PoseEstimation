# Compiler-Einstellungen
CXX = g++
CXXFLAGS = -Wall -g

# Include Pfade
INCLUDES = -I/usr/local/include/opencv4 \
           -I/usr/local/include \
           -I/home/fhtw_user/catkin_ws/src/HW_3/libraries

# Linker Flags und Bibliotheken
LFLAGS = -L/usr/local/lib
LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_features2d -lopencv_xfeatures2d

# Quelldateien und Objektdateien
SRCS = feature_extraction.cpp
OBJS = $(SRCS:.cpp=.o)

# Hauptausführbare Datei
MAIN = feature_extraction

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


clean:
	$(RM) *.o *~ $(MAIN)



# # Der Compiler
# CXX = g++
# CXXFLAGS = -Wall -g

# # Include Pfade
# INCLUDES = 	-I/usr/local/include/opencv4 \
# 			-I/usr/local/include \
# 			-I/home/fhtw_user/catkin_ws/src/HW_3/libraries

# # Compiler Flags
# CFLAGS = -Wall -Wextra -O2 $(INCLUDES)

# # Linker Flags
# LDFLAGS = $(shell pkg-config --cflags --libs opencv4)

# # Das Ziel, das standardmäßig gebaut wird
# all: program

# # Regel, um das Programm zu bauen
# program: feature_extraction.o
# 	$(CC) feature_extraction.o -o program $(LDFLAGS)

# # Regel, um die Objektdatei zu erzeugen
# feature_extraction.o: feature_extraction.cpp
# 	$(CC) $(CFLAGS) -c feature_extraction.cpp

# # Regel, um temporäre Dateien zu löschen
# clean:
# 	rm -f *.o program

# # Regel, um das Programm auszuführen (optional)
# run: program
# 	./program
