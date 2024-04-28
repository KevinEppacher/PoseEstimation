# Der Compiler
CC=g++

# Include Pfade
INCLUDES = -I/usr/local/include/opencv \
           -I/usr/local/include \
           -I/home/fhtw_user/catkin_ws/src/HW_3/libraries

# Compiler Flags
CFLAGS = -Wall -Wextra -O2 $(INCLUDES)

# Linker Flags
LDFLAGS = $(shell pkg-config --libs opencv)

# Das Ziel, das standardmäßig gebaut wird
all: program

# Regel, um das Programm zu bauen
program: feature_extraction.o
	$(CC) $(CFLAGS) feature_extraction.o -o program $(LDFLAGS)

# Regel, um die Objektdatei zu erzeugen
feature_extraction.o: feature_extraction.cpp
	$(CC) $(CFLAGS) -c feature_extraction.cpp

# Regel, um temporäre Dateien zu löschen
clean:
	rm -f *.o program

# Regel, um das Programm auszuführen (optional)
run: program
	./program /home/fhtw_user/catkin_ws/src/HW_3/data/simpson_image_real.png
