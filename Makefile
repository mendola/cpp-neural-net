CC=g++
CFLAGS=-I

all:	main.o net.o loadDigits.o
	$(CC) main.o net.o loadDigits.o -o main -g 

main.o: net.h main.cc
	$(CC) -c main.cc -g 

net.o: net.cc net.h
	$(CC) -c net.cc -g 

loadDigits.o: loadDigits.h loadDigits.cc
	$(CC) -c loadDigits.cc -g

clean:
	rm -rf *.o