CC=g++
CFLAGS=-I

all:	main.o restore.o testparams.o net.o loadDigits.o xor.o
	$(CC) main.o net.o loadDigits.o xor.o -o main -g 
	$(CC) restore.o net.o loadDigits.o -o restore -g 
	$(CC) testparams.o net.o loadDigits.o -o testparams -g
testparams.o: net.h testparams.cc
	$(CC) -c testparams.cc -g

restore.o: net.h restore.cc
	$(CC) -c restore.cc -g

main.o: net.h main.cc xor.h
	$(CC) -c main.cc -g

net.o: net.cc net.h
	$(CC) -c net.cc -g

loadDigits.o: loadDigits.h loadDigits.cc
	$(CC) -c loadDigits.cc -g

xor.o: xor.cc xor.h
	$(CC) -c xor.cc -g

clean:
	rm -rf *.o main
