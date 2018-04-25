CC=g++
CFLAGS=-I

all:	main.o restore.o net.o loadDigits.o xor.o
	$(CC) main.o net.o loadDigits.o xor.o -o main -g 
	$(CC) restore.o net.o loadDigits.o -o restore -g 

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