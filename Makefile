# Build the C kernel that main.py loads with cffi.
#
#   make            build matmul.so
#   make run        build, then run the Optuna search
#   make dashboard  inspect results with optuna-dashboard
#   make clean      remove the compiled shared library

CC := gcc
CFLAGS ?= -O3 -fPIC -fopenmp
LDFLAGS ?= -shared

TARGET := matmul.so
SRC := matmul.c

.PHONY: all run dashboard clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

run: $(TARGET)
	uv run main.py

dashboard:
	uv run optuna-dashboard sqlite:///db.sqlite3

clean:
	rm -f $(TARGET)
