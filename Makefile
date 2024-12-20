all: run

SRC_DIR := src
MAIN_FILE := $(SRC_DIR)/main.py

run:
	PYTHONPATH=$(SRC_DIR) python $(MAIN_FILE)
