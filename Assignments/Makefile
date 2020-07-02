#!/usr/bin/make -f
.PHONY: all clean
SRCS = $(wildcard *.tex)
all: $(SRCS:%.tex=%)
%:
	latexmk -pdf -halt-on-error $@

clean:
	latexmk -C
