all: compile clean

compile: main.tex
	pdflatex main.tex
	pdflatex main.tex

clean:
	find -E . \
	  -regex '.*\.(aux|bbl|bcf|blg|lof|log|lol|lot|nav|out|snm|toc|xml)' \
          -type f -delete
