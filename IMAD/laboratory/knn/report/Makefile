all: compile clean

compile: main.tex
	pdflatex main.tex
	pdflatex main.tex

clean:
	find . \( -name "*.aux" -or -name "*.bbl" -or \
	  	  -name "*.bcf" -or -name "*.blg" -or \
	          -name "*.lof" -or -name "*.log" -or \
	          -name "*.lot" -or -name "*.lol" -or \
		  -name "*.nav" -or -name "*.out" -or \
		  -name "*.snm" -or -name "*.toc" -or \
		  -name "*.xml" \) \
		  -type f -delete
	rm -rf _minted-main/
