rm idiagrams/*
pdflatex -shell-escape idiagrams.tex
rubber --clean idiagrams.tex
rm idiagrams.auxlock
cd idiagrams
mogrify -format png *.pdf
rm *.dpth
rm *.log
cd ..
