using Weave
import Cairo,Fontconfig

name = "Interpolation"
directory = name*"/"
filename = directory * name *".jmd"
weave(filename;fig_path="assets", doctype="pandoc")
notebookname = directory * name *".ipynb"
#NOTE:Need to run the ipython notebook by itself 
convert_doc(filename,notebookname)
#Weave.notebook(filename)
#weave(filename;fig_path="assets", doctype="md2html")
cd(directory)
run(`pandoc -t revealjs --mathjax -V theme=white -s $(name).md -o $(name)_slides.html --slide-level=2 --variable width=1920 --variable height=1080 --metadata title="$(name)"`)
run(`pandoc -t html --katex -s $(name).md -o $(name)_lecture.html --self-contained --css ../pandoc.css --columns 500 --metadata title="$(name)"`)
cd("..")