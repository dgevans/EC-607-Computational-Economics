using Weave,NBInclude

name = "Aiyagari Models"
directory = name*"/"
filename = directory * name *".jmd"
filename_jl = directory * name *".jl"
notebookname = directory * name *".ipynb"
#NOTE:Need to run the ipython notebook by itself 
convert_doc(notebookname,filename)
nbexport(filename_jl,notebookname,markdown=false)

#To Slides
set_chunk_defaults!(:fig_width=>15,:fig_height=>5)
weave(filename;fig_path="assets", doctype="pandoc",mod=Main)#to md file
cd(directory)
run(`pandoc -t revealjs --mathjax -V theme=white -s $(name).md -o $(name)_slides.html --slide-level=2 --variable width=1920 --variable height=1080 --metadata title="$(name)"`)
#run(`pandoc -t html --katex -s $(name).md -o $(name)_lecture.html --self-contained --css ../pandoc.css --columns 500 --metadata title="$(name)"`)
cd("..")