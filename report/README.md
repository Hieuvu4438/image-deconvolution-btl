# Report source

The entry point is `bao-cao.tex`. Edit author and course metadata in `paper-metadata.tex`. Listings in the report read excerpts from the implementation files and display their original line numbers.

~~~sh
pdflatex -interaction=nonstopmode -halt-on-error bao-cao.tex
pdflatex -interaction=nonstopmode -halt-on-error bao-cao.tex
~~~

Run both commands in the directory containing the source. The ZIP includes the figures, tables, and source files referenced by the listings. To refresh the tables and figures from a complete experiment, run `python code/scripts/build_report.py` from the project root.
