# Report source

This English report uses one-column Times typography and small-cap headings based on the public ICLR 2026 example. The A4 front cover follows the PTIT course requirements and uses the supplied logo and border. [ICLR 2026 template](https://github.com/ICLR/Master-Template/tree/master/iclr2026)

The entry point is `bao-cao.tex`. Edit author and course metadata in `paper-metadata.tex`. The reusable style lives in `course-paper.sty`. The report sections, figures and generated tables use paths relative to this directory.

~~~sh
pdflatex -interaction=nonstopmode -halt-on-error bao-cao.tex
pdflatex -interaction=nonstopmode -halt-on-error bao-cao.tex
~~~

Run both commands in the directory containing the source. The ZIP includes all required figures and tables. To refresh those files from a complete experiment, run `python code/scripts/build_report.py` from the BTL root. Recheck the interpretation in the text when changing the experimental protocol.
