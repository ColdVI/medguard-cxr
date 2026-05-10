# Report Build

The LaTeX source is compile-ready, but this local machine did not have `pdflatex` installed during the release pass. `tectonic` was present but crashed before compilation in the local Homebrew runtime, so no PDF was produced here.

Compile from the repository root with:

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
cp main.pdf report.pdf
```

Expected output:

```text
report/report.pdf
```
