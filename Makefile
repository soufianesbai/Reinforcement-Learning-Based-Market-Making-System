.PHONY: pdf rapport presentation clean help

# Default target
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  pdf            Compile both rapport.pdf and presentation.pdf"
	@echo "  rapport        Compile rapport.pdf only"
	@echo "  presentation   Compile presentation.pdf only"
	@echo "  clean          Remove generated PDF and auxiliary files"
	@echo "  help           Show this help message"

# Compile both PDFs
pdf: rapport presentation
	@echo "✅ Both PDFs compiled successfully"

# Compile rapport.pdf
rapport:
	@echo "📄 Compiling rapport.pdf..."
	pdflatex -interaction=nonstopmode -output-directory=rapport rapport/rapport.tex > /dev/null 2>&1 || true
	pdflatex -interaction=nonstopmode -output-directory=rapport rapport/rapport.tex > /dev/null 2>&1 || true
	@if [ -f rapport/rapport.pdf ]; then echo "✅ rapport.pdf compiled"; else echo "❌ rapport.pdf compilation failed"; exit 1; fi

# Compile presentation.pdf
presentation:
	@echo "📊 Compiling presentation.pdf..."
	pdflatex -interaction=nonstopmode -output-directory=rapport rapport/presentation.tex > /dev/null 2>&1 || true
	pdflatex -interaction=nonstopmode -output-directory=rapport rapport/presentation.tex > /dev/null 2>&1 || true
	@if [ -f rapport/presentation.pdf ]; then echo "✅ presentation.pdf compiled"; else echo "❌ presentation.pdf compilation failed"; exit 1; fi

# Clean generated files
clean:
	@echo "🧹 Cleaning LaTeX auxiliary files..."
	rm -f rapport/*.pdf rapport/*.aux rapport/*.log rapport/*.nav rapport/*.snm rapport/*.toc rapport/*.out rapport/*.bbl rapport/*.blg
	@echo "✅ Clean completed"
