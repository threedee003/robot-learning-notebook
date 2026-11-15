# Robot Learning Notebook

A technical book about generative policies for robotics, written as a Jekyll site on top of the [Just the Docs](https://github.com/just-the-docs/just-the-docs) theme.

- Chapters with derivations, runnable PyTorch code, figures, GIFs and videos
- Light and dark mode with a toggle in the header (remembers the reader's choice, follows the OS on first visit)
- LaTeX math with numbered equations (MathJax)
- Full text search (Ctrl+K), copy buttons on code, previous and next page buttons
- New chapters are plain markdown files, nothing else to register

## Run it locally (WSL Ubuntu)

```bash
sudo apt update && sudo apt install -y ruby-full build-essential zlib1g-dev
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
gem install bundler

bundle install
bundle exec jekyll serve --livereload
```

Open http://localhost:4000.

## Add a chapter

```bash
python scripts/new_page.py chapter "Model Predictive Control"
python scripts/new_page.py section 7 "MPPI from Scratch"
```

Then edit the generated markdown. The full guide, with examples for math, code, images, GIFs, videos, callouts and diagrams, is in `appendix/authoring-guide.md` and on the site under **Appendix, How to Add a Chapter**.

## Run the book's code

```bash
cd _includes/code
pip install torch numpy scipy matplotlib
python ch01_bc_mse.py
python ch03_train.py
```

Every listing on the site is included from `_includes/code/`, so the page and the script are always the same file.

Regenerate all figures with `python scripts/make_figures.py`.

## Publish on GitHub Pages

1. Push this folder to a GitHub repository.
2. In the repository, open Settings, Pages, and set Source to **GitHub Actions**.
3. Edit `_config.yml`. Set `url`, `baseurl` (for example `"/robot-learning-book"`), and the GitHub links.

The workflow in `.github/workflows/pages.yml` publishes on every push to `main`.

## Layout

```text
_config.yml            title, author, links, callout types
index.md               home page with chapter cards
chapters/NN-name/      one folder per chapter, index.md plus section files
appendix/              authoring guide, references
assets/media/chNN/     images, gifs, videos
_includes/code/        runnable Python shown in the book
_includes/figure.html  image or gif with caption
_includes/video.html   mp4, webm or YouTube
_includes/codefile.html  show a whole file from _includes/code/
assets/js/book.js      dark mode toggle, page navigation
_sass/custom/custom.scss  book styling for both colour schemes
scripts/               new_page.py, make_figures.py
```

## License

Theme files are from Just the Docs, MIT license, see `LICENSE.txt`.
