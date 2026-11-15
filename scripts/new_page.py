import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTERS = os.path.join(ROOT, "chapters")
MEDIA = os.path.join(ROOT, "assets", "media")

CHAPTER_TEMPLATE = """---
title: "{num}. {title}"
nav_order: {order}
has_children: true
---

# {num}. {title}
{{: .no_toc }}

<p class="rlb-meta">Prerequisites: </p>

1. TOC
{{:toc}}

## Overview

Write the motivation for this chapter here.

## References

1.
"""

SECTION_TEMPLATE = """---
title: "{num}.{sub} {title}"
parent: "{parent}"
nav_order: {sub}
---

# {num}.{sub} {title}
{{: .no_toc }}

1. TOC
{{:toc}}

## First section

Text, math such as $$\\E[\\mathbf{{x}}]$$, and code.

```python
import torch
```

{{% comment %}}
Remove the comment tags around this line once the image exists.
{{% include figure.html src="/assets/media/ch{num:02d}/example.png" alt="Describe the image" caption="Caption text." %}}
{{% endcomment %}}

## References

1.
"""


def slugify(text):
      return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def chapter_dirs():
      out = {}
      if not os.path.isdir(CHAPTERS):
            return out
      for name in os.listdir(CHAPTERS):
            m = re.match(r"(\d+)-", name)
            if m and os.path.isdir(os.path.join(CHAPTERS, name)):
                  out[int(m.group(1))] = name
      return out


def read_title(path):
      with open(path, encoding="utf-8") as f:
            for line in f:
                  m = re.match(r'title:\s*"?(.*?)"?\s*$', line)
                  if m:
                        return m.group(1)
      raise SystemExit(f"no title found in {path}")


def new_chapter(title):
      existing = chapter_dirs()
      num = max(existing, default=0) + 1
      folder = os.path.join(CHAPTERS, f"{num:02d}-{slugify(title)}")
      os.makedirs(folder, exist_ok=False)
      os.makedirs(os.path.join(MEDIA, f"ch{num:02d}"), exist_ok=True)
      path = os.path.join(folder, "index.md")
      with open(path, "w", encoding="utf-8") as f:
            f.write(CHAPTER_TEMPLATE.format(num=num, title=title, order=num + 1))
      print(f"created {os.path.relpath(path, ROOT)}")
      print(f"put media in assets/media/ch{num:02d}/")
      print("add a card for it to index.md if you want it on the home page")


def new_section(num, title):
      existing = chapter_dirs()
      if num not in existing:
            raise SystemExit(f"chapter {num} not found, existing: {sorted(existing)}")
      folder = os.path.join(CHAPTERS, existing[num])
      parent = read_title(os.path.join(folder, "index.md"))
      sub = len([f for f in os.listdir(folder) if f.endswith(".md") and f != "index.md"]) + 1
      path = os.path.join(folder, f"{slugify(title)}.md")
      if os.path.exists(path):
            raise SystemExit(f"{path} already exists")
      os.makedirs(os.path.join(MEDIA, f"ch{num:02d}"), exist_ok=True)
      with open(path, "w", encoding="utf-8") as f:
            f.write(SECTION_TEMPLATE.format(num=num, sub=sub, title=title, parent=parent))
      print(f"created {os.path.relpath(path, ROOT)} as section {num}.{sub} under '{parent}'")


if __name__ == "__main__":
      parser = argparse.ArgumentParser(description="Create a new chapter or section for the book.")
      sub = parser.add_subparsers(dest="kind", required=True)
      c = sub.add_parser("chapter", help='new chapter, e.g. chapter "Model Predictive Control"')
      c.add_argument("title")
      s = sub.add_parser("section", help='new section, e.g. section 7 "MPPI from Scratch"')
      s.add_argument("chapter", type=int)
      s.add_argument("title")
      args = parser.parse_args()
      if args.kind == "chapter":
            new_chapter(args.title)
      else:
            new_section(args.chapter, args.title)
