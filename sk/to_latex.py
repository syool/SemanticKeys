"""
to_latex.py — Convert the manuscript sections to an elsarticle submission.

Pandoc handles the prose. What it cannot do is resolve the `[?]` citation
placeholders left during drafting, or know which of our own generated tables
belongs where. Both are handled here.

Citation resolution works by looking at the sentence around each placeholder
and matching against distinctive terms — "G²CP", "8,560 trials", "seven-field
schema". A placeholder that cannot be resolved is left as `\\cite{CITENEEDED}`
so it fails visibly at compile time rather than silently vanishing.

Usage:
    python to_latex.py                    # writes paper/tex/
    python to_latex.py --compile          # and runs pdflatex
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
TEX = PAPER / "tex"

SECTIONS = [
    "intro_related.md",
    "protocol.md",
    "measurement.md",
    "experiments.md",
    "results.md",
    "discussion.md",
    "limitations.md",
]

# Each placeholder is resolved from the words around it. Order matters: the
# first pattern that matches the surrounding sentence wins, so put the more
# specific patterns first.
CITATION_RULES = [
    (r"G²CP|G2CP|graph operations over a shared", "benkhaled2026g2cp"),
    (r"CodeAgents|modular pseudocode", "yang2025codeagents"),
    (r"8,560|replication of the Natural Language Tools|14 models", "somma2026nltreplication"),
    (r"Natural Language Tools framework of|original study reported", "johnson2025nlt"),
    (r"Two Calls Beat Five Agents|Qwen2\.5-7B", "prajapati2026twocalls"),
    (r"companion paper|Why 7 fields|first principles", "xu2026svaf"),
    (r"Mesh Memory Protocol|seven-field schema", "xu2026mmp"),
    (r"HyLaT|trilemma|latent and text channels", "mou2026hylat"),
    (r"Argent|structured metadata to resist", "sharma2026argent"),
    (r"latent communication|latent channels dominate", "latentsurvey2026"),
    (r"HotpotQA", "yang2018hotpotqa"),
    (r"vLLM", "kwon2023vllm"),
    (r"Rhetorical Structure Theory", "mann1988rst"),
    # A placeholder after a general statement about the class of proposals
    # should cite representatives of it rather than resolving to nothing.
    (r"fixing a schema|requiring agents to\s+exchange instances",
     "benkhaled2026g2cp,yang2025codeagents,xu2026mmp"),
    (r"Frame Semantics|FrameNet", "baker1998framenet"),
]


def resolve_citations(text: str) -> tuple[str, int]:
    """Replace [?] with \\cite{...} using the surrounding sentence."""
    unresolved = 0

    def context_before(pos: int, prev_end: int) -> str:
        """
        Text between the previous citation and this one.

        Whole paragraphs are too wide: §2.1 introduces three protocols in one
        paragraph, so every placeholder in it saw the same text and resolved
        to whichever rule matched first. A citation refers to what precedes
        it, and the previous citation marks where that starts. Hard wrapping
        makes single sentences too narrow, so the window is bounded by the
        paragraph on one side and the preceding citation on the other.
        """
        para_start = text.rfind("\n\n", 0, pos) + 1
        return text[max(para_start, prev_end):pos]

    out, last = [], 0
    for m in re.finditer(r"\[\?\]", text):
        context = context_before(m.start(), last)
        key = None
        for pattern, k in CITATION_RULES:
            if re.search(pattern, context, re.I):
                key = k
                break
        if key is None:
            key = "CITENEEDED"
            unresolved += 1
        out.append(text[last:m.start()])
        out.append(f"\\cite{{{key}}}")
        last = m.end()
    out.append(text[last:])
    return "".join(out), unresolved


# Floats are marked in the Markdown rather than written as LaTeX, so the
# sources stay readable and the assembled reading copy is unaffected.
FLOAT_RE = re.compile(r"\{\{FLOAT:(table|figure):([a-z]+)\}\}")

FIGURE_CAPTIONS = {
    "cost": ("Cumulative input tokens against turn count, 16K window. "
             "Natural-language conditions re-send a growing transcript; "
             "structured conditions propagate a bounded state tuple.",
             "fig:cost"),
    "verbosity": ("Input-token saving against baseline verbosity, one point "
                  "per model. The ordering follows verbosity, not parameter "
                  "count. Four points are plotted rather than fitted.",
                  "fig:verbosity"),
    "scale": ("Accuracy difference against model size. The high-powered "
              "confirmation run is marked separately from the underpowered "
              "ladder points and is not averaged with them.",
              "fig:scale"),
}


def raw(latex: str) -> str:
    """Wrap LaTeX so pandoc passes it through verbatim."""
    return "\n```{=latex}\n" + latex + "\n```\n"


def resolve_floats(text: str, assets: Path) -> tuple[str, list[str]]:
    """
    Replace float markers with LaTeX, or with a visible placeholder.

    A missing table must not vanish silently: the manuscript refers to it by
    number, and a reader would find the reference dangling. The placeholder
    prints in the PDF and names the command that would generate the asset.
    """
    missing = []

    def sub(m):
        kind, name = m.group(1), m.group(2)
        if kind == "table":
            path = assets / "tables" / f"{name}.tex"
            if path.exists():
                return f"\\input{{tables/{name}}}"
            missing.append(f"tables/{name}.tex")
            return (f"\\begin{{table}}[t]\\centering\\fbox{{\\parbox{{0.8\\linewidth}}"
                    f"{{\\textbf{{Missing: tables/{name}.tex}}\\\\"
                    f"Run \\texttt{{scripts/build\\_assets.sh}} on the machine "
                    f"holding the result files.}}}}\\end{{table}}")
        path = assets / "figures" / f"{name}.pdf"
        caption, label = FIGURE_CAPTIONS.get(name, (name, f"fig:{name}"))
        if path.exists():
            return ("\\begin{figure}[t]\\centering\n"
                    f"\\includegraphics[width=0.72\\linewidth]{{figures/{name}}}\n"
                    f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
                    "\\end{figure}")
        missing.append(f"figures/{name}.pdf")
        return (f"\\begin{{figure}}[t]\\centering\\fbox{{\\parbox{{0.8\\linewidth}}"
                f"{{\\textbf{{Missing: figures/{name}.pdf}}\\\\"
                f"Run \\texttt{{scripts/build\\_assets.sh}}.}}}}\\end{{figure}}")

    return FLOAT_RE.sub(sub, text), missing


def strip_drafting(text: str) -> str:
    """Remove drafting apparatus that has no place in a submission."""
    out, skipping = [], False
    for line in text.splitlines():
        if re.match(r"^#{2,4} Notes?\b", line) or re.match(r"^#{2,4} Notes on ", line):
            skipping = True
            continue
        if skipping and re.match(r"^#{1,3} ", line):
            skipping = False
        if not skipping:
            out.append(line)
    text = "\n".join(out)
    # "Draft." notes wrap, so match to the end of the paragraph
    # rather than the end of the line.
    text = re.sub(r"^Draft\..*?(?=\n\n)", "", text, flags=re.M | re.S)
    text = re.sub(r"^---$", "", text, flags=re.M)
    text = re.sub(r"\*Target:.*?\*", "", text, flags=re.S)
    text = re.sub(r"\*\*Word count:.*?\*\*", "", text, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def demote_headings(text: str) -> str:
    """
    Markdown uses `#` for numbered sections; LaTeX wants \\section.

    Our sections carry their own numbers ("# 6. Results"), which elsarticle
    would number again. Strip the manual numbers and let LaTeX count.
    """
    text = re.sub(r"^# \d+\.\s*", "# ", text, flags=re.M)
    text = re.sub(r"^## \d+\.\d+\s*", "## ", text, flags=re.M)
    return text


def extract_abstract() -> str:
    src = (PAPER / "abstract_conclusion.md").read_text()
    start = src.find("Structured communication protocols for large language")
    end = src.find("**Word count")
    return src[start:end].strip() if start != -1 else ""


def extract_conclusion() -> str:
    src = strip_drafting((PAPER / "abstract_conclusion.md").read_text())
    start = src.find("## 9. Conclusion")
    if start == -1:
        return ""
    body = src[start:]
    return re.sub(r"^## 9\.\s*", "# ", body, flags=re.M)


def extract_title() -> str:
    src = (PAPER / "abstract_conclusion.md").read_text()
    m = re.search(r"^## Title\s*\n+(?:Preferred:\s*\n+)?((?:> .*\n)+)", src, re.M)
    if not m:
        return "Untitled"
    return " ".join(
        l.lstrip("> ").strip() for l in m.group(1).strip().splitlines()
    ).replace("**", "").strip()


# Characters the manuscript uses that pdflatex cannot typeset directly under
# the T1/inputenc combination elsarticle expects. Em and en dashes pandoc
# handles itself; the rest need explicit mapping. Applied before pandoc so
# that the replacements pass through as ordinary LaTeX.
UNICODE_MAP = {
    "§": r"\S",
    "×": r"$\times$",
    "−": "--",          # a minus sign in prose is a dash, not math
    "²": r"\textsuperscript{2}",
    "α": r"$\alpha$",
}


def map_unicode(text: str) -> str:
    for ch, repl in UNICODE_MAP.items():
        text = text.replace(ch, repl)
    return text


def escape_latex(text: str) -> str:
    """
    Escape characters LaTeX treats specially.

    Only needed for text that bypasses pandoc — the abstract and the title,
    which are injected into the preamble directly. An unescaped percent sign
    comments out the rest of its line, which silently swallowed a clause of
    the abstract ("on 73.5% of turns holds its accuracy anyway" became "on
    73.5anyway") and produced a grammatical sentence saying something else.
    """
    for ch, repl in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                     ("_", r"\_")):
        text = text.replace(ch, repl)
    return text


def md_to_tex(markdown: str) -> str:
    """Run pandoc, then repair what it does not know about."""
    markdown = map_unicode(markdown)
    proc = subprocess.run(
        ["pandoc", "-f", "markdown+pipe_tables", "-t", "latex",
         "--top-level-division=section", "--wrap=preserve"],
        input=markdown, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pandoc failed: {proc.stderr}")
    tex = proc.stdout

    # Pandoc escapes backslashes inside our already-formed \cite commands.
    tex = tex.replace(r"\textbackslash cite\{", r"\cite{")
    tex = re.sub(r"\\cite\{([a-zA-Z0-9_]+)\}\\?", r"\\cite{\1}", tex)

    # Longtable is pandoc's default for pipe tables and breaks in two-column
    # layouts; elsarticle submissions want floats.
    tex = tex.replace(r"\begin{longtable}", r"\begin{tabular}")
    tex = tex.replace(r"\end{longtable}", r"\end{tabular}")
    tex = re.sub(r"\\endhead\n", "", tex)
    tex = re.sub(r"\\endfirsthead\n", "", tex)
    tex = re.sub(r"\\endlastfoot\n", "", tex)
    tex = re.sub(r"\\caption\{\}\\tabularnewline\n", "", tex)

    return tex


PREAMBLE = r"""%% Generated by sk/to_latex.py — edit the Markdown sources, not this file.
\documentclass[preprint,12pt,authoryear]{elsarticle}

\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{graphicx}
\graphicspath{{figures/}}
\usepackage[T1]{fontenc}
% Latin Modern rather than Computer Modern: the T1-encoded CM fonts are only
% available as bitmaps here, and pdflatex regenerates them at compile time,
% which is slow enough to look like a hang. Latin Modern ships as Type 1 and
% is the same design.
\usepackage{lmodern}
\usepackage[utf8]{inputenc}
\usepackage{textcomp}
\usepackage{hyperref}
\usepackage{longtable}
% Pandoc emits column specifications using \real and \arraybackslash for its
% proportional-width tables; both need these packages.
\usepackage{array}
\usepackage{calc}
\usepackage{ragged2e}

% Pandoc emits these for tightly spaced lists and for shaded code blocks.
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}

\journal{Neurocomputing}

\begin{document}

\begin{frontmatter}

\title{TITLE_HERE}

\author[a]{Author Name}
\ead{author@example.edu}
\affiliation[a]{organization={Institution},
                city={City},
                country={Country}}

\begin{abstract}
ABSTRACT_HERE
\end{abstract}

\begin{keyword}
multi-agent systems \sep large language models \sep
inter-agent communication \sep structured protocols \sep
token efficiency \sep long-horizon interaction
\end{keyword}

\end{frontmatter}

"""


def check_tools(need_latex: bool) -> None:
    """
    Fail with instructions rather than a traceback.

    The conversion needs pandoc and, to compile, a LaTeX installation. Neither
    is needed to run the experiments, so the machine holding the result files
    often has neither — a stack trace ending in FileNotFoundError does not
    make that obvious.
    """
    import shutil

    missing = []
    if shutil.which("pandoc") is None:
        missing.append(("pandoc", "pandoc"))
    if need_latex:
        for exe, pkg in (("pdflatex", "texlive-latex-recommended"),
                         ("bibtex", "texlive-binaries")):
            if shutil.which(exe) is None:
                missing.append((exe, pkg))

    if not missing:
        return

    print("Missing tools:", ", ".join(m[0] for m in missing), file=sys.stderr)
    print("", file=sys.stderr)
    print("  apt-get install -y " + " ".join(sorted({m[1] for m in missing})),
          file=sys.stderr)
    if need_latex:
        print("  apt-get install -y texlive-publishers "
              "texlive-fonts-recommended lmodern", file=sys.stderr)
    print("", file=sys.stderr)
    print("Nothing here needs the experiment data, so this can equally run on",
          file=sys.stderr)
    print("another machine: generate the tables and figures where the results",
          file=sys.stderr)
    print("live (scripts/build_assets.sh, no pandoc required), copy paper/",
          file=sys.stderr)
    print("across, and convert there.", file=sys.stderr)
    sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compile", action="store_true")
    p.add_argument("--outdir", default=str(TEX))
    args = p.parse_args()

    check_tools(need_latex=args.compile)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    total_unresolved = 0
    missing_assets: list[str] = []
    bodies = []
    for name in SECTIONS:
        path = PAPER / name
        if not path.exists():
            print(f"  missing: {name}")
            continue
        md = demote_headings(strip_drafting(path.read_text()))
        md, unresolved = resolve_citations(md)
        total_unresolved += unresolved
        md, missing = resolve_floats(md, out)
        missing_assets.extend(missing)
        bodies.append(md_to_tex(md))
        print(f"  converted {name}" +
              (f"  ({unresolved} unresolved citations)" if unresolved else ""))

    concl, u = resolve_citations(extract_conclusion())
    total_unresolved += u
    bodies.append(md_to_tex(concl))

    abstract, u = resolve_citations(extract_abstract())
    abstract = escape_latex(map_unicode(abstract))
    abstract = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", abstract)
    abstract = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\emph{\1}", abstract)
    total_unresolved += u

    doc = PREAMBLE.replace("TITLE_HERE",
                           escape_latex(map_unicode(extract_title())))
    doc = doc.replace("ABSTRACT_HERE", abstract)
    doc += "\n\n".join(bodies)
    doc += "\n\n\\bibliographystyle{elsarticle-harv}\n"
    doc += "\\bibliography{references}\n\n\\end{document}\n"

    (out / "manuscript.tex").write_text(doc, encoding="utf-8")

    bib = PAPER / "references.bib"
    if bib.exists():
        (out / "references.bib").write_text(bib.read_text(), encoding="utf-8")

    print(f"\nWrote {out / 'manuscript.tex'}")
    if missing_assets:
        print(f"  {len(missing_assets)} float(s) not built; placeholders emitted:")
        for a in missing_assets:
            print(f"    {a}")
        print("  Run scripts/build_assets.sh where the result files are.")
    if total_unresolved:
        print(f"  {total_unresolved} citations could not be resolved and are")
        print(f"  marked \\cite{{CITENEEDED}} — they will show as [?] in the PDF.")
        print(f"  Add a rule to CITATION_RULES or cite them by hand.")

    if args.compile:
        import shutil
        if shutil.which("kpsewhich"):
            for name, pkg in (("elsarticle.cls", "texlive-publishers"),
                              ("lmodern.sty", "lmodern")):
                r = subprocess.run(["kpsewhich", name], capture_output=True,
                                   text=True)
                if not r.stdout.strip():
                    print(f"\nMissing {name} — apt-get install -y {pkg}",
                          file=sys.stderr)
                    sys.exit(1)
        print("\nCompiling...")
        for i, cmd in enumerate([
            ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
            ["bibtex", "manuscript"],
            ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
            ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
        ]):
            r = subprocess.run(cmd, cwd=out, capture_output=True, text=True)
            if r.returncode != 0 and cmd[0] == "pdflatex" and i == 0:
                # First pass fails on undefined references; that is expected.
                pass
        pdf = out / "manuscript.pdf"
        if pdf.exists():
            print(f"  {pdf} ({pdf.stat().st_size // 1024} KB)")
        else:
            log = (out / "manuscript.log")
            if log.exists():
                errs = [l for l in log.read_text(errors="replace").splitlines()
                        if l.startswith("!")]
                print("  compilation failed:")
                for e in errs[:8]:
                    print(f"    {e}")


if __name__ == "__main__":
    main()