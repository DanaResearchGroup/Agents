Extract:
- experiment type mentions
- temperatures
- pressures
- compositions
- reactor type terms
- residence times / ignition definitions
- mechanism references
- figure/table references

Version 1

Yes — **build the parser first**, but not a giant parser.

Build the **minimum parser that supports the simulation plan**, not a general document-understanding system.

For the simulator agent, the parser is the foundation because everything downstream depends on having **clean evidence**:

* what experiment this is
* what conditions are reported
* where those conditions came from
* what is missing or ambiguous

If you skip that and jump into plan generation, the agent will start mixing extraction and assumption-making, which is where things get messy fast.

## But the parser should be narrow

Do **not** start by trying to parse everything in a paper.

Start with a parser that extracts only the things the simulator needs.

So version 1 of the parser should aim to pull:

* title
* abstract if available
* sectioned raw text by page
* figure captions
* table-like condition text if extractable
* mentions of:

  * JSR / PSR / WSR / CSTR
  * shock tube / ignition delay / RCM
  * temperature
  * pressure
  * residence time
  * equivalence ratio
  * composition
  * mechanism names
  * reactor volume / flow rate

That is enough to support the planner.

## The right order

I would build in this order:

### 1. PDF text parser

Goal: get reliable text out of the paper and preserve page references.

Output something like:

```json
{
  "pdf_path": "paper.pdf",
  "title": "...",
  "abstract": "...",
  "pages": [
    {
      "page_num": 1,
      "text": "..."
    },
    {
      "page_num": 2,
      "text": "..."
    }
  ],
  "captions": [
    {
      "figure_id": "Figure 1",
      "page_num": 5,
      "caption": "..."
    }
  ]
}
```

This is the first thing.

### 2. Evidence extractor

This sits on top of the parser output and looks for simulation-relevant snippets.

For example:

```json
{
  "experiment_mentions": [
    {
      "type": "jsr",
      "text": "Experiments were carried out in a jet-stirred reactor at 1 atm...",
      "page_num": 4
    }
  ],
  "conditions": [
    {
      "kind": "temperature",
      "value_text": "950–1150 K",
      "page_num": 4,
      "source_text": "..."
    },
    {
      "kind": "pressure",
      "value_text": "1 atm",
      "page_num": 4,
      "source_text": "..."
    }
  ]
}
```

### 3. Experiment classifier

Only after you have evidence.

It decides:

* JSR
* IDT
* unsupported
* mixed / ambiguous

### 4. Plan builder

This turns extracted evidence into normalized fields and assumptions.

### 5. Template selector + code generator

Only now.

That is the clean flow.

## Why parser first is the right move

Because the simulator agent’s strength will come from **traceability**.

You want to be able to say:

* temperature came from page 4
* residence time came from Table 2
* pressure was explicit
* thermal mode was inferred from “reactor temperature was controlled”

That is only possible if the parser/evidence layer is solid.

Otherwise the agent becomes:

* hard to debug
* hard to trust
* hard to improve

## What not to build first

Do not start with:

* Cantera code templates
* LLM prompts for full paper interpretation
* automatic full experiment reconstruction
* figure-heavy parsing
* perfect table extraction

Those can come later.

For now, the parser only needs to be good enough to support:

* experiment type detection
* condition extraction
* evidence linking

## A practical MVP parser stack

For the simulator agent, I would start with:

* **PyMuPDF / fitz** for text extraction and page access
* regex + heuristics for:

  * figure captions
  * table mentions
  * condition phrases
* optional later:

  * pdfplumber for tables if needed

But first version:

* direct text extraction
* page-level storage
* no OCR unless absolutely necessary

That matches the right MVP philosophy.

## The first parser outputs I would define

Make the parser return three things:

### `PaperDocument`

```python
@dataclass
class PageText:
    page_num: int
    text: str

@dataclass
class FigureCaption:
    figure_id: str
    page_num: int
    caption: str

@dataclass
class PaperDocument:
    pdf_path: str
    title: str | None
    abstract: str | None
    pages: list[PageText]
    captions: list[FigureCaption]
```

### `EvidenceSnippet`

```python
@dataclass
class EvidenceSnippet:
    kind: str
    value_text: str
    page_num: int
    source_text: str
    confidence: float
```

Kinds might be:

* `experiment_type`
* `temperature`
* `pressure`
* `residence_time`
* `equivalence_ratio`
* `composition`
* `mechanism`
* `flow_rate`
* `reactor_volume`

That gives the planner something structured to work with.

## Best way to start tomorrow morning

I would implement exactly these files first:

```text
simulator_agent/
  parser/
    pdf_parser.py
    models.py
    caption_extractor.py
  extractor/
    evidence_extractor.py
  main.py
```

And make the first CLI just do:

```bash
python -m simulator_agent.main --pdf paper.pdf --outdir outputs/
```

with outputs:

* `parsed_document.json`
* `evidence.json`

No simulation code yet.

If that works, then move to:

* experiment classification
* plan schema
* plan builder

## Direct answer

So yes — **build the parser first**.

But build a **simulation-oriented parser**, not a universal paper parser.

The first milestone should be:

* extract paper text by page
* detect captions
* pull simulation-relevant evidence snippets
* save them in a structured format with page references

Once that works, the rest of the simulator agent becomes much easier and much more trustworthy.
