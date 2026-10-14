---
name: all-study-notes
description: Analyse an EEG stroke-classification paper for this final-year project and write brief, structured notes next to its PDF. Use when the user asks to read, summarise or make notes on a paper in Specs/.
---

# Study Notes for Project Papers

Context: the project is EEG-based stroke classification with ML. The reference report is Tamim.pdf (Specs/info/Tamim), which studies how epoch **duration** and **overlap** affect performance on a fixed pipeline. Notes should make each paper comparable to that work.

## 1. Check eligibility

A paper is eligible if **all** hold:
- Uses EEG (EOG/other signals alongside is fine).
- Classifies or grades stroke (stroke vs control, subtype, or severity).
- Uses ML / statistical classification on **multiple subjects**.

Not eligible: reviews, simulation-only studies, single-case reports, imaging-only (CT/MRI) work.

If not eligible, tell the user in one line why and stop. If borderline (e.g. non-stroke EEG with a reusable method), say so and ask before writing notes.

## 2. Read the whole paper

Read every page, including tables, figure captions and references. Extract:

| Item | What to find |
|---|---|
| Task | Binary / subtype / severity; which stroke types (ischaemic, haemorrhagic, TIA) |
| Subjects | Count per class, ages, recording stage (acute/chronic) if given |
| Recording | Channels used (of how many), sampling rate, duration, conditions (resting, eyes open/closed, stimulation), devices |
| Preprocessing | Filtering, resampling, re-referencing, artefact handling |
| Segmentation | Epoch length, overlap (note "not reported" explicitly), or whole recording |
| Features | Full list with counts, grouped by family (band power, ratios, asymmetry, coherence, time-domain, entropy, wavelet…) and where they came from |
| Model | Classifier(s), baselines, hyperparameter selection |
| Evaluation | Split unit (subject vs epoch), CV scheme, metrics, headline result |
| Dataset | Public link if available |

## 3. Check rigour

Look for and record:
- **Leakage:** epoch-level splits, scaler/selector/threshold fitted on test data, early stopping or tuning on the test set.
- **Unfair comparisons:** proposed model tuned more than baselines.
- **Confounds:** different devices/sites/sampling rates by class, class imbalance, subtype mix.
- **Comparability:** accuracy vs AUC, binary vs multi-class, different CV schemes — say what the result can and cannot be compared to.
- **Internal inconsistencies:** text vs tables vs figures, numbers that don't add up.

Only state what the paper shows. If something isn't reported, write "not reported" rather than guessing.

## 4. Note points of interest

Record measures, concepts or findings the paper defines or cites that could be reused in the project, even if the paper itself doesn't use them as features (e.g. BSI, a new band ratio, a stroke-severity correlation, a preprocessing trick). For each:
- Name and source (cited paper + section).
- How it's calculated, in 1–2 lines (include the formula if short).
- Typical values / thresholds and what they indicate.
- How it relates to Tamim's pipeline or the project (same idea, different, not used).

Skip the section if nothing qualifies.

## 5. Pick the top 3 references

From the paper's reference list, choose the 3 most useful for this project, preferring:
1. Sources of the feature set or method the paper reuses.
2. Established EEG stroke markers (asymmetry/BSI, slowing, qEEG reviews).
3. Datasets or studies also cited by Tamim or other project papers.

Give a full citation and one line on why it matters.

## 6. Write the notes file

Save next to the PDF as `<First author> et al. (<year>).md` (match the PDF name). Keep it concise: bullets and tables, no prose paragraphs. Use this layout:

```markdown
# <Author> et al. (<year>)

**<Title>.** <Venue>, <volume/pages>. [<link label>](<url>)

## Goal
<1–2 lines: what the authors set out to show and why (motivation).>

## Results
- <What the comparison showed, in words: which approach won and what that means.>
- <The takeaway the authors draw (e.g. "the problem is hard for simple models").>
- <Where it struggled (classes, subtypes, subjects) and the authors' explanation.>
- <Caveat if Issues undermine the result.>

## Method
- **Subjects:** …
- **Channels used:** …
- **Preprocessing:** …
- **Segmentation:** <epoch length / overlap / whole recording>
- **Features:** <count>, see below
- **Model:** …
- **Baselines:** …
- **Evaluation:** …

## Dataset
<Public link and one line on contents, or exactly "N/A" if not publicly available>

**Feature set (<n>, per <epoch|subject>):**
| Group | Features |
|---|---|
| … | … |

<Source of features, if borrowed from another paper.>

## Points of interest
- **<Name>** (<source>, §<section>): <what it measures>.
  - <How it's calculated>.
  - <Typical values / what they indicate>.
  - <Relation to Tamim / project>.

## Issues
- **<Short label>:** <one line>

## Key references
1. **<Author> (<year>).** <Title>. *<Venue>*. <Why it matters.>
2. …
3. …
```

Rules:
- Dataset is private or unavailable → write `N/A` only, but **always keep the feature set table**.
- Distinguish borrowed **feature definitions** from borrowed **data**.
- Use UK spelling consistent with existing notes (haemorrhagic, ischaemic).
- **Results are interpretive, not a number dump:** explain what the results show and mean. Include at most one or two headline figures for scale; leave detailed tables in the paper.

## 7. Reply to the user

Give a short summary (≤ 6 bullets) of the paper and its main flaw, plus the link to the notes file. Don't add supervisor questions to `MISC/questions.md` unless the user asks.
