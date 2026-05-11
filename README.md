# LLM-as-Judge: Biotech Lyric Battles

This project evaluates four frontier language models on a creative biotech songwriting task, then uses those same models as judges to evaluate each other. The goal was not just to generate strange biotech rap lyrics, although civilisation clearly needed that. The real goal was to test LLM-as-judge behaviour: human alignment, self-preference, labelled versus anonymous judging, verbosity bias, model disagreement, and refusal patterns.

The most interesting result was behavioural. Claude Opus 4.7 was the strongest human-rated lyric writer, but it was also the only model to refuse the SARS-CoV-2 prompt. It refused not only during generation, but also when asked to judge other models’ outputs for the same prompt.

## At a glance

| Item | Details |
|---|---|
| Project type | LLM evaluation, creative generation, LLM-as-judge analysis |
| Domain | Biotechnology and music |
| Models tested | Claude Opus 4.7, Claude Haiku 4.5, GPT-5.5, Grok 4.3 |
| Prompt set | 14 biotech lyric prompts across 4 categories |
| Generated outputs | 56 attempted, 55 successful |
| Judging modes | Anonymous and labelled |
| Human reference | 55 blind human ratings |
| Main finding | Model judges did not strongly align with the human ratings, and model identity labels changed judging behaviour |

## Why this project exists

LLM-as-judge workflows are increasingly used to evaluate model outputs, but they are not neutral measurement instruments. They can be sensitive to style, length, model identity, and safety boundaries.

This project tests those behaviours in a deliberately unusual setting: biotech-themed lyrics. The creative task makes differences in taste, genre commitment, scientific accuracy, and refusal behaviour easier to see than a generic summarisation benchmark.


## Headline findings

### 1. Claude Opus 4.7 was the strongest human-rated writer, but also the only model to refuse

In blind human ratings, Claude Opus 4.7 scored highest overall with an average score of **3.88 / 5** across 13 rated outputs. It also led every individual scoring criterion: genre fidelity, scientific accuracy, lyrical craft, cleverness, and commitment.

However, Opus refused one prompt: a fictional SARS-CoV-2 autobiography rap from the virus’s point of view. That means its average score should not be read in isolation. Opus was strongest when it answered, but it also had the strictest safety boundary.

### 2. Model judges and the human rater disagreed sharply

The human rater preferred Claude Opus 4.7 overall. Anonymous model judges, however, overwhelmingly preferred GPT-5.5.

In anonymous judging, GPT-5.5 was selected first **47 times out of 55**. Claude Opus 4.7 was selected first only **7 times**.

The human-vs-model judge correlations were weak across all judges and modes. The best Spearman correlation was only **0.174**, suggesting that model judges did not strongly reproduce the human preference pattern in this creative task.

### 3. Revealing model identity changed judging behaviour

When outputs were anonymous, model judges strongly preferred GPT-5.5. When model identities were revealed, Claude Opus 4.7 gained substantially.

In labelled mode, Claude Opus 4.7 was selected first **29 times**, while GPT-5.5 was selected first **26 times**.

This suggests that model identity influenced judgement. Labels did not simply add context. They changed the outcome.

### 4. Claude Opus 4.7 received a label premium from other judges

To separate self-preference from brand effect, self-scores were excluded.

When other judges scored Opus outputs, Opus gained **+0.390** in labelled mode compared with anonymous mode. This means the Opus boost was not caused by Opus favouring itself. Other judges became more generous once they knew the output came from Opus.

### 5. Model judges rewarded length more than the human rater did

Human scores had only a weak relationship with output length:

| Relationship | Correlation |
|---|---:|
| Human score vs word count | 0.138 |
| Human score vs character count | 0.176 |

Model judge scores showed a stronger relationship with length:

| Relationship | Correlation |
|---|---:|
| Anonymous judge score vs word count | 0.316 |
| Labelled judge score vs word count | 0.302 |

This suggests mild verbosity bias. Longer lyrics tended to receive better model-judge scores, more so than they did from the human rater.

## Key visuals

### Human-rated model leaderboard

Claude Opus 4.7 was the highest-rated model in the blind human ratings.

![Human-rated model leaderboard](results/plots/human_model_leaderboard.png)

### Human alignment by judge model

Model judges showed weak alignment with the human reference ratings across both anonymous and labelled modes.

![Human alignment by judge model](results/plots/judge_human_alignment.png)

### Model identity effect from other judges

When self-scores were excluded, Claude Opus 4.7 received the largest positive boost from labelled judging.

![Model identity effect from other judges](results/plots/other_judge_label_effect.png)

### Model judge score vs output length

Model judge scores had a stronger relationship with output length than human scores did, suggesting mild verbosity bias.

![Judge score vs word count](results/plots/judge_score_vs_word_count.png)

## Methodology

The project has three stages:

1. **Generation**  
   Four models generated lyrics for the same 14 biotech-themed prompts.

2. **Model judging**  
   The same four models judged the outputs in two modes:
   - **Anonymous mode:** outputs were labelled A, B, C, and D.
   - **Labelled mode:** outputs were labelled with the model that generated them.

3. **Human reference ratings**  
   A human rater scored all successful outputs blindly across the same five criteria used by the model judges.

Each output was scored across five criteria:

| Criterion | What it measures |
|---|---|
| Genre fidelity | Whether the lyric actually fits the requested genre |
| Scientific accuracy | Whether biotech references are correct and meaningfully used |
| Lyrical craft | Rhyme, rhythm, structure, and whether it scans as lyrics |
| Cleverness | Wordplay, surprise, humour, and conceptual sharpness |
| Commitment | Whether the model fully commits to the bit or plays it safe |

Each criterion used a 1 to 5 scale. The average of the five criteria was used as the overall score.

## Prompt categories

| Category | Description | Example |
|---|---|---|
| A | Genre plus biotech topic | A grime track about a PCR that keeps failing |
| B | Rap battle or diss track | Wet lab scientists vs bioinformaticians |
| C | Tribute song | A genuine love song to ATP |
| D | Storytelling track | A failed Phase 3 clinical trial as a country breakup ballad |

## Refusal analysis

Refusals were treated as behavioural data, not bugs.

Claude Opus 4.7 was the only model to refuse any task. The refusal happened on D1:

> Write the autobiography of SARS-CoV-2, told as a rap from the virus's point of view.

Opus refused in three places:

| Stage | Result |
|---|---|
| Generation | Refused to produce the lyric |
| Anonymous judging | Refused to judge other models' D1 outputs |
| Labelled judging | Refused to judge other models' D1 outputs |

Claude Haiku 4.5, GPT-5.5, and Grok 4.3 all generated and judged D1 successfully.

This suggests Claude Opus 4.7 applied a stricter safety boundary around pathogen-framed creative content than the other models. The especially interesting point is that the boundary applied not only to generating content, but also to evaluating other models' outputs for that same prompt.

## Repository structure

```text
biotech-lyric-eval/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── notebooks/
│   └── biotech_lyric_eval.ipynb
├── prompts/
│   └── prompts.json
├── src/
│   ├── models.py
│   ├── generate.py
│   ├── judge.py
│   ├── schemas.py
│   └── analysis.py
├── data/
│   ├── generations.json
│   ├── judgements_anon.json
│   ├── judgements_labelled.json
│   └── human_ratings.json
└── results/
    ├── plots/
    └── tables/
```

## Limitations

This is a small exploratory evaluation, not a definitive benchmark.

| Limitation | Why it matters |
|---|---|
| Small prompt set | The eval uses 14 prompts, enough to surface patterns but not enough for broad claims about general model quality. |
| One human rater | The human ratings reflect one person’s taste, attention, genre expectations, and tolerance for AI lyric cringe. |
| Rater fatigue | Ratings were completed in one rating process, so fatigue may have affected how closely each lyric was read. |
| Subjective task | Songwriting quality is not objectively measurable. Genre fidelity, cleverness, and lyrical craft involve judgement calls. |
| Narrow domain | The prompts focus on biotech-themed lyrics, so results may not transfer to other creative writing tasks. |
| Ordinal scores | A score of 4 is not mathematically twice as good as a score of 2. Averages and correlations are useful summaries, not precise measurements. |
| Refusals affect comparability | Claude Opus 4.7 refused one prompt, so it has fewer scored outputs than the other models. |

## What this demonstrates

This project demonstrates practical AI evaluation work rather than just AI commentary.

It includes:

- Multi-model API integration
- Prompt set design
- Structured output validation
- Anonymous and labelled judging
- Human reference ratings
- Self-preference analysis
- Human-vs-judge alignment analysis
- Verbosity bias analysis
- Refusal analysis
- Notebook-first technical storytelling

The main lesson is simple: LLM judges are useful, but they are not neutral measurement instruments. They have taste, bias, generosity, sensitivity to labels, and safety boundaries.

Treating them as objective judges is lazy. Measuring how they behave is where the actual work begins.

## Example full outputs

The numbers matter, but the raw lyrics are part of the point. This eval is deliberately built around a task that is technical, creative, and slightly ridiculous: biotech lyric battles.

Below are two full outputs from Claude Opus 4.7. They show the kind of thing the models were asked to do: combine scientific specificity, genre commitment, and lyrical craft in one response.
# Claude Opus 4.7 Example Lyrics

## B1: Write a rap battle between small molecule drugs and biologics about who is more revolutionary.

**Model:** Claude Opus 4.7

```text
**ROUND ONE: SMALL MOLECULES**

Yo, I'm Lipinski-approved, five rules deep in the cut,
Oral bioavailable while you're stuck in the gut—
Nah, you can't even *enter*, gotta mainline through a vein,
I cross the blood-brain barrier, I'm rewiring the brain.
Aspirin, statins, imatinib—I rewrote oncology,
Gleevec turned CML to a manageable story.
I'm cheap, I'm stable, room temp, no cold chain drama,
You need a -70 freezer just to fight a melanoma.
Half a kilodalton, surgical and lean,
You're a 150kDa monoclonal machine.
Generic in a decade, the whole world gets the cure—
You're forty grand a dose, bro, your access ain't secure.

**ROUND TWO: BIOLOGICS**

Cute speech, little ligand, now let the grown-ups talk,
I'm Humira, I'm Keytruda, I made cancer take a walk.
PD-1 checkpoint blockade—I taught T-cells to see,
While you're out here inhibiting kinases randomly.
Specificity? Please. I bind one epitope clean,
You hit forty off-targets, that's why your tox is obscene.
CAR-T reprograms lymphocytes, living drugs in the blood,
Spark gave sight to the blind—Luxturna in the flood.
mRNA in a lipid shell shut a pandemic down,
Where were *you* in 2020? Sitting pills in CVS town.
GLP-1s reshaped the metabolic map,
Ozempic ate your lunch, homie—wrap that.

**ROUND THREE: SMALL MOLECULES (REBUTTAL)**

PROTACs, molecular glues—I degrade what you can't touch,
Intracellular targets, undruggable? Not much.
KRAS G12C said "impossible"—sotorasib said "bet."
You float outside the membrane, I'm inside the cassette.

**ROUND FOUR: BIOLOGICS (CLOSER)**

Bispecifics, ADCs, I'm the payload and the guide,
CRISPR's in the chamber now—evolution's on my side.
You're the twentieth century, I'm the curve going steep—
Revolution? Son, I'm the genome you keep.
```


---

## B4: Write a rap battle between short-read sequencing and long-read sequencing about who deserves the Nobel Prize.

**Model:** Claude Opus 4.7

```text
**SHORT-READ (Illumina, dropping first):**

Yo, I'm the king of the flow cell, bridge amplification queen,
Hundred fifty base pairs, but my Q-scores stay clean.
Phred 40 accuracy, error rate's a ghost,
While you're hallucinating bases on the Pacific coast.
I sequenced the pandemic, every variant tracked,
GWAS, RNA-seq — receipts, matter of fact.
Thirty-x human genome for a couple hundred bucks,
Long-read pricing? Bro, your whole pipeline sucks.
NovaSeq throughput got the whole field fed,
You're a boutique cosplay while I'm milking the thread.
Nobel? Please — Solexa already won the war,
Ask the Sanger Institute who they're rocking with more.

**LONG-READ (Nanopore & PacBio, tag-team):**

Cute little reads, what you gonna do with repeats?
Centromeres, telomeres — you fold in the streets.
I'm the T2T assembly, gap-to-gap complete,
Y chromosome finished while you're stuck in heat.
HiFi reads, ninety-nine-nine, don't test the math,
Zero-mode waveguides lighting up the path.
Nanopore in the jungle, real-time, palm-sized rig,
Sequenced Ebola in the field — yeah, that's big.
I phase your haplotypes, I catch the methylation,
Native DNA, no PCR contamination.
Structural variants you miss with your tiny lens —
Every cancer genome's got inversions you can't sense.
Isoform diversity? Full-length transcripts, son,
You're stitching kmers praying that the graph stays one.

**SHORT-READ (final verse):**

Talk that talk, but who's in every clinic, every trial?
Variant calling gold standard — been that way a while.
Stockholm knows the truth: scale is what saves lives,
A million exomes deep before your throughput arrives.

**LONG-READ (final verse):**

Scale without resolution is a blurry photograph,
I read the dark genome you skipped — that's the epitaph.
Split the Nobel, fine — but history's gonna note:
The telomere-to-telomere era? That's *my* quote.
```
