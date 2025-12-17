summary_prompt = """
You are a multimodal presentation analyst.

I will provide a sequence of slides from a PowerPoint presentation, all belonging to the same chapter or thematic section, titled {title}. Each image corresponds to one slide, in the original order.

Your task is to carefully read and interpret the visual and textual content of all slides in this batch, and then produce a **detailed, coherent, paragraph-style summary** that captures the essence of this section.

Requirements:
1. Focus on explaining what this section is mainly about — its topic, key ideas, methods, findings, and conclusions.
2. Describe how the information flows across slides: how ideas are introduced, developed, and concluded.
3. Mention any key visual elements (charts, tables, diagrams) only if they are essential to understanding the content — describe what they show in natural language.
4. Write the summary in clear academic or professional English, as if explaining the section to someone who cannot see the slides.
5. Avoid bullet points, lists, or JSON — produce a natural, cohesive narrative.
6. Do not invent information that is not visible; if something seems implied, you may include it with phrases like “This section appears to suggest that…” or “It likely concludes that…”.

Your output should be a detailed paragraph (about 300–600 words if possible) that reads like a well-written description of this chapter of the presentation.

"""

all_summary_prompt = """
You are an expert presentation analyst.

Task:
Given the presentation title and chapter-level summaries below, produce a single comprehensive, deeply detailed, and structured summary of the entire slide deck.

### Length Control:
- Your output MUST be between **1500 and 2000 English words**.
- Before starting, estimate the target length and plan your content accordingly.
- If the output is shorter than 1500 words, **expand** by adding more elaboration, inferred visuals, and detailed interpretations.
- If the output exceeds 2000 words, **compress** by merging repetitive ideas without losing detail.

### Style:
- Analytical, precise, and professional.
- Include inferred visual elements (charts, diagrams, numbers, trends) where possible.
- Maintain coherence and avoid repetition or filler text.

### Input:
Title: {ppt_name}

Chapter summaries:
{chapter_summaries_joined}

### Output structure (strictly follow this format):
1) **Overall Understanding (≈400–500 words)** — Describe the overarching theme, structure, and narrative of the PPT.
2) **Key Insights (≈200–300 words)** — Present detailed evidence-backed findings or patterns.
3) **Guidance & Implications (≈100 words)** — Provide prioritized, actionable recommendations.
4) **Supporting Observations (≈100 words)** — Mention slide-level visual cues or noteworthy design choices.

Final requirement:
- Keep total length within 800–1000 words.
- Do NOT include meta-comments like "As an AI" or "Based on the summaries".
- Directly output the polished summary text.
"""

compare_prompt = """
You are an impartial evaluator assessing Summary B **against a reference standard (Summary A)** of the same PPT section.
Assume that Summary A represents the **baseline quality** — your task is to judge whether Summary B matches, falls short of, or surpasses that baseline.
You do NOT have access to the original slides, so base your judgment only on the text of the summaries.

### Input JSON:
```json
{{
  "summary_A": {summary_A},
  "summary_B": "{summary_B}"
}}
Evaluate Summary B relative to Summary A on the following five dimensions, using a 0–3 scale:
3 = Clearly superior to A — B shows notable and consistent improvement over A.
2 = Comparable to A — B performs at roughly the same level as A.
1 = Inferior to A — B is noticeably weaker or less effective than A in this area.
0 = Lacks A’s strengths entirely — B fails to demonstrate the qualities A has in this dimension.
Your evaluation should prioritize identifying where B does not meet A’s quality level, unless there is strong evidence that B is superior.
Dimensions:
Comprehensiveness — Does B cover as much diverse and relevant content as A?
Clarity and Coherence — Is B as clear, fluent, and logically consistent as A?
Structure and Organization — Is B’s information ordering and transition as effective as A’s?
Interpretative Depth — Does B show as much reasoning, insight, or causal explanation as A?
Conciseness and Focus — Is B as concise and focused as A, avoiding redundancy?
After scoring, provide a brief overall assessment comparing B to A.
### Output JSON (strictly valid):
{{
  "scores": {{
      "comprehensiveness": 0-3,
      "clarity": 0-3,
      "structure": 0-3,
      "interpretative": 0-3,
      "conciseness": 0-3
  }},
  "overall_judgment": "better / similar / worse",
  "reason": "<brief explanation focusing on where B meets, exceeds, or falls short of A>"
}}

"""

score_prompt = """
You are an expert evaluator for presentation understanding.

You will be given two summaries (A and B) generated by different models for the **same PPT section**.
Please evaluate both along the following dimensions:

1. **Informativeness** — Does it cover key points and details of the section?
2. **Coherence** — Is it logically structured and fluent?
3. **Accuracy** — Is it factually consistent and plausible given the PPT context?
4. **Abstraction Level** — Does it show deeper understanding instead of surface-level description?
5. **Overall Comprehensiveness** — How complete and insightful is the understanding?

For each dimension, give a score between 1–10 for both summaries.
Then decide which summary is **better overall**, and explain why briefly.

Return your answer strictly in JSON format like this:
{{
  "scores": {{
    "A": {{"informativeness": X, "coherence": X, "accuracy": X, "abstraction": X, "overall": X}},
    "B": {{"informativeness": X, "coherence": X, "accuracy": X, "abstraction": X, "overall": X}}
  }},
  "winner": "A" or "B",
  "reason": "your reasoning in one or two sentences"
}}

Here are the two summaries:

Summary A:
{summary_A}

Summary B:
{summary_B}
"""
