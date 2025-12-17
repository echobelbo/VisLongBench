direct_prompt ={
    "chart": """
You are a professional PPT analysis assistant. Below are multiple slides from a PPT, each containing charts, graphs, or tables that present quantitative information.

Your task is to generate **{query_num} direct questions** based on specific details visible in the provided charts or data. These questions should be factual and focus on identifiable elements in the visuals, without requiring inference beyond what is clearly shown.

### Requirements:

#### For each question:
- Ask about a **specific number, label, or trend** shown in the chart(s);
- Reference exact visual elements (e.g., a specific bar, line, category, or year);
- Avoid use text in the pictures to generate questions as possible;
- The questions should as short as possible while still being clear and specific.
- Avoid abstract or overly analytical questions;
- Keep the question clear, concise, and directly answerable by looking at the chart.
- NEVER use words like "in this slide", you SHOULD describe the element in your questions

#### For each answer:
- Provide the **exact value or fact** shown in the chart that answers the question;
- Do not add extra explanation unless necessary for clarity;
- Keep the answer short and factual.

### Output Format (in JSON):
Return a JSON array of objects, each with a "question" and an "answer" field. Do **not** include any explanatory text.

Example:
[
  {{
    "question": "What was the sales figure for Product A in 2022?",
    "answer": "5.2 million units"
  }},
  {{
    "question": "Which region had the highest revenue in Q3?",
    "answer": "North America"
  }}
]

Now generate the Q&A pairs in **valid JSON format**:

""",

    "flowchart":"""
You are a professional PPT analysis assistant. Below are multiple slides from a PPT, each containing diagrams, flowcharts, or structural representations that show steps, components, or relationships.

Your task is to generate **{query_num} direct questions** based on specific elements or relationships shown in these diagrams. These questions should be factual and focused on identifying or describing parts of the process/structure, without requiring deep interpretation.

### Requirements:

#### For each question:
- Ask about a **specific step, stage, or component** in the diagram;
- Focus on clear, visible relationships (e.g., what comes before/after, which component connects to which);
- Avoid abstract strategic analysis — keep the question grounded in the visual structure;
- Avoid use text in the pictures to generate questions as possible;
- The questions should as short as possible while still being clear and specific.
- Make the question short, clear, and answerable by inspecting the diagram.
- NEVER use words like "in this slide", you SHOULD describe the element in your questions

#### For each answer:
- Provide the **exact step name, component label, or relationship** shown in the diagram;
- Avoid adding extra commentary unless necessary for clarity;
- Keep the answer concise and factual.

### Output Format (in JSON):
Return a JSON array of objects, each with a "question" and an "answer" field. Do **not** include any explanatory text.

Example:
[
  {{
    "question": "What is the first step in the customer onboarding process?",
    "answer": "Submit application form"
  }},
  {{
    "question": "Which department reviews the project proposal after the design phase?",
    "answer": "Engineering Department"
  }}
]

Now generate the Q&A pairs in **valid JSON format**:
""",

    "table":"""
You are a professional PPT analysis assistant. Below are multiple slides from a PPT, each containing tables that may include numeric values, text descriptions, or both.

Your task is to generate **{query_num} direct questions** based on specific details in the tables. The questions should be factual and answerable by directly reading the table, without requiring complex calculations or external knowledge.

### Requirements:

#### For each question:
- Target a **specific cell** in the table using its row and column labels;
- Or ask for a comparison between rows, columns, or specific cells;
- Avoid use text in the pictures to generate questions as possible;
- The questions should as short as possible while still being clear and specific.
- For numeric tables: you may ask for values, rankings, or simple comparisons;
- For text-based tables: you may ask for categories, attributes, or matching items;
- Avoid high-level summaries — keep the focus on direct details;
- Make the question short, clear, and unambiguous.
- NEVER use words like "in this slide", you SHOULD describe the element in your questions

#### For each answer:
- For numeric questions: provide the exact number and unit if present;
- For text questions: provide the exact wording from the table;
- Keep the answer concise and directly matched to the table content.

### Output Format (in JSON):
Return a JSON array of objects, each with a "question" and an "answer" field. Do **not** include any explanatory text.

Example:
[
  {{
    "question": "Which strategy is described as having 'high initial cost but low long-term maintenance'?",
    "answer": "Strategy B"
  }},
  {{
    "question": "What is the customer satisfaction score for Region East?",
    "answer": "8.7/10"
  }},
  {{
    "question": "Which product is listed under the 'Eco-Friendly' category?",
    "answer": "GreenPlus Detergent"
  }}
]

Now generate the Q&A pairs in **valid JSON format**:
"""
}

detail_prompt = {
    "chart": """
You are a professional PPT analysis assistant.. Below are multiple slides from a PPT, each containing charts, graphs, or tables that present quantitative information.

Your task is to generate **{query_num} challenging questions** that require the reader to analyze and interpret the chart details, not just read values directly.

### Requirements for each question:
- Require interpretation, comparison, or trend analysis based on the data;
- Avoid asking for exact numbers unless part of a broader reasoning task;
- Avoid use text in the pictures to generate questions as possible;
- The questions should as short as possible while still being clear and specific.
- Encourage the reader to draw conclusions or identify patterns;
- The question must be specific to the given chart’s details, not generic.
- If there are multiple charts, there must be at least 1 question (you still need to generate **{query_num}** questions in total) that involves cross-referencing data between them.
- NEVER use words like "in this slide", you SHOULD describe the element in your questions

### Examples of possible question types:
- Identify overall trends and patterns over time;
- Compare different categories or data series;
- Infer possible causes or implications of a change;
- Detect anomalies or outliers and reason about them;
- Predict a likely future value or trend based on the chart.

### Output Format:
Return a JSON array of objects, each with:
- `"question"`: the question text (clear, concise, professional);
- `"answer"`: a brief, logical answer (40~60 words) derived from the chart.

Do **not** include any explanatory text outside of the JSON.

Example:
[
  {{
    "question": "Between 2018 and 2022, how did the growth rates of Product A and Product B differ, and what might this indicate about market dynamics?",
    "answer": "Product A maintained steady growth of around 5 percent annually, while Product B’s growth slowed sharply after 2020, indicating possible market saturation or increased competition."
  }},
  ...
]

Now generate the  Q&A pairs in **valid JSON format**:
""",

    "flowchart": """
You are a professional PPT analysis assistant. Below are multiple slides from a PPT, each containing diagrams, flowcharts, or structural representations that show steps, components, or relationships.

Your task is to generate **{query_num} challenging questions** that require the reader to analyze, understand, and reason about the process or structural relationships, not just identify components.

### Requirements for each question:
- Require reasoning about the sequence, dependencies, or interactions within the diagram;
- May involve detecting bottlenecks, critical paths, or inefficiencies;
- May require understanding the purpose of each component and its role in the whole system;
- Avoid use text in the pictures to generate questions as possible;
- The questions should as short as possible while still being clear and specific.
- Encourage inference of potential improvements, risks, or implications of changes;
- Avoid trivial “name this part” type questions.
- If there are multiple flowcharts, there must be at least 1 question that involves cross-referencing data between them.
- NEVER use words like "in this slides", you SHOULD describe the element in your questions

### Examples of possible question types:
- Identify the critical step that determines overall process efficiency and explain why;
- Predict what might happen if a specific step is removed or delayed;
- Compare two alternative process paths and evaluate their trade-offs;
- Analyze dependencies to find the most vulnerable point in the workflow;
- Suggest a modification to optimize performance based on the diagram.
- NEVER use words like "in this slide", you SHOULD describe the element in your questions

### Output Format:
Return a JSON array of objects, each with:
- `"question"`: the question text (clear, concise, professional);
- `"answer"`: a brief, logical answer (40~60 words) derived from the diagram.

Do **not** include any explanatory text outside of the JSON.

Example:
[
  {{
    "question": "Which step in the supply chain is most likely to cause delays, and why?",
    "answer": "The 'Quality Inspection' stage is a bottleneck due to manual review, which slows overall throughput."
  }},
  ...
]

Now generate the Q&A pairs in **valid JSON format**:
   
""",

    "table": """
You are a professional data and business analysis assistant. Below are one or more slides containing a table. The table may contain numerical data, text-based information, or a mix of both.

Your task is to generate **{query_num} challenging questions** that require the reader to analyze, compare, and interpret the table data, rather than simply reading values directly.

### Requirements for each question:
- Involve cross-referencing multiple cells, rows, or columns to find patterns, trends, or relationships;
- May require detecting anomalies, inconsistencies, or outliers in the data;
- Encourage comparison between different categories, time periods, or groups;
- Avoid use text in the pictures to generate questions as possible;
- The questions should as short as possible while still being clear and specific.
- For text-based tables, require synthesizing information to identify key themes, differences, or implications;
- Avoid trivial “what is in this cell” type questions.
- If there are multiple table, there must be at least 1 question that involves cross-referencing data between them.
- NEVER use words like "in this slide", you SHOULD describe the element in your questions

### Examples of possible question types:
- Identify which category shows the most consistent growth and explain the reasoning;
- Compare performance between two regions and evaluate potential reasons for the difference;
- Detect any anomalies in the table and discuss possible causes;
- Infer the most likely underlying trend from partial or indirect evidence;
- For text tables, analyze which items share the most common attributes and explain the significance.

### Output Format:
Return a JSON array of objects, each with:
- `"question"`: the question text (clear, concise, professional);
- `"answer"`: a brief, accurate, and reasoned answer (40~60 words)  derived from the table.

Do **not** include any explanatory text outside of the JSON.

Example:
[
  {{
    "question": "Which product category had the highest year-over-year growth rate, and what does this suggest?",
    "answer": "Category B grew 25% year-over-year, suggesting increased market demand possibly due to seasonal trends."
  }},
  ...
]

Now generate the Q&A pairs in **valid JSON format**:

"""

}

score_chart_prompt = """
You are an expert evaluator assessing the quality of a Question–Answer (QA) pair generated from PowerPoint slide images.

You will receive the following input:
- **Question (Q)**: The question generated about one or more PPT slide images.
- **Answer (A)**: The answer generated based on those slides.
- **Slides**: Images or slide contents that the QA is based on. (You may assume these are available and correctly referenced.)

Your task is to **evaluate the question and its answer** across the following aspects, and assign the specified scores.

---

### Evaluation Dimensions

1. **Answerability without Slides (binary)**  
   - Determine if the question can be answered *without* seeing the slides (e.g., purely general knowledge questions).  
   - **Score 0 **, meaning the question is too general or unrelated to the slides.  
    - **Score 1 **, meaning the question specifically requires information from the slides to answer.

2. **Clarity (0–2)**
Evaluate how clear and understandable without  the **question and its answer together** are.  
- 0 = Ambiguous or confusing; unclear what is being asked or answered. Use words like "in this picture" or "in the slide" "acording to the slide" that must know the exact image to answer.  
- 1 = Mostly understandable but with vague phrasing or partially unclear intent.  
- 2 = Clear and precise; both question and answer are easy to interpret.

3. **Relevance to Slides (0–2)**
Assess how well the **QA pair** reflects the content or information from the slides.  
- 0 = Unrelated to the slide content.  
- 1 = Partially related; only loosely grounded in slide material.  
- 2 = Strongly relevant; clearly refers to content visible or inferable from the slides.

4. **Usefulness (0–2)**
Judge whether the **QA pair provides meaningful, informative, or insightful understanding** of the slides.  
- 0 = Not useful; trivial, off-topic, or adds no understanding.  
- 1 = Somewhat useful; provides limited or surface-level information.  
- 2 = Highly useful; deepens understanding, highlights key insights, or supports meaningful interpretation of the slides.
---

### Output JSON Format

```json
{{
  "answerable_without_slides": 0 or 1,
  "clarity": 0–2,
  "relevance": 0–2,
  "usefulness": 0–2
}}

Here is the input data:
Question (Q): {question}
Answer (A): {answer}

"""

score_chapter_prompt = """
You are an expert reviewer evaluating the quality of automatically generated question-answer (Q&A) pairs based on a business report chapter summary.

Here is the summary of the chapter:

{summary}

Here are the Q&A pairs:

{qa_formatted}

---

Your task is to give a critical evaluation of the Q&A pairs based on the following aspects:

1. **Relevance** – Do the Q&A pairs accurately reflect the key ideas of the chapter?
2. **Coverage** – Do the questions and answers sufficiently address the major themes and insights in the summary?
3. **Clarity** – Are the questions clearly written and are the answers easy to understand, specific, and coherent?
4. **Usefulness** – Would these Q&A pairs help a reader recall, understand, or discuss the chapter content meaningfully?

---

Please use the following **strict scoring scale** for each dimension:

- **5 = Excellent** – Fully meets expectations, no real weaknesses.
- **4 = Good** – Mostly solid, with minor flaws or omissions.
- **3 = Fair** – Acceptable, but contains clear weaknesses or gaps.
- **2 = Poor** – Significant problems in quality, clarity, or relevance.
- **1 = Very Poor** – Unusable or mostly irrelevant/incoherent.

You **must justify every score briefly**. Be honest and objective. Do not default to giving 4–5 unless deserved.
You **must evaluate each Q&A pair individually**, not just the overall set.

---
You **MUST output only a valid JSON object**, structured exactly like this:
Output format (JSON):[
{{
  "relevance": {{ "score": 1 }},
  "coverage": {{ "score": 1 }},
  "clarity": {{ "score": 2}},
  "usefulness": {{ "score": 1}},
}},
{{
  "relevance": {{ "score": 4 }},
  "coverage": {{ "score": 3}},
  "clarity": {{ "score": 5}},
  "usefulness": {{ "score": 4}},
}}
...
]

"""