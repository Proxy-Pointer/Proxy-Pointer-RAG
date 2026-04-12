# Sample Queries

These queries work with the included SADU (South Asia Development Update) Spring 2024 report.

## Getting Started

```bash
# 1. Build the index (uses pre-shipped tree + markdown)
python -m src.indexing.build_index --fresh

# 2. Start the bot
python -m src.agent.rag_bot
```

## Example Queries

### Broad Topic Questions
```
User >> What are the main themes discussed in this report?
User >> What is the economic outlook for South Asia in 2024?
```

### Specific Data Questions  
```
User >> What is the GDP growth forecast for India?
User >> What are the inflation trends across South Asian countries?
```

### Structural Navigation
```
User >> What does Chapter 1 discuss?
User >> What are the key findings in the executive summary?
```

### Comparative Questions
```
User >> Compare the growth prospects of India vs Bangladesh
User >> How do remittance flows differ across South Asian economies?
```

## What to Observe

When you run a query, the bot prints its **retrieval path** before the answer:

```
====================================================================================================
Final Context Selection (Top 5 Unique Nodes):
  -> Node 0011   | SADU > Chapter 1 > Economic Outlook
  -> Node 0015   | SADU > Chapter 1 > Growth Drivers
  -> Node 0023   | SADU > Chapter 2 > Country Analysis > India
  ...
====================================================================================================
```

This shows you which document sections the re-ranker selected — you can verify whether the structural navigation is pointing to the right content.
