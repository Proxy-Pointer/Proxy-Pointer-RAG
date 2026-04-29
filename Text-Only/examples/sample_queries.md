# Sample Queries

These queries work with the included AMD FY2022 10-K filing (`data/documents/AMD.md`).

## Getting Started

```bash
# 1. Build the index
python -m src.indexing.build_pp_index --fresh

# 2. Start the bot
python -m src.agent.pp_rag_bot
```

## Example Queries

### Numerical Reasoning
```
User >> For AMD, what percentage of total revenue growth from FY2021 to FY2022 is attributable to the Embedded segment?
User >> For AMD, compute the free cash flow for FY2022 and compare it with FY2021.
User >> For AMD, calculate the current ratio for FY2022 and assess liquidity strength.
```

### Multi-hop Numerical
```
User >> For AMD, compute the operating margin for FY2022 and compare it with FY2021. What is the change in percentage points?
User >> For AMD, what proportion of total revenue in FY2022 is attributable to the top two segments combined (Data Center + Client)?
```

### Causal & Attribution Analysis
```
User >> For AMD, quantify how much of the operating income decline in FY2022 is explained by acquisition-related amortization and costs.
User >> For AMD, calculate the proportion of total assets contributed by goodwill and intangible assets in FY2022.
```

### Trend Analysis
```
User >> For AMD, what is the year-over-year growth rate of Client segment revenue from FY2021 to FY2022, and how does it compare to overall company growth?
User >> For AMD, what percentage of total revenue in FY2022 came from the Data Center segment?
```

### Adversarial (Trick Questions)
```
User >> For AMD, estimate whether inventory buildup contributed significantly to cash flow decline in FY2022.
```

## What to Observe

When you run a query, the bot prints its **retrieval path** before the answer:

```
====================================================================================================
Final Context Selection (Top 5 Unique Nodes):
  -> Node 0080   | AMD > FINANCIAL CONDITION > Liquidity and Capital Resources
  -> Node 0089   | AMD > Advanced Micro Devices, Inc. > Consolidated Balance Sheets
  -> Node 0081   | AMD > FINANCIAL CONDITION > Liquidity and Capital Resources > Operating Activities
  ...
====================================================================================================
```

This shows you which document sections the re-ranker selected — you can verify whether the structural navigation is pointing to the right content by checking the hierarchical breadcrumbs.
