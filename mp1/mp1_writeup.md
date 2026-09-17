### 1. Which strategy won, and on what dimension? (Accuracy? Parse rate? Cost?)
- Structured and CoT tied on accuracy (2.9/3), parse rate (100%), and judge score (4.0/4).
- Structured was cheaper and slightly faster than CoT.
- Zero-shot was cheapest overall but had only 10% parse success.

### 2. What surprised you? Either a strategy worked better than expected, or worse, or a specific snippet failed in a way you didn't predict.
The real surprise is that zero-shot parsed only 1 of 10 responses despite being asked for JSON.

### 3. For *your* capstone domain, which strategy would you reach for first? Justify in 2-3 sentences.
I would start with structured prompting. It produced reliable JSON and tied for the highest accuracy while costing less and running slightly faster than CoT.


### 4. If you had another day, what would you try next? (Different model? More snippets? Different prompts?)
I would try the following:

- More ambiguous snippets
- More missing-value examples
- Additional experience ranges
- Repeated runs to measure consistency
- Another model using the same prompts and dataset