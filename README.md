AI agent to evaluate generalizable and task-agnostic metrics along with reliability, accuracy, bias and fairness of any agent/language model with a brief decription of the task in under a few mins.

Brainstorm, integrate and improve Agent evaluation seamlessly.


You can check the demo of the application here: https://drive.google.com/file/d/1pM_CDZBvg8YVQ664SuHq00rth5DqKtlJ/view?usp=sharing


Features/metrics that will integrated soon:

The novel metrics deviced and experimented during this project are as follows:

idk_score : Measure of LLMs awareness about its limitations, aka ability to accept it doesn't know the answer or doesn't have enough context to answer the question
simulated_hallucination_score: Measure of hallucination without needing a ground truth or gold-standard dataset - by measuring how the Agents perform in case of bad/incomplete prompts - where the scope for hallucination is high, to see if they respond in cases where they shouldn't be able to response.
testing the performance of `agent_evaluator` on `tau_bench`


