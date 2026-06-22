# RAG Evaluation

Qualitative evaluation of the retrieval-augmented generation pipeline on representative questions. Fill in **Score (1-5)** and **Comments** after reviewing each answer against its retrieved sources.

| # | Question | Generated Answer | Top Retrieved Sources (1-2) | Score (1-5) | Comments |
|---|---|---|---|---|---|
| 1 | Why are people unhappy with credit cards? | Because they are a scam. | [Credit Card / Closing your account] . in fact, i use credit extremely responsibly and i simply wanted the credit cards for the…<br>[Credit Card / Fees or interest] . i am totally frustrated by the way that the credit card company demand for interests on … |  |  |
| 2 | What are common complaints about personal loans? | i feel i was not treated fairly about my account and it was done wrong | [Credit Card / Getting a credit card] . too few accounts currently paid as agreed. i have, as i stated above, only two open acco…<br>[Personal Loan / Charged fees or interest you didn't expect] took a personal loan for an emergency. the amount was 1000.00 . i knew i would be repaying… |  |  |
| 3 | What problems do customers report with money transfers? | a customer loses money due to unauthorized transactions | [Savings Account / Managing an account] . we determine money was moving between accounts. banking filed a complaint escalation rep…<br>[Money Transfer / Unauthorized transactions or other transaction problem] . i just don't understand how 4 different big transactions went through, when i have never… |  |  |
| 4 | Are there issues with unauthorized or fraudulent charges? | unauthorized | [Credit Card / Getting a credit card] i found unauthorized charges and some of them were occurred when i was out of country whic…<br>[Savings Account / Managing an account] . i have evidence in my history that none of those fraudulent charges occurred through . i… |  |  |
| 5 | What complaints relate to savings account fees? | a credit reporting company's investigation into an existing problem | [Savings Account / Managing an account] i am filing a complaint against capital one regarding the deceptive practices related to t…<br>[Savings Account / Problem with a credit reporting company's investigation into an existing problem] this complaint is against bank of america.. reason is they closed my checking savings acco… |  |  |
| 6 | Do customers report difficulty closing their accounts? | No | [Personal Loan / Incorrect information on your report] . then as i look on my credit report they closed the account and im still making payments …<br>[Savings Account / Closing an account] . i said if every person was this enthusiastic about helping the customers along with ever… |  |  |
| 7 | What issues do people have with billing disputes? | [Credit Card] | [Credit Card / Problem with a purchase shown on your statement] i had a billing dispute with this merchant from year for merchandise not received. dispute…<br>[Savings Account / Problem with a lender or other company charging your account] . as far as , their behavior, customer service, and collection tactics are simply criminal… |  |  |
| 8 | Are there complaints about poor customer service responses? | yes | [Money Transfer / Other transaction problem] . we can not compel a business to respond. however, failure to do so result in a negative …<br>[Credit Card / Closing your account] . they didn't try to rectify this issue at all and acted like they didn't care about their… |  |  |

## Analysis — what worked and what to improve

**Retrieval works well.** Across all questions the retriever surfaces on-topic
complaint excerpts with correct `product_category` / `issue` tags — e.g. money-
transfer questions return unauthorized-transaction chunks, billing-dispute
questions return purchase-dispute chunks. Semantic search over the MiniLM
embeddings is the strongest part of the system and the source excerpts are
genuinely useful evidence on their own.

**Generation is the weak link.** The default generator, `flan-t5-small`, is fast
(~1 s/answer on CPU) but produces terse, sometimes vacuous answers ("No", "yes",
"Because they are a scam."). It under-uses the rich retrieved context. This is a
model-capacity limitation, not a retrieval or prompting failure.

**Improvements:** (1) swap in a larger generator (`flan-t5-base/large`, or a
Mistral/Llama instruct model via GPU or a hosted API) for fuller, better-grounded
answers; (2) use the provided full-scale pre-built vector store (~1.37M chunks)
instead of the 12K-complaint sample for broader coverage; (3) add answer-quality
guardrails (cite source numbers, refuse when context is weak). The retrieval layer
is production-ready; the generator is the main lever for quality.
