# Framing: Rawaj Pay mule detection

## Decision
For each instant transfer, score the receiving account's mule risk within 300 ms. The top N transfers per day go to Dina's five analysts, with N set by their capacity (assumed ~200 a day, to be confirmed with Dina). A flag means "review", not "block": nothing stops a transfer or freezes a wallet without a human decision, and anything that slows a customer's transfer needs Sherif's agreement. A missed mule costs the victim's money (2025: EGP 61M reported losses over 2,310 scam reports, about EGP 26,000 each) plus reimbursements and reputation. A false flag costs analyst time. A wrongful freeze costs a complaint and merchant trust, and I don't have a figure for that yet.

## Metric
Precision at the analysts' daily capacity, reported together with recall at the same threshold, since they can only review N cases. To pick the threshold, expected cost: false alarms × cost_fp + misses × cost_fn, with illustrative costs of 300 and 20,000 until Dina gives real ones. Capacity caps the number of flags even if cost alone would flag more. Not accuracy: with only a few percent mules, "flag nothing" is about 97% accurate and catches no one.

## Baseline
Hares, Rawaj's rules engine, handles card fraud and obvious account takeovers but not a customer coached by phone to send her own money. The model must beat it on the same decision: more mules caught in the same ~200 reviews a day, with no more wrongly frozen customers. Labels are only customer-reported fraud, so measured precision is a lower bound.

## Classical ML or LLM
A classical model on tabular features (receiver account age, in-degree, amount, channel), possibly alongside Hares rules where they already work. Every transfer needs a fast, cheap score (Rawaj's peak so far is 64 transfers per second), and a classical model lets analysts get a written reason for each flag. An LLM could write the analyst's triage note afterwards, but it shouldn't make the detection. Transaction data should not leave Rawaj's environment for an outside service without asking Dina first.