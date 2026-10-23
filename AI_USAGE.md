# AI usage

## Tools I used, and for which functions
I used Claude to write the first versions of `audit()` and all six functions in `metrics.py`. It also helped write the tests in `test_mine.py`.

I asked it what should be included in `FRAMING.md`, and then I wrote it on my own.

I ran the tests and the sample audit myself, and then I checked the results against the required specs in the README.

## One place where the AI was wrong, incomplete, or not what the spec said
Claude assumed a rate of 200 transfers a second, whereas the Rawaj company brief says the peak so far was 64 transfers per second. I corrected the AI on this later.

There was also a self-contradiction. It first said that nothing is automatically blocked, but then said that a wrong flag freezes the customer.

## The function I wrote without AI, and why

<!-- Which one, and what you learned by writing it yourself. -->
