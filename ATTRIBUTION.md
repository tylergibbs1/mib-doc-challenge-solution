# Attribution

This solution is original work built on the public challenge materials. One
implementation idea was adapted after reviewing public MIT-licensed challenge
solutions:

- The separation between policy-time missing values and output-only
  categorical priors was inspired by the `OUTPUT_ONLY_FALLBACKS` boundary in
  [Abhishek21g/mib-doc-challenge-solution](https://github.com/Abhishek21g/mib-doc-challenge-solution),
  which attributes its render-first stack and fallback approach to Chris
  Strobl's public MIT solution.

This repository reimplements that boundary independently. Its priors are
learned from the public training split, generated out of fold for local
evaluation, and never influence adjudication or confidence. No validation
predictions or case-specific answers from another participant are used.
