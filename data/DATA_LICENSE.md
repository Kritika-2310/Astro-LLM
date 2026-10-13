# Data sources and licenses (v2 audience-style dataset)

- **OpenStax Astronomy 2e**: CC BY 4.0. Attribution: OpenStax, Astronomy 2e (openstax.org).
- **Wikipedia**: CC BY-SA 4.0. Derived data built from Wikipedia text must be shared under the same license.
- Q&A answers in `data/processed_v2/` are LLM-generated (Gemini) from the passages above and filtered; they inherit the share-alike obligation of the Wikipedia-derived rows (`source == "wikipedia"`).
- The OpenStax PDF itself is not committed (see `.gitignore`); download it from openstax.org.
