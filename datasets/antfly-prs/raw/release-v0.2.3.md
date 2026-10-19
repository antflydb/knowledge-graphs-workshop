Release v0.2.3

# Antfly version v0.2.3

### Documentation (2026-09-22)

The documentation is available here:

https://docs.antfly.io

## What's Changed
* fix(metadata): skip already-applied retained catalog entries (v0.2.x) by @ajroetker in https://github.com/antflydb/antfly/pull/744
* fix(runtime): keep dedicated metadata separate from data ownership (v0.2.x) by @ajroetker in https://github.com/antflydb/antfly/pull/745
* fix(backup): require concurrency for lease heartbeats (v0.2.x) by @ajroetker in https://github.com/antflydb/antfly/pull/747
* fix(inference): reuse Bedrock credentials for dimension probes (v0.2.x) by @ajroetker in https://github.com/antflydb/antfly/pull/748
* fix(api): release unpublished state after failed server initialization (v0.2.x) by @ajroetker in https://github.com/antflydb/antfly/pull/750
* fix(antfarm): omit unset graph edge weight (v0.2.x) by @ajroetker in https://github.com/antflydb/antfly/pull/754
* fix(metadata): enforce read budgets through snapshot publication (v0.2.x) by @ajroetker in https://github.com/antflydb/antfly/pull/749
* fix(antfarm): refresh embedded edge weight fix (v0.2.x) by @ajroetker in https://github.com/antflydb/antfly/pull/756
* fix(secrets): backport wall-clock timestamps to v0.2.x by @ajroetker in https://github.com/antflydb/antfly/pull/762
* fix(httpx): harden transport errors and repair release test fixtures (v0.2.x) by @ajroetker in https://github.com/antflydb/antfly/pull/746
* fix(generating): backport OpenAI completion options to v0.2.x by @ajroetker in https://github.com/antflydb/antfly/pull/787
* Backport empty hot standby bootstrap and table creation to v0.2.x by @bpopadiuk in https://github.com/antflydb/antfly/pull/765
* [v0.2.x] Fix exact aggregations during concurrent index publication by @ajroetker in https://github.com/antflydb/antfly/pull/794
* Backport generator tool support to v0.2.x by @ajroetker in https://github.com/antflydb/antfly/pull/796
* Backport secret-store symlink rotation fix to v0.2.x by @ajroetker in https://github.com/antflydb/antfly/pull/799
* Fix artifact enrichment metadata responses on v0.2.x by @bpopadiuk in https://github.com/antflydb/antfly/pull/835
* fix(v0.2.x): prevent shutdown worker races and DB lease leaks by @ajroetker in https://github.com/antflydb/antfly/pull/853
* backport reranker query fixes for v0.2.x by @timkaye11 in https://github.com/antflydb/antfly/pull/848


**Full Changelog**: https://github.com/antflydb/antfly/compare/v0.2.2...v0.2.3

---

Released by Antfly release scripts.
