"""A hardcoded, in-memory fake vector DB - three synthetic multi-section reports,
not report.md.

Each report is split into sections by '## ' headers. Each section carries concrete
IDs/codes as "needles":
  - exact IDs/codes reward BM25's exact-term matching (e.g. asking for an error
    code verbatim)
  - paraphrased conceptual questions with no literal code reward TF-IDF cosine,
    via shared vocabulary with the relevant section's prose - this is bag-of-words
    conceptual overlap, not true dense-embedding semantic search, so don't expect
    it to generalize the way a real embedding model would
"Project Nightingale" is deliberately mentioned in both the cloud-ops and finance
documents, so a query about it should pull top hits from two different source
documents once fused through Retriever's RRF.
"""

from typing import Dict

CLOUD_OPS_DOC = """# Cloud Infrastructure & DevOps Operations Report

This report summarizes infrastructure reliability, migration, delivery pipeline, and
cost work completed by the Cloud Platform team over the current quarter. Four
workstreams are covered below: incident response, the Kubernetes cluster migration,
CI/CD pipeline performance, and cost optimization efforts tied to the upcoming
Project Nightingale launch.

## Incident Response

On March 14th, the primary production database cluster experienced a sustained
connection timeout affecting checkout and account services for approximately 47
minutes. The error surfaced in logs as `ERR_DB_CONN_TIMEOUT_0x4F2A` and was tracked
under incident ticket INC-2026-0091. Root cause analysis traced the failure to a
connection pool exhaustion condition: a recent deploy had lowered the pool's max idle
connections setting, and a traffic spike from a marketing campaign pushed concurrent
requests past the reduced ceiling. The on-call team restored service by rolling back
the connection pool configuration change and manually recycling stuck connections on
the affected database nodes. A permanent fix was shipped the following week, adding
automated alerting on connection pool saturation before it reaches a critical
threshold, and load testing was added to the pre-deploy checklist for any change
touching database configuration. No data loss occurred during the outage, and the
incident postmortem was published internally with the INC-2026-0091 reference for
future audits.

## Kubernetes Cluster Migration

The team completed migration of the `atlas-prod-us-east` cluster from Kubernetes
v1.27 to v1.29 across all 340 production pods with zero unplanned downtime. Migration
was performed in four waves, each moving roughly a quarter of the workload behind a
canary rollout, with automated rollback triggers configured against error-rate and
latency thresholds. Two waves required manual intervention: wave 2 hit a
deprecated API version still referenced by an internal admission webhook, and wave 3
surfaced a resource-limit misconfiguration on a background job queue that had gone
unnoticed under v1.27's more permissive defaults. Both issues were resolved within
the maintenance window. Post-migration, average pod scheduling latency improved by
18%, attributed to v1.29's scheduler improvements. The staging cluster
`atlas-staging-us-east` will be migrated next quarter using the same wave strategy.

## CI/CD Pipeline Performance

Pipeline PIPE-7734, the primary build-and-deploy pipeline for the core services
monorepo, saw its median build time drop from 14.2 minutes to 8.6 minutes this
quarter after introducing dependency-layer caching and splitting the monolithic test
suite into four parallel shards. The change reduced average time-to-deploy for a
typical pull request from open to production by roughly 35%, directly benefiting
release velocity ahead of the Project Nightingale launch. A secondary improvement
came from switching the container build step to a remote build cache shared across
all pipeline runners, cutting redundant compilation work on unchanged dependencies.
Flaky test rate, tracked separately, remains a minor concern at 2.1% of runs
requiring a retry, and is scheduled for a dedicated cleanup pass next quarter.

## Cost Optimization

Cloud spend for the quarter came in at $412,300, a reduction of $68,500 versus the
prior quarter's $480,800, driven primarily by rightsizing over-provisioned compute
instances and reclaiming idle storage volumes. The team moved a significant portion
of batch processing workloads from on-demand `m6i.4xlarge` instances to a mix of
spot and reserved `m6i.2xlarge` instances, cutting compute cost for those workloads
by roughly 40% with negligible impact on batch completion times. Storage costs were
reduced by identifying and archiving 22TB of stale data from decommissioned services
into cold storage. Looking ahead, the team has reserved additional capacity headroom
in anticipation of increased load from Project Nightingale, the product initiative
launching next quarter, and is coordinating with the product organization on
projected traffic estimates to right-size that reservation before launch.
"""

CLINICAL_RESEARCH_DOC = """# Clinical Research & Pharmacology Digest

This digest summarizes progress across four active research programs this quarter:
a rare disease natural history study, an early-phase drug trial, a diagnostic
imaging validation effort, and the status of an ongoing regulatory submission.

## Rare Disease Study

The natural history study of Kessler-Voss syndrome, a rare autoimmune condition
affecting an estimated 1 in 220,000 individuals, continued enrollment this quarter
under cohort ID CV-COH-118, reaching 94 of its target 120 participants. Preliminary
analysis of the first 60 enrolled patients shows a consistent pattern of elevated
inflammatory markers preceding clinical symptom onset by an average of 5.3 months,
a finding that could support earlier diagnosis if replicated in the full cohort.
The study, registered under trial ID CV-TR-004, is being conducted across six
clinical sites, with the two highest-enrolling sites accounting for over half of
total participants. Data collected under this study is informing biomarker
selection for the drug trial described below, since several candidate biomarkers
overlap between the two programs.

## Drug Trial Results

The Phase 1b trial of compound TX-9021, an investigational therapy targeting the
same inflammatory pathway implicated in Kessler-Voss syndrome, met its primary
safety endpoint with no dose-limiting toxicities observed across three dose
cohorts. Exploratory efficacy analysis showed a statistically significant reduction
in the biomarker sIL-6R at the highest tested dose relative to placebo (p=0.003),
though the trial was not powered to detect efficacy and this result should be
treated as hypothesis-generating rather than confirmatory. Twelve of 48 enrolled
participants reported mild injection-site reactions, the most common adverse event,
resolving without intervention in all cases. Based on these results, the sponsor
has elected to advance TX-9021 into Phase 2 planning, with a target enrollment of
180 participants and sIL-6R reduction as a co-primary endpoint alongside a clinical
symptom severity scale.

## Diagnostic Imaging Advances

A validation study of a novel MRI-based diagnostic protocol, designated IMG-PROT-22B,
completed enrollment this quarter with 210 scans evaluated against existing
diagnostic criteria. The protocol, which uses a modified contrast timing sequence to
better differentiate early-stage tissue changes, achieved 91% sensitivity and 88%
specificity against the existing gold-standard diagnostic pathway in this cohort,
an improvement over the 79% sensitivity of the protocol it is intended to replace.
Radiologist inter-rater agreement on protocol IMG-PROT-22B scans was high (Cohen's
kappa of 0.86), suggesting the scoring criteria are reproducible across readers
without extensive additional training. A larger multi-site validation study is
being planned to support eventual clinical adoption.

## Regulatory Submission Timeline

The regulatory submission covering the diagnostic biomarker panel developed under
the rare disease research program, filed under docket FDA-SUB-2026-3391, remains
under active review. The reviewing division issued an initial round of clarifying
questions in the prior quarter, primarily concerning the statistical analysis plan
used for the panel's specificity claims; responses were submitted within the
requested 30-day window. No additional information requests have been received
since, and the team currently estimates a decision within the standard review
timeline, though regulatory timelines can shift based on reviewer workload and
whether additional data requests arise during final review.
"""

FINANCE_STRATEGY_DOC = """# Quarterly Finance & Product Strategy Brief

This brief covers quarterly revenue performance, the status of the upcoming Project
Nightingale product launch, a legal and compliance update, and customer support
metrics for the current quarter.

## Revenue Analysis

Total revenue for Q2 2026 reached $18.4 million, up 12.7% year-over-year and 4.1%
quarter-over-quarter, driven primarily by expansion revenue from existing enterprise
accounts rather than new customer acquisition, which grew more modestly at 3.2%.
Gross margin held steady at 71%, in line with the prior two quarters. The
mid-market segment underperformed its target by roughly 8%, attributed to a longer
than expected sales cycle following a pricing model change introduced at the start
of the quarter; the sales team has since introduced a simplified onboarding path for
that segment to address the slowdown. Enterprise segment revenue exceeded target by
6%, partially offsetting the mid-market shortfall. Leadership expects overall
revenue growth to accelerate next quarter coinciding with the Project Nightingale
launch, discussed below.

## Product Launch Roadmap

Project Nightingale, the company's next major product initiative, remains on track
for a launch in the first month of next quarter. The product introduces a
real-time collaboration layer on top of the existing platform, a feature
consistently requested by enterprise customers in the last two annual satisfaction
surveys. Engineering has completed feature-complete development and is currently in
a hardening phase focused on load testing and edge-case bug fixes; the Cloud
Platform team has separately reserved additional infrastructure capacity in
anticipation of the traffic increase this launch is expected to bring. Marketing has
finalized launch messaging and a phased rollout plan, beginning with a subset of
existing enterprise customers before general availability two weeks later. Early
access feedback from six pilot customers has been positive, with the most common
requested change being finer-grained permission controls, which has been added to
the post-launch roadmap rather than blocking the initial release.

## Legal & Compliance Update

The company received no new material litigation this quarter. An existing
commercial dispute, docketed under case number LC-2026-0147, concerning a
terminated reseller agreement, remains in early-stage mediation, with both parties
having agreed to a confidential settlement conference scheduled for next quarter;
outside counsel does not currently expect the matter to proceed to litigation. On
the compliance side, the annual SOC 2 Type II audit was completed with no material
findings, and a routine review of data processing agreements with three vendors was
completed ahead of schedule with no required contract amendments. The legal team
has begun preliminary review of contractual and compliance obligations related to
the upcoming Project Nightingale launch, particularly around the new collaboration
feature's handling of customer data shared between users.

## Customer Support Metrics

Support ticket volume for the quarter totaled 14,320 tickets, roughly flat versus
the prior quarter's 14,050, despite continued customer growth, which the support
team attributes to a knowledge-base expansion completed two quarters ago that
continues to deflect a meaningful share of routine questions. Median first-response
time improved to 2.4 hours from 3.1 hours, and customer satisfaction (CSAT) on
resolved tickets held at 94%, consistent with the prior three quarters. The support
team has begun preparing dedicated documentation and a staffing plan for the
anticipated increase in ticket volume following the Project Nightingale launch,
based on support volume patterns observed after past major feature releases.
"""

DOCUMENTS: Dict[str, str] = {
    "Cloud Infrastructure & DevOps Operations Report": CLOUD_OPS_DOC,
    "Clinical Research & Pharmacology Digest": CLINICAL_RESEARCH_DOC,
    "Quarterly Finance & Product Strategy Brief": FINANCE_STRATEGY_DOC,
}
