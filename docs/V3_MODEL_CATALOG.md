# V3 Shared Model Catalog

*Delivered by issue #45. Consumed by the quota telemetry layer (#46) and the
model broker (#47).*

## 1. What the catalog is

One machine-level file describing every model/resource the developer,
reviewer, and adjudicator lanes may run on. It exists so vendor and model
choices change in **one place** rather than in every CAO profile and role
prompt.

The catalog is a set of **facts and policy metadata**. It does not rank, and
it does not choose. Ask it two things:

| Question | Answer |
| --- | --- |
| What candidates exist, and what is true about them? | `ModelCatalogEntry` |
| Which are usable for this role, at this difficulty, right now? | `ModelCatalog.eligible()` |

Everything requiring live state — quota headroom, provider health, reserves,
cross-lane diversity, the shadow value of perishable subscription capacity —
is deliberately **not** here. That is the broker's job (#47), fed by
telemetry (#46).

## 2. Catalog policy metadata vs. Hermes fallback configuration

These are different artifacts with different owners, and conflating them is
the mistake this document exists to prevent.

| | **Model catalog (this file)** | **Hermes fallback configuration** |
| --- | --- | --- |
| Answers | *What could we use, and what does it cost/suit?* | *This provider just died mid-request — now what?* |
| Timescale | Phase/session boundaries | Within a single request |
| Owner | aipro policy | Hermes runtime |
| Contents | Price, promotion window, capability, role suitability, quality tier, lineage, concurrency cap | An ordered chain of providers to retry |
| Changes when | Pricing/promotions/models change | Operational tuning |

The broker *derives* an ordered fallback chain from catalog metadata and hands
it to Hermes; the catalog itself never stores one. aipro does not micromanage
per-token retries, and Hermes does not make economic decisions.

## 3. Entry fields

`ref` is the policy-level key used everywhere else in V3
(`v3.domain.ModelRef`). `descriptor` is **opaque** — only the broker and
execution adapter interpret it. That opacity is what keeps vendor and model
names out of routing logic entirely.

| Field | Meaning |
| --- | --- |
| `ref`, `descriptor` | Policy key; opaque provider-owned string |
| `provider`, `endpoint` | Hermes provider id; custom/gateway base URL |
| `resource_class` | `subscription` \| `metered` \| `free_tier` |
| `cost_class` | `free` \| `low` \| `medium` \| `high` |
| `input_price_per_mtok`, `output_price_per_mtok` | Cash list price |
| `promotional`, `promo_starts_at`, `promo_ends_at` | Below-list-price window |
| `capabilities` | `tools`, `coding`, `reasoning`, `vision`, `long_context` |
| `roles` | Lane roles this entry may serve; empty means any |
| `min_task_difficulty` | Difficulty floor (1–5): reserve an entry for hard work |
| `quality_by_role` | Manual quality tier (1–5) per role |
| `family`, `vendor` | Lineage, for the broker's adversarial-diversity penalty |
| `max_context_tokens`, `max_concurrency` | Capacity limits |
| `enabled` | Keep an entry documented without dispatching it |
| `data_policy` | Data/training constraint, e.g. `no-training` |
| `notes`, `source_updated_at` | Provenance for volatile pricing/promotion data |

Unknown keys are preserved in `extras`, so a catalog written by a newer
version round-trips losslessly through an older reader.

## 4. The three states of price

The broker must never conflate these, so `effective_prices()` returns:

- `(0.0, 0.0)` — **free**: an active promotion, `cost_class: free`, or
  `resource_class: free_tier`.
- `(in, out)` — **priced**: the declared list price applies.
- `None` — **unknown**: no declared price and no active promotion.

Unknown is not zero. An entry whose price is unknown is *ineligible*, because
reserve and budget policy cannot reason about it.

This is what makes promotion expiry safe. A promotional entry that also
declares a list price simply reverts to that price when the window closes. A
promotional entry with no list price drops out of the eligible set instead of
silently starting to cost money.

For that to hold, "temporarily free" and "permanently free" must stay
distinct. `cost_class: free` and `resource_class: free_tier` both price at zero
*unconditionally*, so an entry claiming one of them alongside `promotional`
would look time-boxed while in fact staying free and eligible forever after its
window closed. That combination is rejected at load time: express a temporary
offer as a promotion over the class the entry reverts to.

Subscription entries report their **list** price. Whether already-bought
allowance should count as marginally free is an economic judgement about
perishable capacity, and belongs to the broker.

## 5. Wiring it up

Point a V3 config at the shared file:

```yaml
model_router:
  catalog_path: /etc/aipro/model-catalog.yml   # or relative to this config
  lane_assignments:
    developer: subscription-primary
```

Relative paths resolve against the directory holding the config file.
Declaring both `catalog_path` and an inline `catalog` is rejected: the
effective catalog would be ambiguous, and a stale inline entry shadowing the
shared file is exactly the failure the split is meant to prevent.

`V3Config.from_dict` stays pure and never reads the disk. `load_v3_config`
performs the I/O and re-checks `lane_assignments` against the resolved refs;
`resolve_model_catalog(config, base_dir=...)` returns the effective catalog.

See `examples/model-catalog.yml` for a worked sample covering a promotional
free model, a gateway endpoint, and two subscription-backed entries with
independent lineage.

## 6. Reload semantics

Entries and catalogs are frozen, and every load returns a fresh object. An
operator may edit the shared file at any time: running phases keep the catalog
they were handed, and only **future** assignments see the change. A catalog
edit can never re-point a session that is already executing.

The freeze is deep: `capabilities` and `roles` are tuples and `quality_by_role`
is a mapping proxy, normalized at construction. A shallow `frozen=True` would
only block rebinding, leaving a caller able to append a role to an entry a
running phase already holds — changing its eligibility mid-flight and skipping
the entry invariants entirely.

## 7. Inspecting eligibility

```console
$ aipro catalog --catalog examples/model-catalog.yml --role worker --difficulty 3
REF                      RESOURCE      COST      IN/MTOK  OUT/MTOK  PROMO
promo-free-generalist    metered       low        0.0000    0.0000  yes
subscription-primary     subscription  high       3.0000   15.0000  -
subscription-secondary   subscription  high       2.5000   10.0000  -
```

(`gateway-cheap-reviewer` is absent because it declares `roles: [reviewer]`,
and `retired-candidate` because it is disabled.)

`--config` resolves the catalog a V3 config points at, and `--json` emits the
same rows machine-readably for diagnostics.

`--all` adds the ineligible entries, and adds an `ELIGIBLE` column so a mixed
table cannot be misread as a list of dispatchable resources:

```console
$ aipro catalog --catalog examples/model-catalog.yml --all
REF                      RESOURCE      COST      IN/MTOK  OUT/MTOK  PROMO  ELIGIBLE
promo-free-generalist    metered       low        0.0000    0.0000  yes    yes
gateway-cheap-reviewer   metered       low        0.3500    1.1000  -      yes
subscription-primary     subscription  high       3.0000   15.0000  -      no
subscription-secondary   subscription  high       2.5000   10.0000  -      no
retired-candidate        metered       medium     1.0000    3.0000  -      no
```

(The subscription entries read `no` here because `--difficulty` defaults to 1
and both declare a floor of 2.)

The whole listing is evaluated at a single timestamp, so a promotion expiring
mid-scan cannot make the filter and the row it produced disagree.
