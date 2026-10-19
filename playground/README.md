# Text to graph

Paste or type sentences, one per line, and watch a knowledge graph assemble
beside them. Each line is a document; Antfly Inference reads it with
GLiNER2.5 (`fastino/gliner2.5-base-v1`) and returns entities and relations
for the entity and relation types you list. Nothing is stored; the text and
schema live in the URL, so a copied link reproduces the graph.

## Run it

The workshop web app serves a prebuilt copy of this page as the **Text to graph**
tab (`make web`, then http://127.0.0.1:8000/#text-to-graph) and forwards its `/ai`
calls to Antfly inference (`make inference`). Nothing here needs Node unless you
change it.

To change it, edit `src/` and rebuild the committed bundle in
`kg/web/static/playground/` with `make playground` (needs pnpm). For live editing,
`pnpm dev` serves it on http://localhost:5317 and proxies `/ai` to
`127.0.0.1:8090`; `ANTFLY_URL` points the proxy elsewhere and `VITE_ANTFLY_MODEL`
swaps the extractor.

Copied from Antfly's `examples/text-to-graph` (by Drew Lanenga), Apache-2.0; see
[LICENSE](LICENSE).

## How it works

- **One request per line.** `POST /ai/v1/extract` with the line as the only
  input. Results are cached by (schema, line text), so editing one line
  re-reads only that line. Requests run one at a time (about 30–40 ms per
  line on Metal) because the model admits one request at a time; a
  `MODEL_RESOURCE_BUSY` reply is retried.
- **Merging across lines** happens in the browser
  ([`src/graph.ts`](src/graph.ts)): mentions with the same normalized text
  become one node, and a capitalized single name ("Priya") folds into the
  one longer name of the same type that starts or ends with it
  ("Priya Raman"). Spans nested inside a longer span are dropped.
- **Relation selection.** GLiNER2.5 scores every relation type for every
  pair, so a pair often gets a correct type and a weaker wrong one
  ("acted_in" 0.86, "directed" 0.69). The graph keeps each pair's best type,
  plus near-ties within 0.03, above a 0.78 floor. These numbers were tuned
  on the bundled examples.
- **Hovering** a node highlights the lines it came from; hovering a line
  highlights the nodes and edges it produced. New nodes enter from the left
  edge at the height of the line that introduced them.

## Limits

- Each line is read on its own, so pronouns don't resolve: "He likes pears"
  won't link to Johnny.
- Relation types are plain names. The API also accepts typed relations
  (`{"type": "targets", "source": "drug", "target": "protein"}`), which
  would help domains like biomedicine where the base model confuses types.
