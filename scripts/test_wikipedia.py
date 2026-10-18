"""Consumer-visible evidence and provenance boundaries for Wikipedia collection."""
import json
from pathlib import Path
import tempfile
import unittest

from kg.wikipedia import Fetcher, parse_article, sha


HTML = '''<html><head><link rel="canonical" href="https://en.wikipedia.org/wiki/Example"/>
<script>var config={"wgRevisionId":42,"wgArticleId":7};</script></head><body>
<h1 id="firstHeading">Example</h1><div class="mw-parser-output">
<div class="hatnote">For another use, see <a href="/wiki/Unrelated">Unrelated</a>.</div>
<div class="hatnote">Main article: <a href="/wiki/Related">Related</a></div>
<table class="infobox"><tr><td>Father</td><td>Not evidence</td></tr></table>
<h2>House One</h2><h3>Example</h3><p>She was <a href="/wiki/Elsewhere">reportedly</a> an heir.</p>
<blockquote>Excluded external quotation.</blockquote>
<h2>Reception</h2><p>Excluded review.</p>
<h2>Adaptation</h2><p>In the television adaptation, she survives.</p>
</div><li id="footer-info-copyright">Text available under
<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</li></body></html>'''


class WikipediaBoundaries(unittest.TestCase):
    def test_qualification_scope_and_offsets_survive_without_structured_facts(self):
        row, main, _ = parse_article(HTML, {"reception"})
        self.assertEqual(row["text"], "She was reportedly an heir.\n\nIn the television adaptation, she survives.")
        self.assertEqual(main, {"Related"})
        self.assertEqual(row["revision_id"], 42)
        self.assertEqual(row["sections"][0]["headings"], ["House One", "Example"])
        self.assertNotIn("entity_ids", row)
        self.assertNotIn("_edges", row)
        for section in row["sections"]:
            actual = row["text"][section["start_char"]:section["end_char"]]
            self.assertIn(actual, ["She was reportedly an heir.", "In the television adaptation, she survives."])

    def test_heading_label_does_not_follow_generic_prose_links(self):
        html = HTML.replace("<h2>House One</h2>", '<h2>House One</h2><p><a href="/wiki/House_One">House One</a> is a house.</p><h2>Main cast</h2><p>An <a href="/wiki/Ensemble_cast">Main cast</a> description.</p>')
        _, _, links = parse_article(html, {"reception"})
        self.assertIn("House One", links)
        self.assertNotIn("Ensemble cast", links)

    def test_page_indicator_cannot_shadow_article_prose(self):
        html = HTML.replace('<div class="mw-parser-output">', '<div id="mw-content-text"><div class="mw-parser-output">', 1).replace('</div><li id="footer', '</div></div><li id="footer', 1)
        html = html.replace('<h1 id="firstHeading">', '<div id="mw-indicator-good-star"><div class="mw-parser-output">★</div></div><h1 id="firstHeading">')
        row, _, _ = parse_article(html, {"reception"})
        self.assertTrue(row["text"].startswith("She was reportedly an heir."))

    def test_unverified_license_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "license"):
            parse_article(HTML.replace("creativecommons.org/licenses/by-sa/4.0/", "example.org/terms"), set())

    def test_nested_list_paragraph_is_not_duplicated(self):
        html = HTML.replace("<p>She was", "<ul><li><p>She was", 1).replace("an heir.</p>", "an heir.</p></li></ul>", 1)
        row, _, _ = parse_article(html, {"reception"})
        self.assertEqual(row["text"].count("She was reportedly an heir."), 1)

    def test_offline_cache_cannot_cross_robots_boundary_or_hide_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def cached(url, text):
                key = sha(url.encode())
                (root / (key + ".html")).write_text(text)
                (root / (key + ".json")).write_text(json.dumps({"sha256": sha(text.encode())}))
            cached("https://en.wikipedia.org/robots.txt", "User-agent: *\nDisallow: /w/\n")
            cached("https://en.wikipedia.org/w/api.php", "blocked cached payload")
            cached("https://en.wikipedia.org/wiki/Example", "original payload")
            cached("https://example.org/wiki/Example", "off-site cached payload")
            fetcher = Fetcher(root, "Workshop/1 (https://example.org/contact)", 2, True)
            try:
                with self.assertRaisesRegex(ValueError, "robots"):
                    fetcher.fetch("https://en.wikipedia.org/w/api.php")
                with self.assertRaises(ValueError):
                    fetcher.fetch("https://example.org/wiki/Example")
                (root / (sha(b"https://en.wikipedia.org/wiki/Example") + ".html")).write_text("tampered")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    fetcher.fetch("https://en.wikipedia.org/wiki/Example")
            finally:
                fetcher.client.close()


if __name__ == "__main__":
    unittest.main()
