import unittest
from unittest.mock import patch

from agents.claim_extractor import _sanitize_claims, extract_claims


class ClaimExtractorTests(unittest.TestCase):
    def test_sanitize_claims_filters_invalid_entries(self):
        claims = _sanitize_claims(
            [
                {"claim": "Short", "importance": "high"},
                {"claim": "The city reported 42 new cases on Tuesday.", "importance": "urgent", "entities": "bad"},
                {"claim": "The city reported 42 new cases on Tuesday.", "importance": "low"},
                {"claim": "The mayor said the bridge project will cost $50 million.", "importance": "high", "entities": ["Mayor", "Bridge"]},
            ]
        )

        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["importance"], "medium")
        self.assertEqual(claims[0]["entities"], [])
        self.assertEqual(claims[1]["importance"], "high")

    @patch("agents.claim_extractor.extract_entities", return_value={})
    @patch("agents.claim_extractor.groq_chat_json", return_value={})
    def test_extract_claims_uses_fallback_when_llm_returns_no_valid_claims(self, _mock_llm, _mock_entities):
        article = (
            "On Tuesday the city reported 42 new cases across three districts. "
            "Officials said hospital occupancy dropped to 61 percent by evening."
        )

        claims = extract_claims(article)

        self.assertGreaterEqual(len(claims), 1)
        self.assertIn("42", claims[0]["claim"])


if __name__ == "__main__":
    unittest.main()
