import unittest

from agents.falsifier_agent import _normalize_queries as normalize_falsifier_queries
from agents.verifier_agent import _normalize_queries as normalize_verifier_queries


class AgentQueryTests(unittest.TestCase):
    def test_verifier_query_normalization_falls_back_for_non_lists(self):
        queries = normalize_verifier_queries("not-a-list", fallback=["claim text"])
        self.assertEqual(queries, ["claim text"])

    def test_falsifier_query_normalization_deduplicates_and_trims(self):
        queries = normalize_falsifier_queries(
            [" claim text ", "Claim   text", "", "fact check"],
            fallback=["fallback"],
        )

        self.assertEqual(queries, ["claim text", "fact check"])


if __name__ == "__main__":
    unittest.main()
