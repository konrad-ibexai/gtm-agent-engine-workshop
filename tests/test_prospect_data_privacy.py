import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent import data_service, gtm_agent


class ProspectDataPrivacyTests(unittest.TestCase):
    def test_get_prospect_returns_only_contact_fields(self):
        record = {
            "name": "Avery Prospect",
            "email": "avery@example.com",
            "billing_qualification": {"tax_id": "sensitive"},
            "engagement_history": ["enrichment"],
            "account_details": ["enrichment"],
            "tech_stack": ["enrichment"],
            "annual_revenue": 1000000,
        }
        with patch.object(data_service, "get_prospect_record", return_value=record):
            result = gtm_agent.get_prospect.func("LEAD-TEST")

        self.assertEqual(result["prospect"], {
            "prospect_id": "LEAD-TEST",
            "name": "Avery Prospect",
            "email": "avery@example.com",
        })

    def test_build_profile_filters_billing_from_result_and_cache(self):
        prospect_id = "LEAD-PRIVACY-TEST"
        record = {
            "name": "Avery Prospect",
            "email": "avery@example.com",
            "billing_qualification": {"tax_id": "sensitive"},
            "annual_revenue": 1000000,
        }
        data_service._PROFILES.pop(prospect_id, None)
        try:
            with (
                patch.object(data_service, "get_prospect_record", return_value=record),
                patch.object(data_service, "fetch_engagement_history", return_value=[]),
                patch.object(data_service, "fetch_account_details", return_value=[]),
                patch.object(data_service, "fetch_tech_stack", return_value=["CRM"]),
            ):
                result = gtm_agent.build_prospect_profile.func(prospect_id)

            profile = result["prospect_profile"]
            self.assertNotIn("billing_qualification", profile)
            self.assertEqual(profile["annual_revenue"], 1000000)
            self.assertEqual(data_service._PROFILES[prospect_id], profile)
            self.assertNotIn("billing_qualification", data_service._PROFILES[prospect_id])
        finally:
            data_service._PROFILES.pop(prospect_id, None)

    def test_build_profile_removes_billing_from_existing_cache(self):
        prospect_id = "LEAD-STALE-CACHE-TEST"
        data_service._PROFILES[prospect_id] = {
            "prospect_id": prospect_id,
            "annual_revenue": 1000000,
            "billing_qualification": {"tax_id": "sensitive"},
        }
        try:
            cached = data_service.get_profile_from_db(prospect_id)["prospect_profile"]
            self.assertNotIn("billing_qualification", cached)
            result = gtm_agent.build_prospect_profile.func(prospect_id)
            self.assertNotIn("billing_qualification", result["prospect_profile"])
            self.assertNotIn("billing_qualification", data_service._PROFILES[prospect_id])
        finally:
            data_service._PROFILES.pop(prospect_id, None)

    def test_score_prospect_removes_billing_from_scoring_prompt(self):
        profile = {
            "prospect_id": "LEAD-TEST",
            "annual_revenue": 1000000,
            "billing_qualification": {"tax_id": "sensitive"},
            "tech_stack": ["CRM"],
        }
        offering = {
            "required_tech_stack": ["CRM"],
            "min_annual_revenue": 500000,
            "description": "Test offering",
        }
        with (
            patch.object(data_service, "fetch_tech_stack", return_value=["CRM"]),
            patch.object(gtm_agent, "_scoring_llm") as scoring_llm,
        ):
            scoring_llm.invoke.return_value = SimpleNamespace(
                model_dump=lambda: {"score": 90}
            )
            result = gtm_agent.score_prospect.func(profile, offering)

        self.assertEqual(result, {"score": 90})
        prompt = scoring_llm.invoke.call_args.args[0][1]["content"]
        self.assertNotIn("billing_qualification", prompt)
        self.assertNotIn("sensitive", prompt)
        self.assertIn('"annual_revenue": 1000000', prompt)
        self.assertIn('"tech_stack": [\n    "CRM"\n  ]', prompt)

    def test_send_prospect_email_keeps_supported_behavior(self):
        prospect = {"name": "Avery Prospect", "email": "avery@example.com"}
        rep = {"name": "Riley Rep", "email": "riley@example.com"}
        result = gtm_agent.send_prospect_email.func(
            prospect, "Hello", "A short note", runtime=None, from_rep=rep
        )

        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual(result["to"], prospect["email"])
        self.assertEqual(result["to_name"], prospect["name"])
        self.assertEqual(result["from"], rep["email"])
        self.assertEqual(result["from_name"], rep["name"])
        self.assertEqual(result["subject"], "Hello")
        self.assertEqual(result["body"], "A short note")


if __name__ == "__main__":
    unittest.main()
