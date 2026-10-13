import json
import os
import unittest
from unittest.mock import patch

from gtm_agent import gtm_agent


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.previous_tracing = os.environ.get("LANGSMITH_TRACING")
        os.environ["LANGSMITH_TRACING"] = "false"
        self.offering = {
            "required_tech_stack": ["Snowflake"],
            "min_annual_revenue": 1000000,
            "description": "A test offering",
        }
        self.score = gtm_agent.ProspectScore(
            score=85,
            justification="Strong fit",
            rubric_breakdown=gtm_agent.RubricBreakdown(
                revenue_fit=30,
                tech_stack_match=30,
                segment_fit=25,
            ),
        )

    def tearDown(self):
        if self.previous_tracing is None:
            os.environ.pop("LANGSMITH_TRACING", None)
        else:
            os.environ["LANGSMITH_TRACING"] = self.previous_tracing

    def test_loads_records_from_ids_and_returns_structured_score(self):
        profile = {"prospect_id": "LEAD-71001", "tech_stack": ["Old stack"]}
        refreshed_stack = ["Snowflake", "Databricks"]
        with (
            patch.object(
                gtm_agent.data_service, "get_offering", return_value=self.offering
            ) as get_offering,
            patch.object(
                gtm_agent.data_service,
                "get_profile_from_db",
                return_value={"prospect_profile": profile},
            ),
            patch.object(
                gtm_agent.data_service,
                "fetch_tech_stack",
                return_value=refreshed_stack,
            ),
            patch.object(gtm_agent, "_scoring_llm") as scoring_llm,
        ):
            scoring_llm.invoke.return_value = self.score
            result = gtm_agent.score_prospect.invoke(
                {"prospect_id": "LEAD-71001", "offering_id": "OFFER-10005"}
            )

        self.assertEqual(result, self.score.model_dump())
        get_offering.assert_called_once_with("OFFER-10005")
        submitted_profile = json.loads(
            scoring_llm.invoke.call_args.args[0][1]["content"].split(
                "\n\nProspect profile:\n", 1
            )[1]
        )
        self.assertEqual(submitted_profile["tech_stack"], refreshed_stack)
        self.assertEqual(profile["tech_stack"], ["Old stack"])

    def test_returns_error_for_missing_or_invalid_offering(self):
        for offering in (None, {"required_tech_stack": []}):
            with self.subTest(offering=offering):
                with (
                    patch.object(
                        gtm_agent.data_service,
                        "get_offering",
                        return_value=offering,
                    ),
                    patch.object(gtm_agent.data_service, "get_profile_from_db") as get_profile,
                    patch.object(gtm_agent, "_scoring_llm") as scoring_llm,
                ):
                    result = gtm_agent.score_prospect.invoke(
                        {"prospect_id": "LEAD-71001", "offering_id": "OFFER-10005"}
                    )

                self.assertEqual(
                    result,
                    {"score": None, "error": "Cannot score without a valid offering."},
                )
                get_profile.assert_not_called()
                scoring_llm.invoke.assert_not_called()

    def test_builds_and_scores_fallback_profile(self):
        record = {"name": "Test Prospect", "annual_revenue": 5000000}
        history = [{"channel": "demo"}]
        account = {"industry": "technology"}
        tech_stack = ["Snowflake"]
        with (
            patch.object(gtm_agent.data_service, "get_offering", return_value=self.offering),
            patch.object(
                gtm_agent.data_service,
                "get_profile_from_db",
                return_value={"prospect_profile": None},
            ),
            patch.object(
                gtm_agent.data_service, "get_prospect_record", return_value=record
            ),
            patch.object(
                gtm_agent.data_service,
                "fetch_engagement_history",
                return_value=history,
            ),
            patch.object(
                gtm_agent.data_service,
                "fetch_account_details",
                return_value=account,
            ),
            patch.object(
                gtm_agent.data_service,
                "fetch_tech_stack",
                return_value=tech_stack,
            ),
            patch.object(gtm_agent.data_service, "save_profile_to_db") as save_profile,
            patch.object(gtm_agent, "_scoring_llm") as scoring_llm,
        ):
            scoring_llm.invoke.return_value = self.score
            result = gtm_agent.score_prospect.invoke(
                {"prospect_id": "LEAD-71001", "offering_id": "OFFER-10005"}
            )

        self.assertEqual(result, self.score.model_dump())
        saved_profile = save_profile.call_args.args[1]
        self.assertEqual(saved_profile["prospect_id"], "LEAD-71001")
        self.assertEqual(saved_profile["engagement_history"], history)
        self.assertEqual(saved_profile["account_details"], account)
        self.assertEqual(saved_profile["tech_stack"], tech_stack)
        submitted_profile = json.loads(
            scoring_llm.invoke.call_args.args[0][1]["content"].split(
                "\n\nProspect profile:\n", 1
            )[1]
        )
        self.assertEqual(submitted_profile, saved_profile)

    def test_does_not_score_missing_prospect(self):
        with (
            patch.object(
                gtm_agent.data_service,
                "get_offering",
                return_value=self.offering,
            ),
            patch.object(
                gtm_agent.data_service,
                "get_profile_from_db",
                return_value={"prospect_profile": None},
            ),
            patch.object(
                gtm_agent.data_service, "get_prospect_record", return_value=None
            ),
            patch.object(gtm_agent, "_scoring_llm") as scoring_llm,
        ):
            result = gtm_agent.score_prospect.invoke(
                {"prospect_id": "MISSING", "offering_id": "OFFER-10005"}
            )

        self.assertEqual(
            result,
            {"score": None, "error": "Cannot score without a valid prospect."},
        )
        scoring_llm.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
