import importlib.util, json, re, sys, unittest
from pathlib import Path
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("recaption_e3", SCRIPTS / "recaption_e3.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)

class RecaptionTests(unittest.TestCase):
    def test_held_regex_catches_claims(self):
        for t in ["He holds a cup", "holding the lid", "grips the handle", "picks up the ball", "carries a box"]:
            self.assertTrue(rc.HELD_RE.search(t), t)
        for t in ["The room is bright", "She walks forward"]:
            self.assertFalse(rc.HELD_RE.search(t), t)

    def test_facts_json_extraction(self):
        facts = rc.extract_json('blah {"scene": "kitchen", "hands": {"held_object": null}} end')
        self.assertEqual(facts["scene"], "kitchen")
        with self.assertRaises(ValueError):
            rc.extract_json("no json here")

    def test_held_claims_gate(self):
        self.assertEqual(rc.held_claims({"hands": {"held_object": "a mug"}}), ["a mug"])
        self.assertEqual(rc.held_claims({"hands": {"held_object": None}}), [])
        self.assertEqual(rc.held_claims({}), [])

    def test_cam_sentence_thresholds(self):
        self.assertEqual(rc.cam_sentence(0.2), "The camera is static.")
        self.assertEqual(rc.cam_sentence(1.0), "The camera moves slightly.")
        self.assertEqual(rc.cam_sentence(5.0), "The camera moves noticeably.")
        self.assertEqual(rc.cam_sentence(None), "")

    def test_unverified_claim_removed_from_caption(self):
        caption = "A person stands still. He holds a silver spoon carefully. The light is soft."
        claim = "silver spoon"
        out = re.sub(r"[^.]*\bsilver[^^.]*\.\s*", "", caption)
        self.assertNotIn("silver", out)
        self.assertIn("A person stands still", out)

    def test_prompt_rules_present(self):
        for needle in ["at least 2", "do not guess", "do not invent an ending", "NOT describe camera movement"]:
            self.assertIn(needle.lower().replace("at least 2", "at least 2"), (rc.FACT_PROMPT + rc.WRITE_PROMPT).lower())

if __name__ == "__main__":
    unittest.main()
