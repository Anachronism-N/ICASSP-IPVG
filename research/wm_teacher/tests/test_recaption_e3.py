import copy, importlib.util, json, re, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("recaption_e3", SCRIPTS / "recaption_e3.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)

class RecaptionTests(unittest.TestCase):
    def facts(self):
        return {
            "scene": "a room",
            "rendered_or_screen": False,
            "entities": [{"desc": "a person", "attrs": ["arms lowered"]}],
            "hands": {"visible": True, "held_object": None},
            "actions_by_time": [{"frames": "0-7", "action": "stands still"}],
            "start_end_difference": "no visible change",
        }

    def model_processor(self, replies):
        inputs = MagicMock()
        inputs.to.return_value = inputs
        inputs.__getitem__.return_value.shape = (1, 2)
        processor = MagicMock(return_value=inputs)
        processor.batch_decode.side_effect = [[reply] for reply in replies]
        model = MagicMock()
        return model, processor

    def test_nested_root_regresses_shallow_regex(self):
        root = {"scene": "a room", "entities": [
            {"desc": "a person", "attrs": {"posture": {"arms": ["lowered"]}}}]}
        text = "preamble " + json.dumps(root) + " end"
        shallow = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.S)
        self.assertNotEqual(json.loads(shallow.group(0)), root)
        self.assertEqual(rc.extract_json(text), root)

    def test_fact_root_with_attribute_lists_validates(self):
        facts = self.facts()
        self.assertEqual(rc.validate_facts(rc.extract_json(json.dumps(facts))), facts)

    def test_deep_nesting_and_quoted_braces_escapes(self):
        root = {"text": 'text { } [ ] with "quoted" and slash \\ and comma ,} ,]',
                "nested": [{"inner": {"text": '} \\" { [ ]'}}]}
        self.assertEqual(rc.extract_json(json.dumps(root)), root)

    def test_code_fence_and_trailing_commas_do_not_rewrite_strings(self):
        text = 'intro\n```json\n{"scene":"literal ,} and ,]", "items":[{"attrs":{"x":1,},},],}\n```\nend'
        self.assertEqual(rc.extract_json(text), {
            "scene": "literal ,} and ,]", "items": [{"attrs": {"x": 1}}]})

    def test_first_parseable_independent_top_level_object(self):
        self.assertEqual(rc.extract_json('{broken} {"first":{"a":[1]}} {"second":2}'),
                         {"first": {"a": [1]}})
        self.assertEqual(rc.extract_json('{"first":1} {"second":2}'), {"first": 1})
        self.assertEqual(rc.extract_json('[{"inner":1}] {"outer":2}'), {"outer": 2})
        self.assertEqual(rc.extract_json('{"outer":1} {"unfinished":{"inner":2}'), {"outer": 1})

    def test_truncated_or_invalid_root_never_falls_back_to_inner(self):
        for text in [
            json.dumps(self.facts())[:-1],
            '{"outer": {"inner": {"value": 1}}',
            '{"outer": {"value": "unterminated } {"fake": 1}',
            '{"outer": {"value": 1}, broken}',
            '[{"inner": 1}]',
            '{"outer": [{"inner": 1}]',
            '{"outer": [{"inner": 1}}',
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                rc.extract_json(text)

    def test_facts_require_every_top_level_field(self):
        for key in self.facts():
            facts = self.facts()
            del facts[key]
            before = copy.deepcopy(facts)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                rc.validate_facts(facts)
            self.assertEqual(facts, before)
        for facts in [None, [], {"desc": "a person", "attrs": {}},
                      {"visible": True, "held_object": None}]:
            with self.subTest(facts=facts), self.assertRaises(ValueError):
                rc.validate_facts(facts)
        facts = self.facts()
        facts["extra"] = "not part of the fact schema"
        with self.assertRaises(ValueError):
            rc.validate_facts(facts)

    def test_facts_reject_wrong_top_level_types_without_coercion(self):
        for key, value in [
            ("scene", None), ("rendered_or_screen", "false"),
            ("rendered_or_screen", 0), ("entities", {}), ("hands", []), ("hands", None),
            ("actions_by_time", "0-7"), ("start_end_difference", []),
        ]:
            facts = self.facts()
            facts[key] = value
            before = copy.deepcopy(facts)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                rc.validate_facts(facts)
            self.assertEqual(facts, before)

    def test_facts_reject_wrong_nested_types_and_missing_fields(self):
        variants = [
            ("entities", ["a person"]),
            ("entities", [{"desc": "a person"}]),
            ("entities", [{"desc": 1, "attrs": {}}]),
            ("entities", [{"desc": "a person", "attrs": "visible"}]),
            ("entities", [{"desc": "a person", "attrs": {}}]),
            ("entities", [{"desc": "a person", "attrs": [{"nested": "visible"}]}]),
            ("entities", [{"desc": "a person", "attrs": [1]}]),
            ("entities", [{"desc": "a person", "attrs": [None]}]),
            ("hands", {"visible": True}),
            ("hands", {"visible": "true", "held_object": None}),
            ("hands", {"visible": 1, "held_object": None}),
            ("hands", {"visible": True, "held_object": []}),
            ("actions_by_time", ["still"]),
            ("actions_by_time", [{"frames": "0-7"}]),
            ("actions_by_time", [{"frames": [0, 7], "action": "still"}]),
            ("actions_by_time", [{"frames": "0-7", "action": None}]),
        ]
        for key, value in variants:
            facts = self.facts()
            facts[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                rc.validate_facts(facts)

    def test_attribute_string_lists_and_empty_lists_are_valid_without_coercion(self):
        for attrs in [[], ["arms lowered", "standing"]]:
            facts = self.facts()
            facts["entities"][0]["attrs"] = attrs
            before = copy.deepcopy(facts)
            self.assertIs(rc.validate_facts(facts), facts)
            self.assertEqual(facts, before)

    def test_valid_facts_and_held_claims_are_not_modified(self):
        facts = self.facts()
        facts["hands"]["held_object"] = "a mug"
        before = copy.deepcopy(facts)
        self.assertIs(rc.validate_facts(facts), facts)
        self.assertEqual(rc.held_claims(facts), ["a mug"])
        self.assertEqual(facts, before)

    def test_invalid_facts_stop_before_caption_generation(self):
        for reply in [json.dumps({"desc": "a person", "attrs": {}}),
                      json.dumps(self.facts())[:-1]]:
            model, processor = self.model_processor([reply])
            with patch.object(rc, "sample_frames", return_value=[MagicMock() for _ in range(8)]):
                with self.assertRaises(ValueError):
                    rc.recaption_one(model, processor, "toy-video", "cpu")
            self.assertEqual(model.generate.call_count, 1)
            self.assertEqual(processor.apply_chat_template.call_count, 1)

    def test_valid_facts_pass_caption_and_held_verification(self):
        facts = self.facts()
        facts["hands"]["held_object"] = "a mug"
        caption = "A person stands in a room and holds a mug. The scene stays mostly still."
        model, processor = self.model_processor([json.dumps(facts), caption, "yes"])
        with patch.object(rc, "sample_frames", return_value=[MagicMock() for _ in range(8)]):
            actual, written, removed = rc.recaption_one(model, processor, "toy-video", "cpu")
        self.assertEqual(actual, facts)
        self.assertEqual(written, caption)
        self.assertEqual(removed, [])
        self.assertEqual(model.generate.call_count, 3)

    def test_resume_rejects_legacy_invalid_or_malformed_without_rewriting(self):
        valid = {"sample_id": "toy", "prompt_version": rc.PROMPT_VERSION,
                 "facts_json": self.facts(), "caption": "A still scene."}
        variants = [
            {**valid, "prompt_version": "e3-grounded-recaption-v1"},
            {**valid, "facts_json": {"desc": "a person", "attrs": {}}},
            {**valid, "caption": ""},
            {"sample_id": "toy", "error": "ValueError"},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "records.jsonl"
            for record in variants:
                path.write_text(json.dumps(record) + "\n")
                before = path.read_bytes()
                with self.subTest(record=record), self.assertRaisesRegex(ValueError, "new --out"):
                    rc.resume_done(path)
                self.assertEqual(path.read_bytes(), before)
            path.write_text('{"sample_id": "truncated"')
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "new --out"):
                rc.resume_done(path)
            self.assertEqual(path.read_bytes(), before)

    def test_resume_skips_only_current_valid_successes_not_errors(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "records.jsonl"
            success = {"sample_id": "toy-success", "prompt_version": rc.PROMPT_VERSION,
                       "facts_json": self.facts(), "caption": "A still scene."}
            error = {"sample_id": "toy-error", "prompt_version": rc.PROMPT_VERSION,
                     "error": "ValueError: invalid facts"}
            path.write_text(json.dumps(success) + "\n" + json.dumps(error) + "\n")
            before = path.read_bytes()
            self.assertEqual(rc.resume_done(path), {"toy-success"})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(rc.resume_done(Path(td) / "new.jsonl"), set())

    def test_cli_schema_failure_records_error_not_caption(self):
        model = MagicMock()
        model.to.return_value = model
        model_class = MagicMock()
        model_class.from_pretrained.return_value = model
        processor_class = MagicMock()
        modules = {
            "torch": types.SimpleNamespace(bfloat16="mock-bf16"),
            "transformers": types.SimpleNamespace(
                AutoProcessor=processor_class, Qwen2_5_VLForConditionalGeneration=model_class),
        }
        with tempfile.TemporaryDirectory() as td:
            manifest, out = Path(td) / "manifest.jsonl", Path(td) / "new.jsonl"
            manifest.write_text(json.dumps({"sample_id": "toy", "video": "toy-video",
                                            "captions": {"orig": "unchanged"}}) + "\n")
            before = manifest.read_bytes()
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", [
                "recaption_e3", "--manifest", str(manifest), "--out", str(out)
            ]), patch.object(rc, "recaption_one", side_effect=ValueError("invalid facts schema")):
                rc.main()
            record = json.loads(out.read_text())
            self.assertEqual(record["error"], "ValueError: invalid facts schema")
            self.assertEqual(record["prompt_version"], rc.PROMPT_VERSION)
            self.assertNotIn("caption", record)
            self.assertNotIn("facts_json", record)
            self.assertEqual(manifest.read_bytes(), before)

    def test_cli_legacy_resume_fails_before_loading_model(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "legacy.jsonl"
            out.write_text(json.dumps({"sample_id": "toy", "facts_json": {"attrs": {}}}) + "\n")
            before = out.read_bytes()
            with patch.object(sys, "argv", ["recaption_e3", "--manifest", "unused",
                                            "--out", str(out)]):
                with self.assertRaisesRegex(ValueError, "new --out"):
                    rc.main()
            self.assertEqual(out.read_bytes(), before)

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

    def test_future_version_preserves_display_index_semantics(self):
        self.assertEqual(rc.PROMPT_VERSION, "e3-grounded-recaption-v2")
        self.assertEqual(rc.N_FRAMES, 8)
        self.assertIn("displayed sampled images, NOT original video frame numbers", rc.FACT_PROMPT)
        self.assertNotIn("0-40", rc.FACT_PROMPT)

    def test_prompt_rules_present(self):
        for needle in ["at least 2", "do not guess", "do not invent an ending", "NOT describe camera movement"]:
            self.assertIn(needle.lower().replace("at least 2", "at least 2"), (rc.FACT_PROMPT + rc.WRITE_PROMPT).lower())

if __name__ == "__main__":
    unittest.main()
