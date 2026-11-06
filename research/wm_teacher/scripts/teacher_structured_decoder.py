"""Prospective JSON-syntax and existing vocabulary constraints; no output repair or semantic approval."""

FACTS_SCHEMA = {
    'type': 'object',
    'properties': {
        'scene': {'type': 'string'},
        'rendered_or_screen': {'type': 'boolean'},
        'entities': {'type': 'array', 'minItems': 1, 'maxItems': 32, 'items': {
            'type': 'object', 'properties': {
                'id': {'type': 'string', 'pattern': 'E[1-9][0-9]*'},
                'description': {'type': 'string'},
                'held_object': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
            }, 'required': ['id', 'description', 'held_object'], 'additionalProperties': False,
        }},
        'events': {'type': 'array', 'maxItems': 128, 'items': {
            'type': 'object', 'properties': {
                'entity_id': {'type': 'string', 'pattern': 'E[1-9][0-9]*'},
                'frame_start': {'type': 'integer', 'enum': list(range(16)), 'minimum': 0, 'maximum': 15},
                'frame_end': {'type': 'integer', 'enum': list(range(16)), 'minimum': 0, 'maximum': 15},
                'action': {'type': 'string'},
            }, 'required': ['entity_id', 'frame_start', 'frame_end', 'action'], 'additionalProperties': False,
        }},
        'start_end_difference': {'type': 'string'},
    },
    'required': ['scene', 'rendered_or_screen', 'entities', 'events', 'start_end_difference'],
    'additionalProperties': False,
}
CAPTION_SCHEMA = {'type': 'array', 'items': {'type': 'string'}, 'minItems': 4, 'maxItems': 4}


def caption_word_schema():
    return {
        'type': 'array', 'minItems': 4, 'maxItems': 4,
        'items': {
            'type': 'string',
            'pattern': r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*(?:[ ,;:]+[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*){11,21}[.!?]",
        },
    }


def bounded_facts_schema():
    from copy import deepcopy
    schema = deepcopy(FACTS_SCHEMA)
    props = schema['properties']
    props['scene']['maxLength'] = 160
    props['start_end_difference']['maxLength'] = 200
    entity = props['entities']['items']['properties']
    entity['id'] = {'type': 'string', 'enum': ['E' + str(i) for i in range(1, 33)]}
    entity['description']['maxLength'] = 160
    entity['held_object']['anyOf'][0]['maxLength'] = 80
    event = props['events']['items']['properties']
    event['entity_id'] = deepcopy(entity['id'])
    event['action']['maxLength'] = 120
    return schema


class StructuredDecoder:
    def __init__(self, tokenizer, eos_ids, *, bounded=False, word_count=False):
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        self.tokenizer = tokenizer
        self.bounded = bounded
        self.word_count = word_count
        self.eos_ids = [eos_ids] if type(eos_ids) is int else eos_ids
        if type(self.eos_ids) is not list or not self.eos_ids or tokenizer.eos_token_id not in self.eos_ids:
            raise ValueError('Frozen tokenizer EOS must be an actual configured generation EOS')
        self.tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
        self.boundary_tree = {}
        for token, text, _ in self.tokenizer_data.regular_tokens:
            if '"' not in text[:-1] or text.startswith('"') or any(ord(char) < 32 for char in text):
                continue
            node = self.boundary_tree
            for char in text:
                node = node.setdefault(char, {})
            node.setdefault(None, []).append(token)

    def boundary_prefix(self, prefix):
        def allowed(batch_id, sent):
            tokens = prefix(batch_id, sent)
            parser = prefix.token_enforcer.prefix_states[tuple(sent.tolist())].parser
            key = parser.shortcut_key()
            if not isinstance(key, tuple) or key[0] != 'json_freetext':
                return tokens
            # LMFE's freetext shortcut omits tokens like '.\",' that cross a closing quote.
            extra = []
            stack = [(parser, self.boundary_tree)]
            while stack:
                state, node = stack.pop()
                extra.extend(node.get(None, ()))
                chars = state.get_allowed_characters()
                for char, child in node.items():
                    if char is not None and char in chars:
                        stack.append((state.add_character(char), child))
            return sorted(set(tokens).union(extra))
        allowed.token_enforcer = prefix.token_enforcer
        return allowed

    def constraints(self, facts_stage):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        from transformers import LogitsProcessorList
        import run_teacher_qwen3vl32b_recovery as original
        words = original.DENY_DECODE_WORDS + (() if facts_stage else original.CAPTION_DECODE_WORDS)
        forms = {prefix + form for word in words for form in (word, word.title(), word.upper()) for prefix in ('', ' ')}
        ids = [self.tokenizer.encode(form, add_special_tokens=False) for form in sorted(forms)]
        schema = bounded_facts_schema() if self.bounded and facts_stage else FACTS_SCHEMA if facts_stage else CAPTION_SCHEMA
        if self.word_count and not facts_stage:
            schema = caption_word_schema()
        grammar = JsonSchemaParser(schema)
        return {
            'prefix_allowed_tokens_fn': self.boundary_prefix(build_transformers_prefix_allowed_tokens_fn(self.tokenizer_data, grammar)),
            'logits_processor': LogitsProcessorList([original.vectorized_bad_words(ids, self.eos_ids)]),
        }

    def cuda_admission(self, device):
        import torch
        from transformers import PrefixConstrainedLogitsProcessor
        import json
        facts = dict(scene='A red disk.', rendered_or_screen=False,
            entities=[dict(id='E1', description='A red disk', held_object=None)],
            events=[dict(entity_id='E1', frame_start=0, frame_end=15, action='Stays in place.')],
            start_end_difference='No visible change.')
        prompt = self.tokenizer.encode('Synthetic JSON:', add_special_tokens=False)
        caption = ['A red circular disk rests on a plain white table without any visible movement.'] * 4 if self.word_count else ['one.', 'two.', 'three.', 'four.']
        paths = [(True, json.dumps(facts)), (False, json.dumps(caption))]
        lengths = []
        for facts_stage, text in paths:
            constraints = self.constraints(facts_stage)
            mask = PrefixConstrainedLogitsProcessor(constraints['prefix_allowed_tokens_fn'], 1)
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            self.check_token_path(prompt, tokens, constraints, mask, device)
            lengths.append(len(tokens))
        if self.bounded:
            constraints = self.constraints(True)
            fn = constraints['prefix_allowed_tokens_fn']
            mask = PrefixConstrainedLogitsProcessor(fn, 1)
            sent = list(prompt)
            fn(0, torch.tensor(sent, dtype=torch.long))
            repeated = self.tokenizer.encode('a', add_special_tokens=False)
            assert len(repeated) == 1
            path = self.tokenizer.encode('{"scene":"', add_special_tokens=False) + repeated * 160
            for token in path:
                assert token in fn(0, torch.tensor(sent, dtype=torch.long))
                sent.append(token)
            ids = torch.tensor([sent], dtype=torch.long, device='cpu')
            scores = torch.zeros((1, len(self.tokenizer)), device=device, dtype=torch.bfloat16)
            values = constraints['logits_processor'](ids, mask(ids, scores))
            quote = self.tokenizer.encode('"', add_special_tokens=False)
            assert len(quote) == 1 and bool(torch.isfinite(values[0, quote[0]]))
            assert bool(torch.isneginf(values[0, repeated[0]]))
            assert all(bool(torch.isneginf(values[0, stop])) for stop in self.eos_ids)
        if self.word_count:
            period = self.tokenizer.encode('.', add_special_tokens=False)
            extra = self.tokenizer.encode(' visible', add_special_tokens=False)
            assert len(period) == len(extra) == 1
            for words in (11, 12, 22):
                constraints = self.constraints(False)
                fn = constraints['prefix_allowed_tokens_fn']
                mask = PrefixConstrainedLogitsProcessor(fn, 1)
                sent = list(prompt)
                fn(0, torch.tensor(sent, dtype=torch.long))
                text = '["' + ' '.join(['visible'] * words)
                for token in self.tokenizer.encode(text, add_special_tokens=False):
                    assert token in fn(0, torch.tensor(sent, dtype=torch.long))
                    sent.append(token)
                ids = torch.tensor([sent], dtype=torch.long, device='cpu')
                scores = torch.zeros((1, len(self.tokenizer)), device=device, dtype=torch.bfloat16)
                values = constraints['logits_processor'](ids, mask(ids, scores))
                assert bool(torch.isfinite(values[0, period[0]])) == (words >= 12)
                assert bool(torch.isfinite(values[0, extra[0]])) == (words < 22)
                assert all(bool(torch.isneginf(values[0, stop])) for stop in self.eos_ids)
        if str(device) != 'cpu':
            torch.cuda.synchronize(device)
        return dict(CPU_input_GPU_score_path_pass=str(device) != 'cpu',
                    synthetic_facts_tokens=lengths[0], synthetic_caption_tokens=lengths[1],
                    dtype='bfloat16', scores_device=str(device), model_loaded=False,
                    bounded_string_CPU_input_score_path_pass=self.bounded,
                    caption_word_count_path_pass=self.word_count,
                    word_count_boundary_mask_pass=self.word_count)

    def check_token_path(self, prompt, tokens, constraints, mask, device):
        import torch
        sent = list(prompt)
        scores = torch.zeros((1, len(self.tokenizer)), device=device, dtype=torch.bfloat16)
        constraints['prefix_allowed_tokens_fn'](0, torch.tensor(sent, dtype=torch.long))
        for token in tokens:
            ids = torch.tensor([sent], dtype=torch.long, device='cpu')
            values = mask(ids, scores.clone())
            values = constraints['logits_processor'](ids, values)
            assert bool(torch.isfinite(values[0, token]))
            assert all(bool(torch.isneginf(values[0, eos])) for eos in self.eos_ids)
            sent.append(token)
        ids = torch.tensor([sent], dtype=torch.long, device='cpu')
        values = constraints['logits_processor'](ids, mask(ids, scores.clone()))
        assert bool(torch.isfinite(values[0, self.tokenizer.eos_token_id]))
