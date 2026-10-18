"""Generate the fixed, nonhuman two-event reference probe; no model or data download."""
import argparse
from pathlib import Path
from ipvg.common import sha256_file, write_json
from ipvg.data import write_manifest


CASES = (
    ('red_ball', (240, 20, 20), 'ellipse', ('moves smoothly to the right', 'moves smoothly to the left')),
    ('blue_square', (20, 40, 240), 'rectangle', ('moves smoothly upward', 'moves smoothly downward')),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Choose a new probe directory; preserve previous inputs')
    if not output.parent.is_dir():
        raise FileNotFoundError('Output parent must exist')
    from PIL import Image, ImageDraw
    output.mkdir()
    rows = []
    for subject, color, shape, actions in CASES:
        image = Image.new('RGB', (320, 192), (235, 235, 235))
        getattr(ImageDraw.Draw(image), shape)((136, 72, 184, 120), fill=color)
        reference = output / (subject + '.png')
        image.save(reference)
        for seed in (42, 43):
            rows.append({
                'sample_id': subject + '_seed' + str(seed), 'identity_id': subject,
                'source_video_id': subject + '_programmatic', 'reference_images': [str(reference)],
                'prompt': 'A single ' + subject.replace('_', ' ') + ' on a plain light gray background. Fixed camera, only the object moves.',
                'timeline': [{'start_s': 0, 'end_s': 2, 'action': 'The ' + subject.replace('_', ' ') + ' ' + actions[0] + '.'},
                             {'start_s': 2, 'end_s': 4.125, 'action': 'The ' + subject.replace('_', ' ') + ' ' + actions[1] + '.'}],
                'track': 2, 'split': 'dev', 'seed': seed,
                'source': 'self-generated primitive, no restricted media'})
    write_manifest(output / 'pilot.jsonl', rows[:1])
    write_manifest(output / 'batch.jsonl', rows[1:])
    write_manifest(output / 'manifest.jsonl', rows)
    write_json(output / 'preparation_report.json', {
        'status': 'passed', 'cases': 4, 'seeds': [42, 43], 'reference_size': [320, 192],
        'reference_sha256': {name + '.png': sha256_file(output / (name + '.png')) for name, _, _, _ in CASES},
        'manifest_sha256': sha256_file(output / 'manifest.jsonl'),
        'notice': 'Use identical inputs for flatten, window and swapped modes; not a general action or identity benchmark.'})
    print('Prepared 4 cases: 1 pilot and 3 batch; reuse the same inputs for all three modes')


if __name__ == '__main__':
    main()
