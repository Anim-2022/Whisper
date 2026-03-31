import argparse
import re
from pathlib import Path


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def levenshtein(seq1, seq2) -> int:
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1
    if not seq2:
        return len(seq1)

    previous = list(range(len(seq2) + 1))
    for i, item1 in enumerate(seq1, start=1):
        current = [i]
        for j, item2 in enumerate(seq2, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if item1 == item2 else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, prediction: str) -> float:
    ref_words = normalize_text(reference).split()
    pred_words = normalize_text(prediction).split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return levenshtein(ref_words, pred_words) / len(ref_words)


def char_error_rate(reference: str, prediction: str) -> float:
    ref_chars = list(normalize_text(reference))
    pred_chars = list(normalize_text(prediction))
    if not ref_chars:
        return 0.0 if not pred_chars else 1.0
    return levenshtein(ref_chars, pred_chars) / len(ref_chars)


def evaluate(pred_dir: Path, ref_dir: Path):
    pred_files = sorted(pred_dir.glob("*.txt"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction .txt files found in {pred_dir}")

    total_wer = 0.0
    total_cer = 0.0
    matched = 0

    print(f"{'File':40} {'WER':>8} {'CER':>8}")
    print("-" * 60)

    for pred_path in pred_files:
        ref_path = ref_dir / pred_path.name
        if not ref_path.exists():
            print(f"{pred_path.name:40} {'missing':>8} {'missing':>8}")
            continue

        prediction = pred_path.read_text(encoding="utf-8")
        reference = ref_path.read_text(encoding="utf-8")

        wer = word_error_rate(reference, prediction)
        cer = char_error_rate(reference, prediction)
        total_wer += wer
        total_cer += cer
        matched += 1

        print(f"{pred_path.name:40} {wer:8.3%} {cer:8.3%}")

    if matched == 0:
        raise FileNotFoundError(f"No matching reference .txt files found in {ref_dir}")

    print("-" * 60)
    print(f"{'Average':40} {total_wer / matched:8.3%} {total_cer / matched:8.3%}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate transcription quality against reference .txt files."
    )
    parser.add_argument("--pred-dir", required=True, type=Path, help="Directory with predicted .txt files")
    parser.add_argument("--ref-dir", required=True, type=Path, help="Directory with reference .txt files")
    args = parser.parse_args()

    evaluate(args.pred_dir, args.ref_dir)


if __name__ == "__main__":
    main()
