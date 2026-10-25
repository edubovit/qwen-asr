import json
import re
import unicodedata

from qwen_asr_cli.runtime import workspace_root


def words(text):
    text = "".join(char for char in text.casefold()
                   if not unicodedata.category(char).startswith("P"))
    return re.findall(r"\S+", text)


def distance(left, right):
    row = list(range(len(right) + 1))
    for index, word in enumerate(left, 1):
        following = [index]
        for column, other in enumerate(right, 1):
            following.append(min(following[-1] + 1, row[column] + 1,
                                 row[column - 1] + (word != other)))
        row = following
    return row[-1]


def main():
    directory = workspace_root() / ".local" / "benchmark-attention"
    names = ["torch29-sdpa", "torch210-sdpa", "torch210-flash"]
    reports = {name: json.loads((directory / f"{name}.json").read_text()) for name in names}
    logs = {name: [json.loads(line) for line in
                   (directory / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
            for name in names}
    comparisons = {}
    for baseline in names[:2]:
        left, right = logs[baseline], logs[names[2]]
        assert len(left) == len(right)
        assert all((a["start"], a["end"]) == (b["start"], b["end"]) for a, b in zip(left, right))
        differences = sum(distance(words(a["text"]), words(b["text"]))
                          for a, b in zip(left, right))
        total = sum(len(words(row["text"])) for row in left)
        comparisons[baseline] = {
            "speedup": reports[baseline]["inference_seconds"]
                       / reports[names[2]]["inference_seconds"],
            "different_raw_chunks": sum(a["text"] != b["text"] for a, b in zip(left, right)),
            "word_edits_relative_to_baseline": differences,
            "baseline_words": total,
            "word_disagreement_rate": differences / total if total else None,
        }
    result = {"runs": reports, "flash_vs": comparisons,
              "limits": "Single full pass per configuration after warmup. Not an accuracy "
                        "benchmark: there is no human reference. ASR only, no timestamp alignment. "
                        "Timing excludes decode, model load, warmup and profiling."}
    (directory / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
