import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Offset a numeric CSV column by a delta.")
    parser.add_argument("--input", required=True, help="Path to input CSV.")
    parser.add_argument("--column", required=True, help="Column name to adjust.")
    parser.add_argument("--delta", required=True, help="Decimal offset, e.g. -0.01 or 0.005.")
    parser.add_argument("--out", default="", help="Output CSV path (default: in-place).")
    parser.add_argument("--encoding", default="utf-8", help="File encoding.")
    parser.add_argument("--precision", type=int, default=None, help="Force fixed decimal places.")
    parser.add_argument(
        "--no-preserve-decimals",
        action="store_true",
        help="Do not preserve original decimal places.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report counts.")
    return parser.parse_args()


def format_value(value, raw, precision, preserve_decimals):
    if precision is not None:
        return f"{value:.{precision}f}"
    raw_has_exp = "e" in raw.lower()
    if preserve_decimals and not raw_has_exp:
        decimals = 0
        if "." in raw:
            decimals = len(raw.split(".", 1)[1])
        return f"{value:.{decimals}f}"
    return format(value, "f")


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    delta = Decimal(args.delta)
    preserve_decimals = not args.no_preserve_decimals

    with in_path.open("r", encoding=args.encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Missing CSV header.")
        rows = list(reader)

    if args.column not in fieldnames:
        raise ValueError(f"Missing column: {args.column}")

    updated = 0
    skipped_empty = 0
    skipped_invalid = 0
    for row in rows:
        raw = (row.get(args.column) or "").strip()
        if not raw:
            skipped_empty += 1
            continue
        try:
            value = Decimal(raw) + delta
            row[args.column] = format_value(value, raw, args.precision, preserve_decimals)
            updated += 1
        except (InvalidOperation, ValueError):
            skipped_invalid += 1
            continue

    out_path = Path(args.out) if args.out else in_path
    if not args.dry_run:
        with out_path.open("w", encoding=args.encoding, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(
        f"Updated {updated} rows (empty: {skipped_empty}, invalid: {skipped_invalid}). "
        f"Output: {out_path}"
    )


if __name__ == "__main__":
    main()
