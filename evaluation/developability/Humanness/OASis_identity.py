"""Calculate full-VHH OASis identity at 50% prevalence for one sequence library."""
import math
from pathlib import Path
import sqlite3
import sys
from itertools import islice

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (
    PROJECT_ROOT, SEQUENCE_FIELDS, STATUS_FIELDS, iter_sequences, output_path,
    parse_args, positive_int, project_path, write_results,
)

METRIC_FIELDS = ["oasis_identity", "oasis_human_9mers", "oasis_total_9mers"]


def get_oasis_human_peptides(connection, peptides, minimum_subjects, query_size):
    human = set()
    peptide_list = sorted(peptides)
    for start in range(0, len(peptide_list), query_size):
        batch = peptide_list[start:start + query_size]
        placeholders = ",".join("?" for _ in batch)
        statement = f"""
            SELECT peptides.peptide, COUNT(*) AS num_subjects
            FROM peptides
            LEFT JOIN subjects ON peptides.subject = subjects.id
            WHERE peptides.peptide IN ({placeholders})
              AND subjects.StudyPath <> 'Corcoran_2016'
              AND subjects.CompleteHeavySeqs >= 10000
            GROUP BY peptides.peptide
            HAVING COUNT(*) >= ?
        """
        human.update(row[0] for row in connection.execute(statement, [*batch, minimum_subjects]))
    return human


def calculate_records(args, connection):
    # Preserve the original denominator and study filter used by the research script.
    total_subjects = connection.execute(
        "SELECT COUNT(*) FROM subjects WHERE CompleteHeavySeqs >= 10000"
    ).fetchone()[0]
    if total_subjects == 0:
        raise ValueError("OASis database has no subjects with CompleteHeavySeqs >= 10000")
    minimum_subjects = math.ceil(total_subjects * 0.5)
    records = iter(iter_sequences(args))
    while True:
        batch = list(islice(records, args.batch_size))
        if not batch:
            return
        peptide_lists = [
            [record["vhh"][index:index + 9] for index in range(len(record["vhh"]) - 8)]
            for record in batch
        ]
        unique = {peptide for peptides in peptide_lists for peptide in peptides}
        human = get_oasis_human_peptides(connection, unique, minimum_subjects, args.query_size)
        for record, peptides in zip(batch, peptide_lists):
            result = {key: record[key] for key in SEQUENCE_FIELDS}
            if not peptides:
                result.update(status="failed", error_message="Full VHH must contain at least 9 residues")
            else:
                count = sum(peptide in human for peptide in peptides)
                result.update(
                    oasis_identity=count / len(peptides), oasis_human_9mers=count,
                    oasis_total_9mers=len(peptides), status="success", error_message="",
                )
            yield result


def configure(parser):
    parser.add_argument("--oasis-db", type=project_path, default=PROJECT_ROOT / "datasets/OASis_9mers_v1.db")
    parser.add_argument("--batch-size", type=positive_int, default=2000)
    parser.add_argument("--query-size", type=positive_int, default=800)


def main():
    args = parse_args(__doc__, configure=configure)
    connection = sqlite3.connect(args.oasis_db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return write_results(
            output_path(args, "oasis_identity"), SEQUENCE_FIELDS + METRIC_FIELDS + STATUS_FIELDS,
            calculate_records(args, connection),
        )
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
