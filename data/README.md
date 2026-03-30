# Data layout for the Course Planning Assistant

Place **real PDFs** in these subfolders under `data/`. The indexer scans **every** `.pdf` file **recursively** inside each subfolder.

| Folder | Purpose | `doc_type` tag (for metadata) |
|--------|---------|-------------------------------|
| **`academic_policy/`** | Regulations, exams, attendance, grading, prerequisite enforcement, etc. | `policy` |
| **`courses/`** | Individual course or subject outlines (e.g. 12 course PDFs), syllabi, prerequisites per course | `course_catalog` |
| **`programme_requirements/`** | B.Tech eligibility, admission, degree structure, credit rules, programme-level requirements | `program_requirements` |

Folder names are matched **case-insensitively** and spaces are fine (e.g. `Academic Policy`, `Programme Requirements`). If the folder name contains **requirement**, **eligibility**, or **admission**, it is treated as programme requirements. If it contains **course** or **catalog**, it is treated as courses.

### Legacy

Optional: you may still put PDFs in **`data/pdfs/`** (flat). They are indexed with `doc_type` inferred from the **filename**.

### After adding or changing PDFs

Rebuild the vector index:

```bash
python main.py ingest
```

Or start the Flask app and use **Reindex all PDFs**.

### Optional synthetic data

To regenerate demo PDFs (not needed if you use real files):

```bash
python data/generate_synthetic_catalogs.py
```

That script writes to `data/pdfs/` by default (legacy layout).
