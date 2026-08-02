# Academic V2 prerequisites

MEDGUARD-CXR Academic V2 is research-only software. It is not a medical device and must not be used for diagnosis, treatment, triage, or patient care.

The run-all notebook can automate engineering work, but it cannot accept legal terms or complete identity-bound account steps. Before a full run, the repository owner must complete:

- MIMIC-CXR-JPG: become a PhysioNet credentialed user, complete the required CITI training, and sign the data-use agreement.
- VinDr-CXR: obtain access under the applicable PhysioNet DUA or Kaggle terms.
- CheXpert: accept the dataset usage terms and obtain download access.
- RSNA Pneumonia Detection: accept the Kaggle competition/dataset terms.
- Gated models: accept any model-specific license or Hugging Face terms.

Set the following Colab Secrets once, as applicable: `GITHUB_TOKEN`, `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`, `PHYSIONET_USERNAME`, and `PHYSIONET_PASSWORD`. The notebook checks only whether a secret exists and never prints its value.

Restricted images, reports, patient identifiers, raw label rows, path lists, and per-sample predictions must remain under the private Google Drive workspace. They must not be copied into the repository, public result branch, report bundle, or release archive.

After access is in place, the intended human workflow is: select a Colab GPU runtime, open `notebooks/00_medguard_academic_v2_run_all.ipynb`, and choose **Run all**. Re-running the notebook uses the run registry and atomic markers to resume rather than silently duplicating completed work.

Current implementation note: the initial Academic V2 slice provides contract validation, deterministic planning, access blocking, and synthetic resume proof. Real training executors for every planned stage are not yet complete; those runs remain `pending` rather than being misreported as completed.
