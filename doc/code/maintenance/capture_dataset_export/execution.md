# Characteristics and Interpretation Data Set Export

## Purpose

Prints the already white and non_white collections of Parquet data, shared characterizations and standardized information into the existing Hugging Face data set.

## Scope

This user has approved the SAEBASe Capture export portal and OpenSAE/SAEBase-Evaluation update; read the available measurements, release readiness and cache to be stored in hugglingface-maintain skills.

## Background, objectives and approaches

Existing collections have saved two modes of activation and Neuronpedia interpretation. Separate the operation, sample, activation, dictionaries and standardized information from one another, allowing the interpretation of the same feature to be reused by multiple observations and retaining the numerical meanings of both rankings. Releases are written around data uses, reading methods and actual overlay.

## Implementation record

- User approves to collate and upload the existing results; use Capture to officially export the portal and Hugging Face official client.

- Add `tasks/capture/export.py` to support the selection of inputs in mode and matrix paths, with operational, sample, observation, shared interpretation and standardized records in the directory.
- Two existing matrices exported 104,766 samples, 8,644,248 characterization observations, 115,804 characteristics; 73,331 features contained explanations. 228 of 240 operational entries had collected data.
- Keep `tests/product/test_capture_export.py` to cover the actual Parquet reading, origin and standardization model, post-interpret sharing and standardization file; this execution is passed. Temporary test input has been deleted.
- Published copies contain 685 collection documents and updated dataset descriptions and source descriptions. Original measurements are kept as they are. Document-by-document check and check with table line numbers; actual data access links standardized observations to interpretation.

## AI Views and follow-up

The data are stored separately by operation, sample, observation and shared features and can be used directly for cross-model comparison and interpretation searches. Following the new collection results, a matrix is selected using the same export entry to generate a new release copy; the over-covered numbers in the data set description are taken from the current export list.

## Adjustments in Actual Reading

Remote daasets read the parsing format of the entire repository selected by the first configuration, with CSV configurations allowing new Parquets to be considered CSV. For 25 configurations, Parquet reading copies are generated, table-by-table confirmation of column values and types, and original CSVs are maintained. The original metric_results' matching paths match the two columns' different tables, and now clearly point to the original data/metric_results.csv and the detailed history table continues to be provided by historic_full_Metrics.31 The configurations are used in Parquet; local actual configurations are available to cover the six new entry and original indicator entry points.

## Release Results

Published at https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation and ultimately submitted 881889adfd2324e34e9b30d47e05e915224f4201. Actual reading at the five sector entrances at the remote end, 240 running records and 192 original indicator logs; all collected documents of the size and large remote SHA-256 file. The web page preview interface is currently busy and responsive; actual data access is available. Full release information is available at huggingface-maintain/results/publish-capture-20261115/.
