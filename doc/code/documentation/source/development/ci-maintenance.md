# Continuous integration and scientific validation

## Purpose

Description of the results confirmed by the daily inspection of the project, central access and the release of the certification.

## Scope

This applies to the maintenance of workflows, the selection of the scope of inspections and the interpretation of the validation results.

## Maintenance of inspection function

When you add or modify an inspection, you first determine what, what or what work the developer needs to locate after receiving the result, and then select the object and the content of the inspection. Reasons, documents and recommendations for processing are provided from the module that is the subject of the inspection, and they are retained in the aggregation process. Check the scene to read the results from the actual call access and use this information to process the problem and select the circumstances that apply to this change.

## Daily inspections

[Workstream](../../../../../.github/workflows/ci.yml)Based on Git Change selection check. Documents and skills statements are reviewed by content, project-in-project access information is checked using local interfaces, and scientific codes are tested using the corresponding product..

The day-to-day workflow uses the contents of the warehouse and the development dependencies required; central environmental queries are called by a clear central access task. The results are summarized in `ci-required`.

## Central access

The central entrance is at [access instructions](../../../../../.github/CONTRIBUTING.md)]. The normal query uses the current binding and actual interface, requiring a fixed version of the visual access to the expected engine file.

## Publication and scientific research

[Organisation](../../../../../.github/workflows/release-validation.yml)Starts with a clearly selected source code to check software installation, complete offline functionality, packing and selection methods.[Release Process](release-process.md).

[The return of scientific methods](../../../../../.github/workflows/research-regression.yml)Use fixed original SAR and RSS Source Check Integration.[Upstream compatibility inspections](../../../../../.github/workflows/upstream-compatibility.yml)Used to observe upstream changes.

Software tests, synthetic models, authentic model experiments and scientific findings each maintain the corresponding factual evidence for the relevant maintenance and scientific missions.
