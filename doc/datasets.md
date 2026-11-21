# Data Set

## Purpose

Description of the purpose, selection, output and limitation of the data set.

## Scope

For the selection of actual operations, the interpretation of results and the maintenance of project descriptions.

## Purposes and access

SAEBASE assigns samples of questions and answers, language behavior and security assessments to language models and to slur-codifiers. The following is a list of official data set configurations.

## Shared Parameters

Controls a fixed sample `dataset.sample_count` and `dataset.sampling_seed`. `dataset.use_original_split` decides whether to follow the input split. Full take-up values can be found in [parameter](conf.md) (for configuration files, default values and overlay rules, how the model's operating equipment is set). [Data configation directory](../conf/datasets/catalog.yaml)]

## Component

### ARCChallenge（`arc_challenge`）

**Scientific reasoning selection **. Composed of questions, options and correct answers, use the ARC-Challenge subset. It is possible to compare the fractions given by the model to the right option to change before and after the intervention.

The ** item is used for **. Select with `dataset=arc_challenge`. Presets 500 entries by torrent `42`. Configure the selected records as training or assessment inputs for this task. Data preparation is followed by text missions to read tips and reference targets. The difficulty of the subject and the selection rating do not represent open scientific reasoning.

This adaptation is clearly supported by `capture`, `intervene`, `generate`, and `precheck`.

[Project fit](../datasets/arc_challenge/) · [Upstream information](https://arxiv.org/abs/1803.05457)

### BBH（`bbh`）

**Multiple difficult-to-excise tasks**.BIG-Bench Hard . The sub-task covers different types of questions, such as date extrapolation or logical reasoning. SAEBase Compared feature capture methods on a fixed title.

The ** item is used for **. Select with `dataset=bbh`. Presets 500 entries by torrent `42`. Configure the selected records as training or assessment input for this task. Data preparation is followed by text jobs to read tips and reference targets.

This adaptation is clearly supported by `capture`, `intervene`, `generate`, and `precheck`.

[Project fit](../datasets/big_bench_hard/) · [Upstream information](https://github.com/suzgunmirac/BIG-Bench-Hard)

### CounterFact（`counterfact`）

** Factual Completion **. The hint usually gives the entity and the relationship, and the label distinguishes between the correct facts and the alternative facts used for the comparison.

The ** project is used as **. Select with `dataset=counterfact`. Presets 500 entries by torrent `42`. Configure the selected records as training or assessment inputs for this task. Data preparation is followed by text missions reading tips and reference targets. The subset is determined by a specific model. Cross-models look at the intersection of the topics that the parties retain.

[Project fit](../datasets/counter_fact/) · [Upstream information](https://rome.baulab.info/)

### GreaterThan（`greater_than`）

** year size comparison **. The input is an unwritten range of years, and the model needs to be continued with the next two numbers larger than the first year. Read out that the comparison is greater than the candidate with no greater than the threshold and is prepared to create a pool of candidates by the current model word.

**Project usage**.Here. `dataset=greater_than` Select. Defines the number of actual retentions by using input record labels, depending on the data preparation and the model selected. The text job reading tips and reference targets after data preparation. When the fraction does not meet the requirements of the grouping, the availability of the sample and the comparable range between the different models changes.

[Project fit](../datasets/greater_than/) · [Upstream information](https://arxiv.org/abs/2305.00586)

### GSM8K（`gsm8k`）

** Primary school math **. The question gives a day-to-day context in which the reference answer contains the calculation process and the final number. It can be used to see how the final answer to the model or the resulting answer changes with the characteristics.

**Project usage**.Here. `dataset=gsm8k` Select. Preset by torrent `42` Select 500 bar. Configure the selected records as training or assessment input for this task. Data preparation is followed by text task reading tips and reference targets. Matching the final number can only measure the calibre of the answer. Process interpretation or reasoning quality needs to be assessed separately..

This adaptation is clearly supported by `capture`, `intervene`, `generate`, and `precheck`.

[Project fit](../datasets/grade_school_math/) · [Upstream information](https://github.com/openai/grade-school-math)

### HarmBench（`harmbench`）

**Security Conduct Assessment**.The tasks and labels are determined by the project agreement and data adaptation..

**Project usage**.Here. `dataset=harmbench` Select. Preset by torrent `42` Select 500 bar. Configure the selected records as training or assessment input for the current mission. Data preparation is followed by text task reading tips and reference targets. The results depend on the classification and response assessment calibration, and the overall security cannot be inferred from a single model output..

This adaptation is clearly supported by `capture`, `intervene`, `generate`, and `precheck`.

[Project fit](../datasets/harm_bench/) · [Upstream information](https://arxiv.org/abs/2402.04249)

### HotpotQA（`hotpotqa`）

** Multiject questions **. Each question has a natural language question and reference answer, and the original data also provides a cross-section-supported fact. It is appropriate to study how models fit together multiple clues.

**Project usage**.Here. `dataset=hotpotqa` Select. Preset by torrent `42` Select 500 bar. Configure the selected records for the current task as a training or assessment input. Data preparation is followed by text jobs to read the hints and reference targets. The comparison results need to indicate whether the input contains the supporting paragraph and the answer rating method used..

This adaptation is clearly supported by `capture`, `intervene`, `generate`, and `precheck`.

[Project fit](../datasets/hotpot_qa/) · [Upstream information](https://aclanthology.org/D18-1259/)

### HumanEval（`humaneval`）

**Python Function Completion**.The original sample provides a function signature, description, reference realization and test. The project can use a programming hint to study the creation or characteristics, check the tasks selected for the configuration and read them out..

**Project usage**.Here. `dataset=humaneval` Select. Preset by torrent `42` Select 500 bar. Configure the selected records for training or evaluating input according to this task. Data preparation is followed by text task reading of tips and reference targets. The probability of the reference code, text similarity and cell testing are three different results..

This adaptation is clearly supported by `capture`, `intervene`, `generate`, and `precheck`.

[Project fit](../datasets/human_eval/) · [Upstream information](https://github.com/openai/human-eval)

### IOI（`ioi`）

** Indirect guest recognition **. The sentence describes two people and gives uncompleted junctions.

The ** item is used for **. Select with `dataset=ioi`. Presets the 500 by torrent `42`. Configure the selected record as a training or assessment input for this task. The text job will read the hint and reference target after the data are prepared. Models can be used to use samples of human names and intended filtering effects. The conclusions apply to the selected sentence.

[Project fit](../datasets/indirect_object_identification/) · [Upstream information](https://arxiv.org/abs/2211.00593)

### MMLUPro（`mmlu_pro`）

**Interdisciplinary choice.**.The sample consists of a dry, alternative and correct answer letter. When prepared, checking the answer and the interference letter can be used as a single word in the model, filtering the hyperlong hint..

**Project usage**.Here. `dataset=mmlu_pro` Select. Preset by torrent `42` Select 500 bar. The selected records are configured for training or assessment input for the current task. The text job is prepared to read the hints and reference targets. The selected subject and answer letter rating determines the scope of the experiment, and the results and upstream scores are compared by the same tip and rating..

[Project fit](../datasets/mmlu_professional/) · [Upstream information](https://github.com/TIGER-AI-Lab/MMLU-Pro)

### RAVEL（`ravel`）

** Language Model Properties decomposition **. Enter both the entity and the attribute, for example, the two properties corresponding to the city.

The ** project is used as **. Select with `dataset=ravel`. Presets the 500 by torrent `42`. Configure the selected records as training or assessment inputs for this task. The text-reading tips and reference targets after the data are prepared. In order to describe both the target properties and the maintenance properties, a change in the single target fraction cannot explain the level of decomposition.

This adaptation is clearly supported by `capture`, `intervene`, `generate`, and `precheck`.

[Project fit](../datasets/ravel/) · [Upstream information](https://arxiv.org/abs/2402.17700)

### SimpleCopying（`simple_copying`）

** Short series copy **. Projectes B using the A, B and A Sequence Indices model. When preparing, compares the word fractions of the model for the correct continuation and interference items.

** Items are used **. Select by `dataset=simple_copying`. Splits the input record label, with the actual number of retentions depending on the data preparation and the model selected. Data preparation is followed by text jobs reading the hints and reference targets. This is checked in the controlled short context mode and cannot be directly extrapolated to a long document for reproduction.

[Project fit](../datasets/simple_copying/)

### TruthfulQA（`truthfulqa`）

** Commonly misunderstood factual answer **. The title is based on situations that are prone to erroneous answers or popular misunderstandings. The reference answers are used to compare model answers.

The ** project is used at **. Select by `dataset=truthfulqa`. Presets 500 entries by torrent `42`. Configure the selected records for training or assessment input according to this task. The data are prepared by text jobs to read tips and reference targets. The operational report should indicate a rating by answer option, production of a factual judgement or other calibration.

This adaptation is clearly supported by `capture`, `intervene`, `generate`, and `precheck`.

[Project fit](../datasets/truthful_qa/) · [Upstream information](https://arxiv.org/abs/2109.07958)

### XSTest（`xstest`）

**Over-rejection check**.Check if the model refuses a non-hazardous request because of surface sensitive words with a cross-tip that should answer and should be rejected.

**Project usage**.Here. `dataset=xstest` Select. Preset by torrent `42` Select 500 bar. Configure the selected records to be used as training or evaluation input for this task. The text job to read the hint and reference target after data preparation. The rejection rate is to be explained separately by the hint category, and not all rejections are considered to have failed..

This adaptation is clearly supported by `capture`, `intervene`, `generate`, and `precheck`.

[Project fit](../datasets/xs_test/) · [Upstream information](https://arxiv.org/abs/2308.01263)


## Independent pointer source

The following sources are provided by a register of indicative sources. The coverage is determined by a prepared model-specific prompt product through a corresponding text experiment or a baseline formulation selection.

### MedQA

The results describe the behaviour of the model under the corresponding test and scoring rules. [local hint formulation](../conf/datasets/catalog.yaml)]

### MATH500

A subset of assessments of mathematical topics, which is entered as a question, is taken from the answers prepared by the project. It is appropriate to assess the impact of characteristic interventions on the answer on a fixed subject. The subset, the resolution of the answer and the complete solution quality are recorded separately. [local hint formula](../conf/datasets/catalog.yaml)]

### GPQAExtended

The project uses the title and answer format of the model to compare. The report needs to indicate which actual subset and answer rating is used. [Local tip formulation](../conf/datasets/catalog.yaml)]

### LiveCodeBench

The results generated by the answer probability or text should be interpreted according to the protocol used. The correct rate for the performance of the test is a separate indicator. [local hint formulation](../conf/datasets/catalog.yaml)]
