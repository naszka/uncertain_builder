# Code for the paper Aligning predictive uncertainty with clarification questions in grounded dialog

[link to the paper](https://aclanthology.org/2023.findings-emnlp.999/)

## How to

1. Install the [Iglu gridworld environment](https://github.com/iglu-contest/gridworld)


`pip install git+https://github.com/iglu-contest/gridworld.git@master`

The gridworld environment contains three datasets

(The Minecraft Dialog Corpus)[https://aclanthology.org/P19-1537/]
```python
from gridworld.data import CDMDateset
dataset = CDMDataset()
```

(The IGLU multiturn datacollection)[https://arxiv.org/abs/2305.10783]
```python
from gridworld.data import IGLUDataset
dataset = IGLUDataset()
```
(The IGLU single turn data collection)[https://arxiv.org/abs/2305.10783]
```python
from gridworld.data import SingleTurnIGLUDataset
dataset = SingleTurnIGLUDataset()
```

In this package we extend these datasets to include ambiguous dialog turns
```python
from uncertain_builder.data import QuestionDataset

# IGLU dataset where we included dialog turns
# that were followed by clarification questions
question_dataset = QuestionDataset(parse=True, split='test')
```

```python
from uncertain_builder.data import CDMQDataset

# CDMDataset where we included dialog turns
# that were followed by clarification questions
question_dataset = CDMQDataset(parse=True, split='test')
```

We also include two datasets that include ambiguity minimal pairs.
We extend the  SingleTurnIGLUDataset and the IGLUDataset this way

```python
from uncertain_builder.data import AmbiguityPairsDataset, AmbiguityPairsDatasetMulti

# CDMDataset where we included dialog turns
# that were followed by clarification questions
ambig_dataset = AmbiguityPairsDataset(parse=True, split='test')
ambig_dataset_multi = AmbiguityPairsDatasetMulti(parse=True, split='test')
```

Iterate over the datat. The is_question field indicates if the datapoint was followed by a question

```python
for task_id in question_dataset.tasks.keys():
    task = question_dataset.tasks[task_id]
        for j, subtask in enumerate(task):
            print(subtask.is_question)
```


