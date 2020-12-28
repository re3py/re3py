# RE3PY - relational tree learners in python!
Welcome to RE3PY's official repo. To install the library, please try 
(latest):
```
pip install -r requirements.txt

python setupy.py install
```
For additional visualizations, please also install py3plex library via `pip install py3plex` call.

# Testing
To execute the series of tests which determine proper installation, execute:
```
python -m pytest tests
```
![gif-reel](images/reel.gif)

# Full replicability
To fully replicate the working environment, we offer a singularity container.
Go to ./container folder and run
```
bash make.sh
```
Note that you will need singularity installed for this to work, see https://github.com/hpcng/singularity.
An image called ranking.sif will be created in ./container folder. You can run arbitrary scripts that include rrank library by simply doing:
```
singularity exec ranking.sif python somescript.py --some_parameter --some_other_parameter 
```

# Usage examples
The series of core examples is available in the ./examples folder.

# Syntax

In order to grow a tree (or an ensemble of trees), we need

- a file with descriptive relations
- a file with target relation
- a settings file

## Descriptive- and Target-Relation Files
Relations are thought of as sets, e.g., the relations such as `likes(Person, Food)` (`Ana likes potato`, `Ana likes beef`, `Bob likes beef`, etc.) or `loveTriangle(Person1, Person2, Person3)` (`loveTriangle(Ana, Bob, Charles)`, `loveTriangle(Denis, Dolores, Deborah)`, etc.) are represented as exhaustive listings of facts.

Descriptive-relations file contains the facts for all the descriptive relations (those that can be used in the tree splits), whereas the target-relation file contains only the description of the target relation.

In both files, a fact per line is given, and the syntax is the following:

```
likes(Ana, potato)
likes(Ana, beef)
likes(Bob, beef)
...
likes(Jaques, omlette)
loveTriangle(Ana, Bob, Charles)
...
```

The order of the facts is not important.

## Settings File
The settings file has three important sections:

- relations
- aggregates
- atom tests

### Relations


### Aggregates
The supported aggregates are count, count unique, min, max, mean, sum, mode, flatten, flatten unique, and projection.




## Other Parameters
are passed directly to the learner's constructor (e.g., depth of a tree, maximal number of atom tests in a split etc.)

## Example

TODO

### Standard tabular data forms
Dataset in the stadard form

| x1  | ... | xN |  y   |
| ----- | ---- | ---- | ----  |
| M     |  ... | 2.2 |  red   |
| F     |  ... | 3.2 |  black |
|  ...  | ... | ...  | ...    |

can be converted into descriptive relations

```
x1(example1, M)
...
xN(example1, 2.2)
x1(example2, F)
...
xN(example2, 3,2)
...
```

and target relation

```
y(example1, red)
y(example2, black)
...
```

If we then specify

- the relations as `y(Example, nominal1)`, `x1(Example, nominal2)`, ..., `xN(Example, numericJ)`,
- the atom tests as `x1(old, new)`, ... , `xN(old, new)`
- the allowed aggregates as `mode`, `sum` (one for nominal and one for numeric attribtues - does not matter which since all the relations are -to-one),
- the number of atom tests in a node as `1`

the resulting relational decision tree should be the same as the standard _non-relational_ decision tree for this dataset.

