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

The following sections describe each of them.

## Descriptive- and Target-relation Files
In the code and in the input files, relations are thought of as sets whose elements are given exhaustively. Every such set contains tuples of objects of given types.
For example, if `Ana likes potato`, `Ana likes beef`, `Bob likes beef`, `likes(Jacques, omlette)`, etc., this can be encoded as the relation

`likes(Person, Food) = {(Ana, potato), (Ana, beef), (Bob, beef), (Jacques, omlette), ...}`.

The relation does not have to be binary. Any other arity (greater or equal to 1) is possible. For example, the ternary relation `loveTriangle(Person, Person, Person)` is given as  

`loveTriangle(Person, Person, Person) = {loveTriangle(Ana, Bob, Charles)`, `loveTriangle(Denis, Dolores, Deborah), ...}`.

The relations are divided into two groups:

- **Descriptive relations** are the relations that can be used in the tree splits. They can be unary, binary, ternary, etc.
- **Target relation** is the relation whose elements need to be predicted. For technical reasons, the target relation **cannot be unary**.

### Descriptive-relations file
Descriptive-relations file contains the elements of all the descriptive relations, one element of a relation per line:

```
likes(Ana, potato)
likes(Ana, beef)
likes(Bob, beef)
likes(Jacques, omlette)
...
loveTriangle(Ana, Bob, Charles)
loveTriangle(Denis, Dolores, Deborah)
...
```

The spaces are optional and the order of the facts is not important.

### Target-relation file

This file contains only the elements of the target relation (the relation whose elements need to be predicted). The same syntax as above applies.
For example, if we are predicting the whether a person is vegetarian or not, the target relation should be given as

```
vegetarian(Ana, no)
vegetarian(Bob, no)
vegetarian(Jacques, yes)
vegetarian(Charles, no)
...
```

For a given datum, e.g., `vegetarian(Person, nominal)`, a learner uses only the `Person` part and returns a value of the type `nominal`.
The values that we are predicting (above, these are the `yes` and `no` values) should always be on the last component in the tuples.

Note that `vegetarian` is actually a unary relation. The trick with the additinal component is only necessary to make other problems (including multi-class or regression) simpler.
For example, we can predict relations such as

- `friends(Person, Person, nominal) = {(Ana, Bob, yes), (Ana, Charles, yes), (Ana, Denis, no), ...}` where (given a pair of `Person`s) `yes` or `no` is predicted
- `age(Person, numeric) = {(Ana, 19), (Bob, 18), (Denis, 45), ...}` where (given a `Person`) the numeric value is predicted
- ...


## Settings File
The settings file has three important sections:

- relations
- aggregates
- atom tests

The syntax is the following:



### Relations


### Aggregates
The supported aggregates are count, count unique, min, max, mean, sum, mode, flatten, flatten unique, and projection.


### Atom tests

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

