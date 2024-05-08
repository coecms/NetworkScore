# NetworkScore
This is a python program that builds a network (graph) of nodes on a plane and uses an xarray dataset of events at the grid locations.
The edges of the graph that are in the grid locations are deleted.
Note, in graph theory, edges are the lines that connect two nodes.

There are many command line parameters, use the --help command line parameter to see a list of all of them.
The configure_args function loads a global array called config with all of the command line parameters, which is used by the functions in this script

## Example usage:

```python
python NetworkScore.py 1000 1000 100 100 --web 8 9 --score-nodes "54,28,51" --output output.csv

python NetworkScore.py --help
```
## Installation notes:

See the requirements.txt file for required python libraries.
