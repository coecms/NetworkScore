# NetworkScore

The paper associated with this code has been accpeted for publication and the DOI will be placed here upon publication.

This python program builds a spider web graph using the ```Networkx``` library, based on the Aoyanagi and Okumura (2010) model. This web is defined by ```n``` radials and ```m``` rings that form concentric circles. Each point of intersection of radial lines and rings is a node of the web. You select three target nodes to be the basis of the network efficiency. 

Superimposed over the spider’s web is a grid. We use a 100 by 100 grid but any grid dimension is available. Some individual grid boxes (cells) do not intersect with the spider’s web (e.g. at the
corners of the grid) and hence extreme events can occur that have no impact on the web. Below is an example of a 10 x 10 grid with a web of 9 radials and 8 rings:

<img src="https://github.com/coecms/NetworkScore/assets/20108650/4905d804-fdbc-4c59-ba03-4fbbd83ea669" alt="Image" width="300">

An ```xarray``` dataset is created with the same grid dimensions and contains the 'events' that may occur at each point in the grid. This dataset has two basic characteristics:
- The magnitude of an event. For example, an event of size 1 stays on the grid for a 1 timestep, an event of size 3 stays on the grid for 5 timesteps, etc.
- The frequency of events. The frequency of each magnitude can be varied with a weight.

## Example usage:

```python
python NetworkScore.py 1000 1000 100 100 --web 8 9 --score-nodes "54,28,51" --output output.csv
```
where 1000 1000 is the map_height and map_width, respectively. 100 100 is the grid_rows and grid_cols. -- web 8 9 sets up a spider web of radials and rings, --score-nodes "54,28,51" spcifies which are the target nodes, and --output output.csv specifies the csv file the data is saved too. If you want to see the other options or any more info then you  an run:

```
python NetworkScore.py --help
```
## Installation notes:

See the requirements.txt file for required python libraries.
