#! /usr/local/bin/python

"""
NetworkScore.py

Authors: Ed Saribatir, Samuel Green

This is a python program that builds a network (graph) of nodes on a plane and uses an xarray dataset of events at the grid locations.
The edges of the graph that are in the grid locations are deleted.
Note, in graph theory, edges are the lines that connect two nodes.

There are many command line parameters, use the --help command line parameter to see a list of all of them
The configure_args function loads a global array called config with all of the command line parameters, which is used by the functions in this script

Example usage:
python NetworkScore.py --help

python NetworkScore.py 1000 1000 100 100 --web 8 9 --score-nodes "54,28,51" --output output.csv

Installation notes:

pip install networkx
pip install xarray
"""

import argparse
import csv
import inspect
import math
import networkx as nx
import numpy as np
import os
import random
import sys
import xarray as xr
from datetime import datetime

################ Setup variables: ################
config 			= None

time			= 0
timestepevents	= 0
timestamp		= None
scoremax		= 0

rounding_decimal_places = 6

# data is an associative array, each element contains a child associative array with 3 elements
# timestep, timestepevents, score
data = {}

# the network_graph is the original graph structure, before any events have occured
network_graph = nx.MultiGraph()

# the timestep_graph is the current state of the graph for the current timestep (edges may have been removed due to events)
timestep_graph = nx.MultiGraph()

# nodeids are integers, the center nodeid is 0
centernodeid = 0
##################################################

def new_node(nodeid, x, y):
	"""
	Add a new node to the network graph.

	Args:
		nodeid (int): The ID of the node.
		x (float): The x-coordinate of the node.
		y (float): The y-coordinate of the node.

	Returns:
		None
	"""
	col = x // config['grid_tile_width']
	row = y // config['grid_tile_height']

	network_graph.add_node(nodeid, x=x, y=y, row=row, col=col, id=nodeid, label=nodeid)

def new_edge(nodeid1, nodeid2):
	"""
	Create a new edge between two nodes in the network graph.

	Args:
		nodeid1 (int): The ID of the first node.
		nodeid2 (int): The ID of the second node.

	Returns:
		None
	"""
	tiles = edge_tiles(nodeid1, nodeid2)
	network_graph.add_edge(nodeid1, nodeid2, "Edge_" + str(nodeid1) + "_" + str(nodeid2), tiles=tiles)

def debug_nodes(graph):
	"""
	Print the nodes of the given graph for debugging purposes.
	
	Parameters:
		graph (object): The graph object containing nodes.

	Returns:
		None
	"""
	nodes = ' '.join(map(str, graph.nodes()))
	debug(nodes)

def edge_tiles(nodeid1, nodeid2):
	"""
	Returns a list of tiles that form the edge line between two nodes in a network graph that pass through grid squares.

	Parameters:
	- nodeid1 (int): The ID of the first node.
	- nodeid2 (int): The ID of the second node.

	Returns:
	- tiles (list): A list of dictionaries representing the tiles. Each dictionary contains the 'col' and 'row' coordinates of a tile.

	"""
	global time

	# Access the node with id nodeid1/2 in the network graph
	node1 = network_graph.nodes[nodeid1]
	node2 = network_graph.nodes[nodeid2]

	# Get the (x,y)-coordinates of node1 and node2
	x1, x2 = node1['x'], node2['x']
	y1, y2 = node1['y'], node2['y']

	# Calculate the column and row coordinates for node1 and node2
	col1, col2 = x1 // config['grid_tile_width'], x2 // config['grid_tile_width']
	row1, row2 = y1 // config['grid_tile_height'], y2 // config['grid_tile_height']

	# Debug information about the nodes and their coordinates
	debug('node1: ' + str(nodeid1) + ' (' + str(col1) + ',' + str(row1)) + ' (' + str()')
	debug('node2: ' + str(nodeid2) + ' (' + str(col2) + ',' + str(row2) + ')')

	# Initialize the tiles list with the tile of the first node
	tiles = [{'col': col1, 'row': row1}]

	# If the nodes are not in the same tile
	if not (col1 == col2 and row1 == row2):
		# Add the tile of the second node to the tiles list
		tiles.append({'col': col2, 'row': row2})
	else:
		# If the nodes are in the same tile, return the tiles list
		return tiles

	# True if the line between the nodes is vertical
	noslope 	= (col1 == col2)
	# True if the line between the nodes is horizontal
	zeroslope 	= (row1 == row2)

	if noslope:
		# If the line between the nodes is vertical
		col = col1
		# If the first node is above the second node
		if row1 < row2:
			# Set the row increment to 1
			drow = 1
			# Append all tiles from the first node to the second node to the list
			for row in range(row1 + drow, min(row2, config['grid_rows']), drow):
				tiles.append({'col': col, 'row': row})
		 # If the first node is below the second node
		else:
			# Set the row increment to -1
			drow = -1
			# Append all tiles from the first node to the second node to the list
			for row in range(row1 + drow, max(row2, 0), drow):
				tiles.append({'col': col, 'row': row})

	elif zeroslope:
		# If the line between the nodes is horizontal
		debug('zeroslope')
		row = row1
		# If the first node is to the left of the second node
		if col1 < col2:
			# Set the column increment to 1
			dcol = 1
			# Append all tiles from the first node to the second node to the list
			for col in range(col1 + dcol, min(col2, config['grid_cols']), dcol):
				tiles.append({'col': col, 'row': row})
		 # If the first node is to the right of the second node
		else:
			# Set the column increment to -1
			dcol = -1
			# Append all tiles from the first node to the second node to the list
			for col in range(col1 + dcol, max(col2, 0), dcol):
				tiles.append({'col': col, 'row': row})
	else:
		# If the line between the nodes is neither vertical nor horizontal
		col = col1
		row = row1

		# While the current tile is not the second node's tile
		while not (col == col2 and row == row2):
			debug('(col,row): (' + str(col) + ',' + str(row) + ')')
			debug('(col1,row1): (' + str(col1) + ',' + str(row1) + ')')
			debug('(col2,row2): (' + str(col2) + ',' + str(row2) + ')')
			# Calculate the next tile in the line based on the node coordinates and directions.
			# If the first node is to the left of the second node
			if x1 < x2:
				# If the first node is above the second node
				if y1 < y2:
					col, row = line_next_tile_col_row(col, row, x1, y1, x2, y2, 'top', 'right')
				# If the first node is below the second node
				else:
					col, row = line_next_tile_col_row(col, row, x1, y1, x2, y2, 'bottom', 'right')
			# If the first node is to the right of the second node
			else:
				# If the first node is above the second node
				if y1 < y2:
					col, row = line_next_tile_col_row(col, row, x1, y1, x2, y2, 'top', 'left')
				# If the first node is below the second node
				else:
					col, row = line_next_tile_col_row(col, row, x1, y1, x2, y2, 'bottom', 'left')
			debug('(col,row): (' + str(col) + ',' + str(row) + ')')
			# Append the calculated tile to the tiles list
			tiles.append({'col': col, 'row': row})
			debug('tile: (' + str(col) + ',' + str(row) + ')')

	return tiles

def	line_next_tile_col_row(col, row, x1, y1, x2, y2, toporbottom, rightorleft):
	"""
	This function calculates the next grid tile that a line passes through
    based on its slope and the current tile it is at.

	Args:
		col (int): The current column.
		row (int): The current row.
		x1 (float): The x-coordinate of the first point.
		y1 (float): The y-coordinate of the first point.
		x2 (float): The x-coordinate of the second point.
		y2 (float): The y-coordinate of the second point.
		toporbottom (str): Indicates whether the line is at the top or bottom of the tile.
		rightorleft (str): Indicates whether the line is at the right or left of the tile.

	Returns:
		list: A list containing the next column and row.

	"""
	debug('(col, row): (' + str(col) + ',' + str(row) + ') (x1,y1): (' + str(x1) + ',' + str(y1) + ') (x2,y2): (' + str(x2) + ',' + str(y2) + ') ' + toporbottom + ' ' + rightorleft)

	# Calculate the slope(m) and intercept(b) of the line following x = my + b
	m = (y2 - y1) / (x2 - x1)
	b = y1 -m * x1
	# Initialize the change in row and column to be 0
	drow = 0
	dcol = 0

	# Calculate the x-coordinates of the left and right boundaries of the tile
	leftside = col * config['grid_tile_width']
	rightside = (col + 1) * config['grid_tile_width']

	# Check the position of the line relative to the tile
    # and determine the next tile accordingly
	match toporbottom:
		# If the line is at the top of the tile
		case 'top':
			debug('test top')
			# calculate the x-coordinate of the line at the top boundary of the tile
			topy = (row + 1) * config['grid_tile_height']
			topx = round((topy - b) / m, rounding_decimal_places)

			debug({'topx': topx, 'topy': topy, 'leftside': leftside, 'rightside': rightside})

			# If the x-coordinate of the line at the top boundary of the tile
        	# is within the boundaries of the tile, the line passes through the top boundary,
        	# so the next tile is the one above
			if leftside <= topx and topx <= rightside:
				debug('is top')
				drow = 1

		# The logic here is similar to the 'top' case, but for the bottom boundary
		case 'bottom':
			debug('test bottom')
			bottomy = row * config['grid_tile_height']
			bottomx = round((bottomy - b) / m, rounding_decimal_places)

			debug({'bottomx': bottomx, 'bottomy': bottomy, 'leftside': leftside, 'rightside': rightside})

			if leftside <= bottomx and bottomx <= rightside:
				debug('bottom')
				drow = -1

	# Calculate the y-coordinates of the bottom and top boundaries of the tile
	bottomside 	= row * config['grid_tile_height']
	topside 	= (row + 1) * config['grid_tile_height']
	
	match rightorleft:
		# If the line is at the right of the tile
		case 'right':
			debug('test right')
			# calculate the y-coordinate of the line at the right boundary of the tile
			rightx = (col + 1) * config['grid_tile_width']
			righty = m * rightx + b

			debug({'rightx': rightx, 'righty': righty, 'bottomside': bottomside, 'topside': topside})

			# If the y-coordinate of the line at the right boundary of the tile
        	# is within the boundaries of the tile, the line passes through the right boundary,
        	# so the next tile is the one to the right
			if bottomside <= righty and righty <= topside:
				debug('right')
				dcol = 1
		# The logic here is similar to the 'right' case, but for the left boundary
		case 'left':
			debug('test left')
			leftx = (col) * config['grid_tile_width']
			lefty = m * leftx + b

			debug({'leftx': leftx, 'lefty': lefty, 'bottomside': bottomside, 'topside': topside})

			if bottomside <= lefty and lefty <= topside:
				debug('left')
				dcol = -1

	# Add the change in row and column to the current row and column
	row += drow
	col += dcol
	# Ensure that the row and column are within the grid boundaries
	row = max(row, 0)
	row = min(row, config['grid_rows'] - 1)
	col = max(col, 0)
	col = min(col, config['grid_cols'] - 1)

	return [col, row]

def delete_edges_at_tile(row, col):
	"""
	Deletes edges in the `timestep_graph` that are connected to the specified grid tile.

	Args:
		row (int): The row index of the tile.
		col (int): The column index of the tile.

	Returns:
		None
	"""

	# Declare the graph as a global variable
	global timestep_graph

	# Initialize an empty list to hold edges that need to be deleted
	edges_to_delete = []

	# Iterate over all edges in the graph
	for edge in timestep_graph.edges.items():
		edgeinfo = edge[0] # Edge information, contains node IDs
		edgedata = edge[1] # Edge data, contains tile details
		nodeid1 = edgeinfo[0] # Extract node ID 1
		nodeid2 = edgeinfo[1] # Extract node ID 2

		# Iterate over all tiles that the edge passes through
		for tile in edgedata['tiles']:
			# Check if the tile's indices match the provided indices
			if tile['row'] == row and tile['col'] == col:
				# If a match is found, add the edge to the deletion list
				edges_to_delete.append([nodeid1, nodeid2])
				break # Break the loop as we've found a match

	# Check if there are any edges to delete
	if len(edges_to_delete) > 0:
		# Pick a random edge to delete
		edgenode = random.choice(edges_to_delete)
		nodeid1 = edgenode[0]
		nodeid2 = edgenode[1]

		# Log a debug message with details of the edge to be deleted
		debug('delete edge at tile (' + str(col) + ',' + str(row) + '): nodeid1: ' + str(nodeid1) + ' nodeid2: ' + str(nodeid2))
		# Remove the edge from the graph
		timestep_graph.remove_edge(nodeid1, nodeid2)

def build_graph():
	"""
	Builds a network graph based on the configuration settings.

	Args:
		None
		
	Returns:
		None
	"""
	# Check if the web configuration contains two elements
	if len(config['web']) == 2:
		# If so, set the number of web rings and radials
		config['web_rings'] 	= config['web'][0]
		config['web_radials'] 	= config['web'][1]

		# Initialize node index 
		index = 0

		# Calculate the central node coordinates
		nodex = int(config['map_height'] / 2)
		nodey = int(config['map_width'] / 2)
		# Create a new node at the center
		new_node(index, nodex, nodey)

		# Iterate over each ring
		for ring in range(0, config['web_rings']):
			# Within each ring, iterate over each radial
			for radial in range(0, config['web_radials']):
				# Increment the node index
				index += 1
				# Calculate the angle for the current radial
				theta = 2 * math.pi * radial / config['web_radials']
				# Calculate the x and y coordinates for the new node
				nodex = int((config['map_width'] / 2) + (config['map_width'] / 2) * ((config['map_width'] + config['grid_tile_width'] / 2) / config['map_width']) * ((ring + 1) / (config['web_rings'] + 1)) * math.sin(theta))
				nodey = int((config['map_height'] / 2) + (config['map_height'] / 2) * ((config['map_width'] + config['grid_tile_height'] / 2) / config['map_width']) * ((ring + 1) / (config['web_rings'] + 1)) * math.cos(theta))
				# Create a new node at the calculated coordinates
				new_node(index, nodex, nodey)

		# Create edges between the nodes in each web ring
		for ring in range(0, config['web_rings']):
			for radial in range(0, config['web_radials'] - 1):
				n = ring * config['web_radials'] + radial + 1
				new_edge(n, n + 1)

			# Close the ring by connecting the last node to the first
			new_edge(n + 1, n - config['web_radials'] + 2)

		# web radials
		for ring in range(config['web_rings'] - 1):
			for radial in range(config['web_radials']):
				n1 = ring * config['web_radials'] + radial + 1
				n2 = n1 + config['web_radials']

				new_edge(n1, n2)

		# center radials
		for radial in range(1, config['web_radials'] + 1):
			new_edge(0, radial)

	# If debugging is enabled, print out some information about the network
	if config['debug']:
		debug("Network type: web")
		debug("Web rings: " + str(config['web_rings']))
		debug("Web radials: " + str(config['web_radials']))
		debug("Node count: " + str(len(network_graph.nodes())))

def calculate_score():
	"""
	Calculate the score for the network based on the shortest path lengths from a central node.

	This function uses the `nx.single_source_dijkstra_path_length` method from the NetworkX library
	to calculate the shortest path lengths from a central node to all other nodes in the network.
	The score is calculated by summing the reciprocal of the path lengths for each node, excluding
	nodes that are not specified in the `config['score_nodes']` list.

	The calculated score is stored in the `data` dictionary under the current `time` key. The maximum
	score encountered so far is also updated in the `scoremax` variable.

	Note: This function assumes that the variables `timestamp_graph`, `data`, `time`, `scoremax`,
	and `config` are already defined and accessible.

	Parameters:
		None

	Returns:
		None
	"""
	global timestamp_graph
	global data
	global time
	global scoremax

	# Iterate over each node in the graph and its shortest path length from the center node
	for key, value in nx.single_source_dijkstra_path_length(timestep_graph, centernodeid).items():
		# If there are no specific nodes to score or if the current node is in the list of nodes to score
		if len(config['score_nodes']) == 0 or key in config['score_nodes']:
			# If the path length to the current node is not zero
			if not value == 0:
				# Add the inverse of the path length to the score for the current time step
				data[time]['score'] += 1 / value
	# Update the maximum score seen so far if the current time step's score is higher
	scoremax = max(scoremax, data[time]['score'])

def create_timestep_graph():
	"""
	Creates a timestep graph by copying the network graph.

	The function creates a new graph called timestep_graph by making a copy of the network_graph.

	Parameters:
		None

	Returns:
		None
	"""
	# Declare the timestep_graph as a global variable
	global timestep_graph
	# Create a copy of the network_graph and assign it to timestep_graph
	timestep_graph = network_graph.copy()

def create_xarray_data():
	"""
	Create an xarray dataset with event data.

	Parameters:
		None

	Returns:
		ds (xr.Dataset): Xarray dataset containing event data.
	"""
	# Create the following dimensions
	print(config['event_sev'])
	print(config['event_weights'])
	T = np.arange(1, config['timesteps'] + 1)
	lat = np.arange(0, 100, 1.0)
	lon = np.arange(0, 100, 1.0)

	events_units = 'Scaler'
	lats_units = 'Degrees'
	lons_units = 'Degrees'
	Ts_units = 'Days'

	# Matrix of zeros to match lat,lon grid
	# Using this matrix to store the event severity levels
	Z = np.zeros((config['timesteps'], len(lat), len(lon)))
	Y = np.zeros((100, 100))
	
	print(f"Creating Xarray event data for {config['timesteps']} timesteps/events with a {len(lat)} x {len(lon)} grid.")

	# Iterates over the timesteps
	for i in range(config['timesteps']):
		# Matrix of zeros to match lat,lon grid
		X = np.zeros((100, 100))

		# Generate random coordinates equal to the timestep number
		coordinates = np.random.randint(0, 100, (i + 1 , 2))
		# Set these coordinates to be 1
		X[coordinates[:, 0], coordinates[:, 1]] = 1
		# Set the coordinates to be 1,3,5
		Y[coordinates[:, 0], coordinates[:, 1]] = random.choices(config['event_sev'], weights=config['event_weights'])[0]

		# Find all the coordinates of the non-zero values in the Y matrix
		non_zero_indices = np.nonzero(Y)
		# Set these coordinates in X to be 1
		X[non_zero_indices] = 1
		# Reduce the non-zero values by 1 each timestep
		Y[non_zero_indices] = Y[non_zero_indices] - 1

		# Save the values in the Z matrix
		Z[i, :, :] = X

	# Create an xarray dataset
	ds = xr.Dataset(
		{
			'events': (('time', 'lat', 'lon'), Z),
		},
		coords={
			'time': T,
			'lat': lat,
			'lon': lon,
		},
	)

	# Add attributes to variables
	ds['events'].attrs['units'] = events_units
	ds['lat'].attrs['units'] = lats_units
	ds['lon'].attrs['units'] = lons_units
	ds['time'].attrs['units'] = Ts_units

	print('Done...')

	return ds

def process_events():
	"""
	Process events and calculate scores for each timestep.

	This function loops through the events and calculates scores for each timestep based on the events data.
	It creates a timestep graph, deletes edges at each tile, and calculates the score for the timestep.
	The scores are stored in the 'data' dictionary.

	Args:
		None

	Returns:
		None
	"""
	global time
	global timestepevents

	# Call the create_xarray_data function to create an xarray dataset with event data
	ds = create_xarray_data()

	# Loop through each event in the xarray data
	for time, timevariable in enumerate(ds.events):
		# If a maximum generation limit is set and the current time exceeds it, break the loop
		if config['max_generations'] is not None and time >= config['max_generations'][0]:
			break
		# If a debug timestep is set and the current time is a multiple of it, print a debug message with the current time
		if config['debug_timestep'] is not None:
			if time % config['debug_timestep'] == 0:
				print(datetime.now().strftime("%Y-%m-%d %H:%M:%S timestep: " + str(time)))

		# Create a copy of the network graph for this timestep
		create_timestep_graph()
		# Log the current time for debugging
		debug("time: " + str(time))
		
		# Initialize a counter for the number of events at this timestep
		timestepevents = 0

		# Get the indices where the timevariable is not zero
		lat, lon = np.where(timevariable != 0)
		# For each of these indices
		for row, col in zip(lat, lon):
			# Increment the event counter
			timestepevents += 1
			# Delete edges connected to this index in the timestep graph
			delete_edges_at_tile(row.item(), col.item())

		# Store the time, number of events, and an initial score of 0 in the data dictionary
		data[time] = {'time': time, 'timestepevents': timestepevents, 'score': 0}

		# Calculate the score for this timestep
		calculate_score()

		# Log the number of events for debugging
		debug("timestepevents: " + str(timestepevents))

		# If a maximum generation limit is set and the current time exceeds it, break the loop
		if config['max_generations'] is not None and time >= config['max_generations'][0]:
			break

def create_output_directory():
	"""
	Creates the output directory based on the configuration settings.

	The output directory is determined by the 'output_directory' value in the configuration.
	If 'output_no_subdirectory' is False, a subdirectory with the current timestamp will be created.
	If the output directory does not exist, it will be created.

	Returns:
		None
	"""
	global timestamp
	global output_directory
	# Set the output directory path from the configuration	
	output_directory = config['output_directory']
	# If configuration does not exclude subdirectories, append a timestamp to the output directory path
	if not config['output_no_subdirectory']:
		# os.sep includes the correct separator for the current operating system
		output_directory += os.sep + timestamp
	# If the output directory does not exist yet
	if not os.path.isdir(output_directory):
		os.makedirs(output_directory) # Create the output directory

def log(severity, message):
	"""
	Logs the given message with the specified severity level.

	Parameters:
	- severity (str): The severity level of the log message.
	- message (str or dict): The message to be logged. If it's a dictionary, it will be converted to a string.

	Returns:
	None
	"""
	global output_directory

	# If the message is a dictionary, convert it to a string
	if type(message) is dict:
		message = str(message)
	# Construct the log line with current date-time, severity level, function name where log() was called, and message
	line = "[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "][" + severity + "][" + inspect.stack()[2].function + "] " + message + "\n"
	# Open the log file in append mode
	with open(output_directory + os.sep + config['log_filename'], "a") as f:
		# Write the log line to the log file
		f.write(line)
	# If debugging is enabled, also print the log line to the console
	if config['debug']:
		print(line, end="")

def trace(message):
	"""
	Logs a trace message if the 'trace' configuration option is enabled.

	Args:
		message (str): The message to be logged.
	"""
	if config['trace']:
		log('TRACE', message)

def debug(message):
	"""
	Logs a debug message if the 'debug' configuration is enabled.
	
	Args:
		message (str): The debug message to be logged.
	"""
	if config['debug']:
		log('DEBUG', message)

def info(message):
	"""
	Logs an informational message if the 'info' configuration is enabled.

	Args:
		message (str): The message to be logged.
	"""
	if config['info']:
		log('INFO ', message)

def warn(message):
	"""
	Logs a warning message if the 'warn' configuration flag is set.

	Args:
		message (str): The message to be logged.
	"""
	if config['warn']:
		log('WARN ', message)

def error(message):
	"""
	Logs an error message if the 'error' configuration is enabled.

	Args:
		message (str): The error message to be logged.
	"""
	if config['error']:
		log('ERROR', message)

def configure_args():
	"""
	Configure command-line arguments for the NetworkScore program.

	Returns:
		dict: A dictionary containing the parsed command-line arguments.
	"""
	global config

	parser = argparse.ArgumentParser(description="NetworkScore", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("map_height", type=int, help="map height", default=1000)
	parser.add_argument("map_width", type=int, help="map width", default=1000)
	parser.add_argument("grid_rows", type=int, help="grid rows", default=100)
	parser.add_argument("grid_cols", type=int, help="grid cols", default=100)
	parser.add_argument("-W", "--web", metavar=('RADIALS','RINGS'), type=int, nargs=2, help="web RADIALS RINGS")
	parser.add_argument("-R", "--random", metavar=('NODES','EDGES'), type=int, nargs=2, help="random NODES EDGES")
	parser.add_argument("-MG", "--max-generations", metavar=('GENERATION_COUNT'), nargs=1, type=int, help="max generations GENERATION_COUNT", default=None)
	parser.add_argument("-O", "--output", metavar=('FILENAME'), nargs='?', help="output file FILENAME", const='output.csv')
	parser.add_argument("-OD", "--output-directory", metavar=('DIRNAME'), nargs='?', help="output directory DIRNAME", default='output')
	parser.add_argument("-ONS", "--output-no-subdirectory", action='store_true', help="output no subdirectory", default=False)
	parser.add_argument("-DT", "--debug-timestep", metavar=('TIMESTEPS'), nargs='?', type=int, help="debug timestep TIMESTEPS", const=100, default=None)
	parser.add_argument("-TL", "--trace", action='store_true', help="trace", default=False)
	parser.add_argument("-DL", "--debug", action='store_true', help="debug", default=False)
	parser.add_argument("-IL", "--info", action='store_true', help="info", default=False)
	parser.add_argument("-WL", "--warn", action='store_true', help="warn", default=False)
	parser.add_argument("-EL", "--error", action='store_true', help="error", default=False)
	parser.add_argument("-LF", "--log-filename", metavar=('FILENAME'), nargs=1, help="log file FILENAME", default='output.log')
	parser.add_argument("-SN", "--score-nodes", nargs=1, help="score nodes NODES", default='')
	parser.add_argument("-TS", "--timesteps", type=int, help="data timestep value", default=1000)
	parser.add_argument("-ES", "--event-sev", nargs=1, help="Event severity", default='')
	parser.add_argument("-EW", "--event-weights", nargs=1, help="Event weights", default='')

	args = parser.parse_args()

	config = vars(args)

	config['grid_tile_height'] 	= config['map_height'] // config['grid_rows']
	config['grid_tile_width'] 	= config['map_width'] // config['grid_cols']
	config['score_nodes'] 		= list(map(int, config['score_nodes'][0].split(',')))
	config['event_sev'] 		= list(map(int, config['event_sev'][0].split(',')))
	config['event_weights'] 		= list(map(int, config['event_weights'][0].split(',')))

# output data to the output csv file
def output_csv():
	"""
	Write the data to a CSV file.

	This function writes the data stored in the `data` variable to a CSV file
	specified by the `output_directory` and `config['output']` values.

	If there is no data available, it will print a debug message and return
	without writing anything.

	If the `config['debug']` flag is set, it will also print the data in a
	formatted manner.

	Note: This function assumes that the `data` variable is a dictionary where
	the keys represent the field names and the values represent the corresponding
	data rows.

	Returns:
		None
	"""
	global output_directory
	global data

	filename = output_directory + os.sep + config['output']

	if len(data) == 0:
		debug('No data')
		return

	with open(filename, "w") as file:
		writer = csv.DictWriter(file, fieldnames=data[0])
		writer.writeheader()
		writer.writerows(data.values())

	if config['debug']:
		debug(str([[value for value in row.values()] for row in data.values()]))
	
def main():
	"""
	This is the main function of the NetworkScore program.
	It performs the following steps:
	1. Sets the global timestamp variable to the current date and time.
	2. Configures the program arguments.
	3. Creates the output directory.
	4. Builds the graph.
	5. Processes events.
	6. Outputs the result to a CSV file if specified in the configuration.
	"""
	global timestamp

	# note: timestamp variable is used for the name of the output directory, not to be confused with the timestep variable which is used during the processing of events in each timestep
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

	configure_args()

	create_output_directory()

	if config['debug']:
		debug(os.path.basename(__file__) + " Execution started")
		debug("Configuration: " + str(config))

	build_graph()

	process_events()

	if config['output'] is not None:
		output_csv()

if __name__ == '__main__':
    sys.exit(main())
    
