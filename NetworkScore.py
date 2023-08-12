#! /usr/local/bin/python

"""
NetworkScore.py

This is a python program that builds a network (graph) of nodes on a plane and takes as input a netCDF file of events at the grid locations.
The edges of the graph that are in the grid locations are deleted.
Note, in graph theory, edges are the lines that connect two nodes.

There are many command line parameters, use the --help command line parameter to see a list of all of them
The configure_args function loads a global array called config with all of the command line parameters, which is used by the functions in this script

Example usage:
./NetworkScore.py --help

./NetworkScore.py 1000 1000 100 100 --web 8 9 --netcdf Event_severity.nc --max-generations 20 --save-chart --save-image --output --debug-timestep 1 --score-nodes "54,28,51"

Installation notes:

pip install netCDF4
pip install networkx[default]
pip install xarray
"""

import argparse
import csv
import inspect
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
import numpy as np
import os
import random
import sys
import xarray as xr

from datetime import datetime
from numpy import arange
from scipy.optimize import curve_fit

# Variables for Xarray data:
# Number of Timesteps:
Number_Of_Ticks = 3000
# Event severity:
event_lev = [1, 3, 5]
# Weights for each Event severity level:
# 1 - 60%, 3 - 30%, 5 - 10%
weights = [1.0, 0.0, 0.0]

config 			= None

imagefileindex  = 0
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

fig = None
ax = None

# nodeids are integers, the center nodeid is 0
centernodeid = 0

# add a new node to the network_graph
def new_node(nodeid, x, y):
	col = x // config['grid_tile_width']
	row = y // config['grid_tile_height']

	color = config['node_color']

	if nodeid in config['score_nodes']:
		color = config['score_node_color']

	if nodeid == 0:
		color = config['origin_node_color']

	network_graph.add_node(nodeid, x=x, y=y, row=row, col=col, id=nodeid, color=color, size=config['node_size'], label=nodeid)

# create a new edge between nodes
def new_edge(nodeid1, nodeid2):
	tiles = edge_tiles(nodeid1, nodeid2)
	network_graph.add_edge(nodeid1, nodeid2, "Edge_" + str(nodeid1) + "_" + str(nodeid2), color=config['edge_color'], width=config['edge_width'], tiles=tiles)

# create a new edge between nodes
def debug_nodes(graph):
	nodes = ' '.join(map(str, graph.nodes()))
	debug(nodes)

# determine which grid squares an edge between two nodes will pass through,
# used to remove the edge when an event occurs on that grid square
def edge_tiles(nodeid1, nodeid2):
	global time

	node1 = network_graph.nodes[nodeid1]
	node2 = network_graph.nodes[nodeid2]

	x1 = node1['x']
	y1 = node1['y']

	x2 = node2['x']
	y2 = node2['y']

	col1 = x1 // config['grid_tile_width']
	row1 = y1 // config['grid_tile_height']

	col2 = x2 // config['grid_tile_width']
	row2 = y2 // config['grid_tile_height']

	debug('node1: ' + str(nodeid1) + ' (' + str(col1) + ',' + str(row1) + ')')
	debug('node2: ' + str(nodeid2) + ' (' + str(col2) + ',' + str(row2) + ')')

	tiles = []
	tiles.append({'col': col1, 'row': row1})

	if not (col1 == col2 and row1 == row2):
		tiles.append({'col': col2, 'row': row2})
	else:
		return tiles

	noslope 	= (col1 == col2)
	zeroslope 	= (row1 == row2)

	if noslope:
		# edge line is vertical

		col = col1

		if row1 < row2:
			drow = 1
			for row in range(row1 + drow, min(row2, config['grid_rows']), drow):
				tiles.append({'col': col, 'row': row})
		else:
			drow = -1
			for row in range(row1 + drow, max(row2, 0), drow):
				tiles.append({'col': col, 'row': row})

	elif zeroslope:
		debug('zeroslope')
		# edge line is horizontal
		row = row1

		if col1 < col2:
			dcol = 1
			for col in range(col1 + dcol, min(col2, config['grid_cols']), dcol):
				tiles.append({'col': col, 'row': row})
		else:
			dcol = -1
			for col in range(col1 + dcol, max(col2, 0), dcol):
				tiles.append({'col': col, 'row': row})

	else:
		col = col1
		row = row1

		while not (col == col2 and row == row2):
			debug('(col,row): (' + str(col) + ',' + str(row) + ')')
			debug('(col1,row1): (' + str(col1) + ',' + str(row1) + ')')
			debug('(col2,row2): (' + str(col2) + ',' + str(row2) + ')')

			if x1 < x2:
				if y1 < y2:
					col, row = line_next_tile_col_row(col, row, x1, y1, x2, y2, 'top', 'right')
				else:
					col, row = line_next_tile_col_row(col, row, x1, y1, x2, y2, 'bottom', 'right')
			else:
				if y1 < y2:
					col, row = line_next_tile_col_row(col, row, x1, y1, x2, y2, 'top', 'left')
				else:
					col, row = line_next_tile_col_row(col, row, x1, y1, x2, y2, 'bottom', 'left')

			debug('(col,row): (' + str(col) + ',' + str(row) + ')')

			tiles.append({'col': col, 'row': row})

			debug('tile: (' + str(col) + ',' + str(row) + ')')

	return tiles

# used by the edge_tiles function, to determine the next square that the edge passes through
def	line_next_tile_col_row(col, row, x1, y1, x2, y2, toporbottom, rightorleft):
	debug('(col, row): (' + str(col) + ',' + str(row) + ') (x1,y1): (' + str(x1) + ',' + str(y1) + ') (x2,y2): (' + str(x2) + ',' + str(y2) + ') ' + toporbottom + ' ' + rightorleft)

	m = (y2 - y1) / (x2 - x1)
	b = y1 -m * x1

	drow = 0
	dcol = 0

	match toporbottom:
		case 'top':
			debug('test top')
			topy = (row + 1) * config['grid_tile_height']
			topx = round((topy - b) / m, rounding_decimal_places)

			leftside = col * config['grid_tile_width']
			rightside = (col + 1) * config['grid_tile_width']

			debug({'topx': topx, 'topy': topy, 'leftside': leftside, 'rightside': rightside})

			if leftside <= topx and topx <= rightside:
				debug('is top')
				drow = 1

		case 'bottom':
			debug('test bottom')
			bottomy = row * config['grid_tile_height']
			bottomx = round((bottomy - b) / m, rounding_decimal_places)

			leftside = col * config['grid_tile_width']
			rightside = (col + 1) * config['grid_tile_width']

			debug({'bottomx': bottomx, 'bottomy': bottomy, 'leftside': leftside, 'rightside': rightside})

			if leftside <= bottomx and bottomx <= rightside:
				debug('bottom')
				drow = -1

	match rightorleft:
		case 'right':
			debug('test right')
			rightx = (col + 1) * config['grid_tile_width']
			righty = m * rightx + b

			bottomside 	= row * config['grid_tile_height']
			topside 	= (row + 1) * config['grid_tile_height']

			debug({'rightx': rightx, 'righty': righty, 'bottomside': bottomside, 'topside': topside})

			if bottomside <= righty and righty <= topside:
				debug('right')
				dcol = 1

		case 'left':
			debug('test left')
			leftx = (col) * config['grid_tile_width']
			lefty = m * leftx + b

			bottomside 	= row * config['grid_tile_height']
			topside 	= (row + 1) * config['grid_tile_height']

			debug({'leftx': leftx, 'lefty': lefty, 'bottomside': bottomside, 'topside': topside})

			if bottomside <= lefty and lefty <= topside:
				debug('left')
				dcol = -1

	row += drow
	col += dcol

	row = max(row, 0)
	row = min(row, config['grid_rows'] - 1)

	col = max(col, 0)
	col = min(col, config['grid_cols'] - 1)

	return [col, row]

#	calculate which tiles that edge is part of
	
#	for row in range(node1.row):


#def debug_edges():
#	edgeslist = ''
#	for nodeid in G.nodes():
#		edges = ','.join(G.edges())
##	debug(nodes)

# loop through all edges and delete edges at tile
def delete_edges_at_tile(row, col):
	global timestep_graph

	edges_to_delete = []

	for edge in timestep_graph.edges.items():
		edgeinfo = edge[0]
		edgedata = edge[1]
		nodeid1 = edgeinfo[0]
		nodeid2 = edgeinfo[1]

		for tile in edgedata['tiles']:
			if tile['row'] == row and tile['col'] == col:
				edges_to_delete.append([nodeid1, nodeid2])
				break

#	for edgenodes in edges_to_delete:
#		nodeid1 = edgenodes[0]		
#		nodeid2 = edgenodes[1]		
#
#		debug('delete edge at tile (' + str(col) + ',' + str(row) + '): nodeid1: ' + str(nodeid1) + ' nodeid2: ' + str(nodeid2))
#		timestep_graph.remove_edge(nodeid1, nodeid2)

	if len(edges_to_delete) > 0:
		# delete random edge
		edgenode = random.choice(edges_to_delete)
		nodeid1 = edgenode[0]
		nodeid2 = edgenode[1]

		debug('delete edge at tile (' + str(col) + ',' + str(row) + '): nodeid1: ' + str(nodeid1) + ' nodeid2: ' + str(nodeid2))
		timestep_graph.remove_edge(nodeid1, nodeid2)

# delete node at grid square row, col
def delete_node_at_tile(row, col):
	global timestep_graph

	nodes = timestep_graph.nodes(data=True)
	nodestodelete = []
	for [nodeid, nodedata] in nodes:
		if nodedata['row'] == row and nodedata['col'] == col:
			if nodeid is not centernodeid:
				nodestodelete.append(nodeid)

	for nodeid in nodestodelete:
		debug("delete node: " + str(nodeid))
		timestep_graph.remove_node(nodeid)

	if len(nodestodelete) > 0:
		debug_nodes(timestep_graph)

# place an image at a grid square
def place_image_at_tile(row, col):
	if config['save_images']:
		img = mpimg.imread(config['eventicon_filename'])
		x = col * config['grid_tile_width']
		y = row * config['grid_tile_height']
		s = str(col) + ',' + str(row)
		ax.imshow(img, extent=(x, x + config['grid_tile_width'], y, y + config['grid_tile_height']))
		draw_text(x + config['grid_tile_width'] / 2 - 20, y + config['grid_tile_height'] / 2 + 10, s, color='yellow')

# write string s text at x, y position
def draw_text(x, y, s, **fontdict):
	plt.text(x, y, s, fontdict)

# prepare matplotlib plot of graph
def prepare_plot():
	global fig, ax

	plt.rcParams.update({'font.size': config['font_size'], 'text.color': config['font_color']})
	plt.rcParams["figure.figsize"] = [config['map_height'] // 200, config['map_width'] // 200]
	plt.rcParams["figure.dpi"] = config['dpi']
	plt.rcParams["figure.autolayout"] = False

	fig = plt.figure(0, figsize=(config['map_height'] // 200, config['map_width'] // 200))
#	fig, ax = plt.subplots(figsize=(config['map_height'] // 200, config['map_width'] // 200))
	ax = plt.gca()

	ax.set_facecolor(config['background_color'])
	ax.axis('off')
	ax.set_aspect('equal')
	ax.use_sticky_edges = False
	ax.set_xlim([0, config['map_width'] + 1])
	ax.set_ylim([0, config['map_height'] + 1])

	if config['show_grid']:
		ax.vlines(list(range(0, config['map_width'] + 1, config['grid_tile_width'])), 0, config['map_height'], linestyles='dashed', colors=str(config['grid_color']), linewidth=0.4)
		ax.hlines(list(range(0, config['map_height'] + 1, config['grid_tile_height'])), 0, config['map_width'], linestyles='dashed', colors=str(config['grid_color']), linewidth=0.4)

		for row in range(0, config['map_height'] // config['grid_tile_height']):
			for col in range(0, config['map_width'] // config['grid_tile_width']):
				x = col * config['grid_tile_width'] + config['grid_tile_width'] / 2 - 20
				y = row * config['grid_tile_height'] + config['grid_tile_width'] - 20
				s = str(col) + ',' + str(row)
				draw_text(x, y, s, color=config['grid_color'])

	fig.set_facecolor(config['background_color'])

# clear plot
def clear_plot():
#	plt.clf()
#	matplotlib.pyplot.close()

	plt.close()
#	plt.clf()

# used to determine line of best fit
# define the true objective function
def objective(x, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0):
	return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

# plot chart, this is a chart of the score of the graph for each timestep, and a line of best fit
def plot_chart():
	global scoremax
	scale = 1

	if not scoremax == 0:
		scale = scoremax

	plt.close()
	plt.rcParams.update({'font.size': config['font_size'], 'text.color': config['font_color']})
	plt.rcParams["figure.figsize"] = [config['map_height'] // 200, config['map_width'] // 200]
	plt.rcParams["figure.dpi"] = config['dpi']
	plt.rcParams["figure.autolayout"] = False

	plt.figure(0, figsize=(config['map_height'] // 200, config['map_width'] // 200))
	fig, ax = plt.subplots()
	ax.margins(0, 0)
	ax.axis('on')
	ax.set_facecolor('white')
	xpoints = [row['timestepevents'] for row in data.values()]
	ypoints = [row['score'] / scale for row in data.values()]
	plt.scatter(xpoints, ypoints, s=3)
	plt.ylim(ymax = scoremax / scale, ymin = 0.0)
	if len(xpoints) >= 10:
		popt, _ = curve_fit(objective, xpoints, ypoints)
		a, b, c, d, e, f = popt
		x_line = arange(min(xpoints), max(xpoints), 1)
		y_line = objective(x_line, a, b, c, d, e, f)
		plt.plot(x_line, y_line, '--', color='red')
	plt.xlabel(config['chart_label_x'])
	plt.ylabel(config['chart_label_y'])
	plt.title(config['chart_title'], color='black')
	plt.grid()
#	plt.show()

# plot graph
def plot_graph():
	global timestep_graph
	global timestepevents

	pos				= {}
	labels			= {}
	colors 			= []
	sizes			= []
	edge_colors		= []
	widths			= []

	# plot nodes
	for node in timestep_graph.nodes.items():
		nodedata = node[1]
		pos[nodedata['id']] 	= [nodedata['x'], nodedata['y']]
		labels[nodedata['id']] 	= nodedata['label']
		colors.append(nodedata['color'])
		sizes.append(nodedata['size'])
		draw_text(nodedata['x'] - 15, nodedata['y'] - config['node_size'], str(nodedata['col']) + ',' + str(nodedata['row']), color='white')

	# plot edges
	for edge in timestep_graph.edges.items():
		edgedata = edge[1]
		edge_colors.append(edgedata['color'])
		widths.append(edgedata['width'])

	if config['show_statistics']:
		statistics = ["Web radials: " + str(config['web_radials']),
						"Web rings: " + str(config['web_rings']),
						"Time: " + str(time),
						"Events: " + str(timestepevents),
						"Nodes: " + str(len(timestep_graph.nodes)),
						"Edges: " + str(len(timestep_graph.edges)),
						"Score: " + str(round(data[time]['score'],1))]

		draw_text(0, 1000, '\n'.join(statistics), color='white')

	nx.draw_networkx(timestep_graph, pos, labels=labels, node_color=colors, node_size=sizes, edge_color=edge_colors, font_size=config['font_size'], font_color=config['font_color'], width=widths)

# build the graph based on configuration parameters
# currently only the web type of network is supported
def build_graph():
	if len(config['web']) == 2:
		config['web_rings'] 	= config['web'][0]
		config['web_radials'] 	= config['web'][1]

		index = 0

		nodex = int(config['map_height'] / 2)
		nodey = int(config['map_width'] / 2)
		new_node(index, nodex, nodey)

		for ring in range(0, config['web_rings']):
			for radial in range(0, config['web_radials']):
				index += 1

				theta = 2 * math.pi * radial / config['web_radials']

				nodex = int((config['map_width'] / 2) + (config['map_width'] / 2) * ((config['map_width'] + config['grid_tile_width'] / 2) / config['map_width']) * ((ring + 1) / (config['web_rings'] + 1)) * math.sin(theta))
				nodey = int((config['map_height'] / 2) + (config['map_height'] / 2) * ((config['map_width'] + config['grid_tile_height'] / 2) / config['map_width']) * ((ring + 1) / (config['web_rings'] + 1)) * math.cos(theta))

				new_node(index, nodex, nodey)

		# web ring edges
		for ring in range(0, config['web_rings']):
			for radial in range(0, config['web_radials'] - 1):
				n = ring * config['web_radials'] + radial + 1
				new_edge(n, n + 1)

			# close the ring
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

	if config['debug']:
		debug("Network type: web")
		debug("Web rings: " + str(config['web_rings']))
		debug("Web radials: " + str(config['web_radials']))
		debug("Node count: " + str(len(network_graph.nodes())))

# output a image of the graph to a file
def save_plot_image():
	global timestep_graph
	global output_directory
	global imagefileindex
	global timestamp

	filename = output_directory + os.sep + config['filename'] + str(imagefileindex) + ".png"
	plt.savefig(filename, dpi=config['dpi'])
	debug('Saved image: ' + filename)
	imagefileindex += 1

# output a image of the chart to a file
def save_chart_image():
	global output_directory

	filename = output_directory + os.sep
	if isinstance(config['chart_filename'], list):
		filename += config['chart_filename'][0]
	else:
		filename += config['chart_filename']
	plt.savefig(filename, dpi=config['dpi'])
	debug('Saved image: ' + filename)

# calculate score for the network
def calculate_score():
	global timestamp_graph
	global data
	global time
	global scoremax

	for key, value in nx.single_source_dijkstra_path_length(timestep_graph, centernodeid).items():
		if len(config['score_nodes']) == 0 or key in config['score_nodes']:
			if not value == 0:
				data[time]['score'] += 1 / value

	scoremax = max(scoremax, data[time]['score'])

# create a graph based on the network_graph for this timestep
def create_timestep_graph():
	global timestep_graph
	timestep_graph = network_graph.copy()

# used by the edge_tiles function, to determine the next square that the edge passes through
def create_xarray_data():
	# Create the following dimensions
	T = np.arange(1, Number_Of_Ticks + 1)
	lat = np.arange(0, 100, 1.0)
	lon = np.arange(0, 100, 1.0)

	events_units = 'Scaler'
	lats_units = 'Degrees'
	lons_units = 'Degrees'
	Ts_units = 'Days'

	# Matrix of zeros to match lat,lon grid
	# Using this matrix to store the event severity levels
	Z = np.zeros((Number_Of_Ticks, len(lat), len(lon)))
	Y = np.zeros((100, 100))

	print(f'Creating Xarray event data for {Number_Of_Ticks} timesteps/events with a {len(lat)} x {len(lon)} grid.')

	# Iterates over the timesteps
	for i in range(Number_Of_Ticks):
		# Matrix of zeros to match lat,lon grid
		X = np.zeros((100, 100))

		# Generate random coordinates equal to the timestep number
		coordinates = np.random.randint(0, 100, (i + 1 , 2))
		# Set these coordinates to be 1
		X[coordinates[:, 0], coordinates[:, 1]] = 1
		# Set the coordinates to be 1,3,5
		Y[coordinates[:, 0], coordinates[:, 1]] = random.choices(event_lev, weights=weights)[0]

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
# process the events netcdf file
def process_events_netcdf():
	global time
	global timestepevents

	ds = create_xarray_data()

	# loop through events
	for time, timevariable in enumerate(ds.events):
		if config['max_generations'] is not None and time >= config['max_generations'][0]:
			break

		if config['debug_timestep'] is not None:
			if time % config['debug_timestep'] == 0:
				print(datetime.now().strftime("%Y-%m-%d %H:%M:%S timestep: " + str(time)))

		create_timestep_graph()
		debug("time: " + str(time))
		
		timestepevents = 0

		if config['save_images']:
			clear_plot()
			prepare_plot()

		lat, lon = np.where(timevariable != 0)
		for row, col in zip(lat, lon):
			timestepevents += 1
			#delete_node_at_tile(row.item(), col.item())
			delete_edges_at_tile(row.item(), col.item())
			place_image_at_tile(row.item(), col.item())

		data[time] = {'time': time, 'timestepevents': timestepevents, 'score': 0}

		calculate_score()

		debug("timestepevents: " + str(timestepevents))

		if config['save_images']:
			plot_graph()
			save_plot_image()

		if config['max_generations'] is not None and time >= config['max_generations'][0]:
			break

# generate video of the graph images using ffmpeg
def video():
	global output_directory
	global time

	rate = config['video_length']
	if not time == 0:
		rate = time / config['video_length']

	chartcommand = ""
	chartfiltercomplex = ""
	if config['save_chart']:
		chartcommand = " -r 30 -loop 1 -t 5 -i " + output_directory + os.sep + config['chart_filename'] + "  "
		chartfiltercomplex = " -filter_complex '[0:v] [1:v] concat=n=2:v=0:v=1 [v1]' -map \"[v1]\" "

	command = "ffmpeg -hide_banner -loglevel error " + chartcommand + " -start_number 0 -r " + str(round(rate, 4)) + " -i " + output_directory + os.sep + config['filename'] + "%1d.png " + chartfiltercomplex + " -c:v libx264 -r 30 -pix_fmt yuv420p -y " + " " + output_directory + os.sep + config['video']
	debug(command)
	os.system(command)

# create output directory
def create_output_directory():
	global timestamp
	global output_directory

	output_directory = config['output_directory']

	if not config['output_no_subdirectory']:
		output_directory += os.sep + timestamp

	if not os.path.isdir(output_directory):
		os.makedirs(output_directory)

# log message with severity
def log(severity, message):
	global output_directory

	if type(message) is dict:
		message = str(message)

	line = "[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "][" + severity + "][" + inspect.stack()[2].function + "] " + message + "\n"

	with open(output_directory + os.sep + config['log_filename'], "a") as f:
		f.write(line)

	if config['debug']:
		print(line, end="")

# log trace message to output
def trace(message):
	if config['trace']:
		log('TRACE', message)

# log debug message to output
def debug(message):
	if config['debug']:
		log('DEBUG', message)

# log info message to output
def info(message):
	if config['info']:
		log('INFO ', message)

# log warning message to output
def warn(message):
	if config['warn']:
		log('WARN ', message)

# log error message to output
def error(message):
	if config['error']:
		log('ERROR', message)

# load values into the global config array, based on command line parameters
def configure_args():
	global config

	parser = argparse.ArgumentParser(description="NetworkScore", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("map_height", type=int, help="map height", default=1000)
	parser.add_argument("map_width", type=int, help="map width", default=1000)
	parser.add_argument("grid_rows", type=int, help="grid rows", default=100)
	parser.add_argument("grid_cols", type=int, help="grid cols", default=100)
	parser.add_argument("-W", "--web", metavar=('RADIALS','RINGS'), type=int, nargs=2, help="web RADIALS RINGS")
	parser.add_argument("-R", "--random", metavar=('NODES','EDGES'), type=int, nargs=2, help="random NODES EDGES")
	parser.add_argument("-NC", "--node-color", nargs=1, help="node color", default='blue')
	parser.add_argument("-SNC", "--score-node-color", nargs=1, help="score node color", default='green')
	parser.add_argument("-ONC", "--origin-node-color", nargs=1, help="origin node color", default='red')
	parser.add_argument("-NS", "--node-size", nargs=1, type=int, help="node size", default=30)
	parser.add_argument("-EC", "--edge-color", nargs=1, help="edge color", default='blue')
	parser.add_argument("-EW", "--edge-width", nargs=1, type=int, help="edge width", default=1.5)
	parser.add_argument("-FC", "--font-color", nargs=1, help="font color", default='white')
	parser.add_argument("-FS", "--font-size", nargs=1, type=int, help="font size", default=5)
	parser.add_argument("-BC", "--background-color", nargs=1, help="background color", default='black')
	parser.add_argument("-SG", "--show-grid", action='store_true', help="show grid", default=False)
	parser.add_argument("-GC", "--grid-color", nargs=1, help="grid color", default='darkred')
	parser.add_argument("-SS", "--show-statistics", action='store_true', help="show statistics", default=False)
	parser.add_argument("-EIF", "--eventicon-filename", metavar=('EVENT_ICON_FILENAME'), nargs=1, help="event icon filename", default="event.png")
	parser.add_argument("-SI", "--save-images", action='store_true', help="save images", default=False)
	parser.add_argument("-FP", "--filepath", metavar=('FILEPATH'), nargs=1, help="filepath", default='./')
	parser.add_argument("-FN", "--filename", metavar=('FILENAME'), nargs=1, help="filename", default='image')
	parser.add_argument("-DPI", "--dpi", metavar=('DPI'), nargs=1, help="DPI", default=300)
	parser.add_argument("-MG", "--max-generations", metavar=('GENERATION_COUNT'), nargs=1, type=int, help="max generations GENERATION_COUNT", default=None)
	parser.add_argument("-V", "--video", metavar=('FILENAME'), nargs='?', help="video output FILENAME", const='video.mp4', default=None)
	parser.add_argument("-VL", "--video-length", metavar=('SECONDS'), nargs='?', type=int, help="video length SECONDS", default=30)
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
	parser.add_argument("-CDF", "--netcdf", metavar=('FILENAME'), nargs='?', type=str, help="netCDF FILENAME", default=None)
	parser.add_argument("-SC", "--save-chart", action='store_true', help="save chart", default=False)
	parser.add_argument("-CSS", "--chart-screen-seconds", nargs=1, type=int, help="chart screen seconds", default=1)
	parser.add_argument("-CFN", "--chart-filename", metavar=('FILENAME'), nargs=1, type=str, help="chart filename FILENAME", default='chart.png')
	parser.add_argument("-CLX", "--chart-label-x", nargs=1, help="chart label x LABEL", default='Events per timestep')
	parser.add_argument("-CLY", "--chart-label-y", nargs=1, help="chart label y LABEL", default='Network efficiency')
	parser.add_argument("-CT", "--chart-title", nargs=1, help="chart title TITLE", default='Network efficiency vs Events per timestep')
	parser.add_argument("-SN", "--score-nodes", nargs=1, help="score nodes NODES", default='')

	args = parser.parse_args()

	config = vars(args)

	config['grid_tile_height'] 	= config['map_height'] // config['grid_rows']
	config['grid_tile_width'] 	= config['map_width'] // config['grid_cols']
	config['score_nodes'] 		= list(map(int, config['score_nodes'][0].split(',')))

# output data to the output csv file
def output_csv():
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

# main function		
def main():
	global timestamp

	# note: timestamp variable is used for the name of the output directory, not to be confused with the timestep variable which is used during the processing of events in each timestep
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

	configure_args()

	create_output_directory()

	if config['debug']:
		debug(os.path.basename(__file__) + " Execution started")
		debug("Configuration: " + str(config))

	build_graph()

	process_events_netcdf()

	if config['output'] is not None:
		output_csv()

	if config['save_chart']:
		plot_chart()
		save_chart_image()


	if config['video'] is not None:
		video()


if __name__ == '__main__':
    sys.exit(main())
    
