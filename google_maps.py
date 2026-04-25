# -*- coding: utf-8 -*-
"""
this file gets google maps data and makes a network representation out of it.
"""

import googlemaps
import numpy
import math
import json
import os

# setup
api_key = os.environ.get("GMAPS_API_KEY")
gmaps = googlemaps.Client(key=api_key)


# pick 10 gps coordinates within a r-mile radius "square" of some center
stl_center_latlon = (38.63368635748576, -90.2510896334572)

# 1deg lat = 69.17 miles
# 1deg long = 69.17 * cos(lat)

def get_random_coords(center, radius, number_to_select):
    coords = []
    lat_sc = 2 * radius/69.17
    lon_sc = 2 * radius/(69.17*math.cos(math.pi * center[0]/180))
    lat_min = center[0] - radius/69.17
    lon_min = center[1] - radius/(69.17*math.cos(math.pi * center[0]/180))
    
    scalings = numpy.random.rand(number_to_select, 2)
    
    for scale in scalings:
        sc_lat = float(scale[0])
        sc_lon = float(scale[1])
        coords.append((lat_min + (lat_sc * sc_lat), lon_min + (lon_sc * sc_lon)))
    
    return coords


def get_addresses(coords):
    addresses = []
    
    for coord in coords:
        rg_res = gmaps.reverse_geocode(coord)
        addresses.append(rg_res[0]['formatted_address'])
        
    return addresses


def save_address_set(addresses, filename):
    file = open(filename, 'w')
    for address in addresses:
        file.write(f'{address}\n')
        
    file.close()


def get_address_set(filename):
    file = open(filename, 'r')
    
    addresses_unclean = file.readlines()
    addresses = []
    
    for add in addresses_unclean:
        addresses.append(add.strip('\n'))
    
    file.close()
    return addresses


def make_travel_time_matrix(addresses):
    tt_matrix = [[0 for i in addresses] for j in addresses]
    directions_matrix = [[0 for i in addresses] for j in addresses]
    
    # this only works with a one-leg journey though it could? be modified for multi-leg journeys.
    for i in range(len(addresses)):
        for j in range(len(addresses)):
            if i != j:
                place1 = addresses[i]
                place2 = addresses[j]
                dirs_res = gmaps.directions(place1, place2, mode='driving')
                tt_matrix[i][j] = int(dirs_res[0]['legs'][0]['duration']['value'])
                directions_matrix[i][j] = dirs_res
     
    return tt_matrix, directions_matrix


def save_travel_time_matrix(tt_matrix, filename):
    file = open(filename, 'w')
    for row in tt_matrix:
        rtw = ''
        for col in row:
            rtw = rtw + str(col) + ', '
        
        file.write(f'{rtw[:-2]}'+ '\n')
        
    file.close()
    
    
def get_travel_time_matrix(filename):
    file = open(filename, 'r')
    rows_unclean = file.readlines()
    ttm = []
    
    for row in rows_unclean:
        row_clean = row.strip('\n')
        cols = row_clean.split(', ')
        int_cols = [int(c) for c in cols]
        ttm.append(int_cols)
        
    file.close()
    return ttm


def save_directions_matrix(directions_matrix, filename='directions_matrix.json'):
    with open(filename, 'w') as f:
        json.dump(directions_matrix, f)


def get_directions_matrix(filename='directions_matrix.json'):
    with open(filename, 'r') as f:
        return json.load(f)


def make_map_route(directions_matrix, route_plan):
    '''
    Parameters
    ----------
    directions_matrix : list(list)
        matrix of addresses to map.
    route_plan : networkx DiGraph
        a DiGraph with as many nodes as there are addresses and 
        one edge coming in and out of each node (enforced) to make the route with.

    Returns
    -------
    a.

    '''
    return None


def make_new_problem(rad, num_pts):
    
    cwd = os.getcwd()
    problem_number = 1
    path_exists = True
    path = f"{rad}m_{num_pts}_{problem_number}"
    
    while path_exists:
        path = f"{rad}m_{num_pts}_{problem_number}"
        
        if os.path.isdir(path):
            # directory exists
            problem_number = problem_number + 1
        else:
            # directory doesn't exist (make it)
            path_exists=False
            print(f"creating new problem at: {cwd}/{path}")
            os.mkdir(path)
            
            # get addresss set
            adds1 = get_addresses(get_random_coords(stl_center_latlon, rad, num_pts))
            save_address_set(adds1, f"{path}/address-set.txt")
            adds1 = get_address_set(f'{path}/address-set.txt')
            
            # make travel time and directions matrices
            tt_matrix_1, dirs_mat_1 = make_travel_time_matrix(adds1)
            
            # save travel times
            save_travel_time_matrix(tt_matrix_1, f'{path}/ttm.txt')
            
            # save directions matrix
            save_directions_matrix(dirs_mat_1, f'{path}/dir-mat.json')
            
            
    return True
            

if __name__ == 'main':
    # FYI this doesnt do anything you have to call make_new_problem() in the console.
    
    rad = 4 # radius in miles to sample from
    num_pts = 6 # number of points to sample
    if make_new_problem(rad, num_pts) == True:
        print("Problem Created")
    
    
    
    
    
    
    
    
    
    
    
    