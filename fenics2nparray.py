from fenics import *
import numpy as np

def fenics2nparray(data, boundary_value, r_hole, x, y, hole_coor):
    """
    Writes fenics solution to numpy array.

    Arguments: 
        data (fenics solution): input data 
        boundary_value (array): if point from numpy array not found on fenics mesh, 
            value is set to boundary_value
        x (array): x values of numpy array
        y (array): y values of numpy array  

    Returns: 
        data_grid (array): output data
        r (array): sqrt(X**2 + Y**2)
    """
    Nx = len(y)
    Ny = len(x)	
    X,Y = np.meshgrid(x,y)
    data_grid = np.zeros([Nx, Ny])

    r = np.sqrt((X-hole_coor[0])**2 + (Y-hole_coor[1])**2)

    for i in range(Nx):
        for j in range(Ny):
            x_val = X[i,j]
            y_val = Y[i,j]
            point = Point(x_val, y_val)
            try:
                data_val = data(point)
                data_grid[i,j] = data_val
            except:
                data_grid[i,j] = boundary_value

    return data_grid, r
