from examples.cfd import plot_field, init_hat
import numpy as np

# Some variable declarations
nx = 50000
ny = 50000
nt  = 100
xmin = 0.
xmax = 2.
ymin = 0.
ymax = 1.

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

# Initialization
p  = np.zeros((nx, ny))
pd = np.zeros((nx, ny))
b  = np.zeros((nx, ny))

# Source
b[int(nx / 4), int(ny / 4)]  = 100
b[int(3 * nx / 4), int(3 * ny / 4)] = -100

for it in range(nt):
    pd = p.copy()
    p[1:-1,1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy**2 +
                    (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx**2 -
                    b[1:-1, 1:-1] * dx**2 * dy**2) / 
                    (2 * (dx**2 + dy**2)))

    p[0, :] = 0
    p[nx-1, :] = 0
    p[:, 0] = 0
    p[:, ny-1] = 0
    
plot_field(p, xmax=xmax, ymax=ymax, view=(30, 225))

from devito import Grid, Function, TimeFunction, Operator, configuration, Eq, solve

# Silence the runtime performance logging
configuration['log_level'] = 'ERROR'

# Now with Devito we will turn `p` into `TimeFunction` object 
# to make all the buffer switching implicit
grid = Grid(shape=(nx, ny), extent=(1., 2.))
p = Function(name='p', grid=grid, space_order=2)
pd = Function(name='pd', grid=grid, space_order=2)
p.data[:] = 0.
pd.data[:] = 0.

# Initialise the source term `b`
b = Function(name='b', grid=grid)
b.data[:] = 0.
b.data[int(nx / 4), int(ny / 4)]  = 100
b.data[int(3 * nx / 4), int(3 * ny / 4)] = -100

# Create Laplace equation base on `pd`
eq = Eq(pd.laplace, b, subdomain=grid.interior)
# Let SymPy solve for the central stencil point
stencil = solve(eq, pd)
# Now we let our stencil populate our second buffer `p`
eq_stencil = Eq(p, stencil)

# Create boundary condition expressions
x, y = grid.dimensions
t = grid.stepping_dim
bc = [Eq(p[x, 0], 0.)]
bc += [Eq(p[x, ny-1], 0.)]
bc += [Eq(p[0, y], 0.)]
bc += [Eq(p[nx-1, y], 0.)]
          
# Now we can build the operator that we need
op = Operator([eq_stencil] + bc)

# Run the outer loop explicitly in Python
for i in range(nt):
    # Determine buffer order
    if i % 2 == 0:
        _p = p
        _pd = pd
    else:
        _p = pd
        _pd = p

    # Apply operator
    op(p=_p, pd=_pd)
     
plot_field(p.data, xmax=xmax, ymax=ymax, view=(30, 225))
