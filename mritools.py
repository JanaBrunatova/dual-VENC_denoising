from dolfin import *
import dolfin
import numpy as np
from ufl import j
import pywt

def SpatialInterpolation(u_list, V_in, V_out, V_aux = None, direction = None):
    '''
    interpolates the input data into a different mesh.
    Arguments:
        u_list: list of Fenics functions containing input data
        V_in: FunctionSpace or VectorFunctionSpace that u_list functions live in
        V_out: FunctionSpace or VectorFunctionSpace of same elements and order on new mesh
        V_aux: auxiliary VectorFunctionSpace needed if interpolating vector data to scalar data
        direction: selects which of the vector components to interpolate to scalar data
    Returns:
        list of Fenics functions on new mesh
    '''
    if V_in.mesh().ufl_cell() == hexahedron:
        raise Exception('hexahedra interpolation is not supported')
    LI = LagrangeInterpolator
    
    ndim = V_in.mesh().topology().dim()
    u_meas_list = [Function(V_out, name = 'measurement') for u in u_list]
    u_meas_aux = []
    if V_aux:
        u_meas_aux = Function(V_aux)
        comp_assigner = FunctionAssigner([V_out]*ndim, V_aux)


    for u, u_meas in zip(u_list, u_meas_list):
        if u_meas_aux:
            if not direction:
                raise Exception('no direction for scalar interpolation given')
            if direction.count(0) == 2 and direction.count(1) ==1:
                LI.interpolate(u_meas, u.sub(direction.index(1)))
            else:
                assert u_meas.value_shape() == []
                #normalize projection direction
                module = np.sqrt(np.dot(direction, direction))
                direction_norm = [elem / module for elem in direction]
                
                #info('- direction: {}'.format(direction))

                LI.interpolate(u_meas_aux, u)

                u_i = [u_meas] + [u_meas.copy(True) for j in range(ndim -1)]

                comp_assigner.assign(u_i, u_meas_aux)
                u_meas.vector()[:] *= direction_norm[0]
                for ui, d in zip(u_i[1:], direction_norm[1:]):
                    if d:
                        u_meas.vector().axpy(d, ui.vector())

        else:
            #info('- Full velocity vector')
            LI.interpolate(u_meas, u)
    #info('done!')
    return u_meas_list

def TemporalUndersampling(u_list, rate, timestamps = None):
    '''
    undersamples the input data by rate
    Arguments:
        u_list: list of Fenics functions
        rate: integer
        timestamps: timestamps associated to the entries in u_list
    Returns:
        subsampled u_list
        subsampled timestamps
    '''
    info('undersampling in time with rate {}'.format(rate))
    if timestamps:
        return u_list[::rate], timestamps[::rate]
    else:
        return u_list[::rate]

def TemporalInterpolation(u_list, dt_old, dt_new, timestamps = None):
    '''
    interpolates the input data from the old into the new timestep
    Arguments:
        u_list: list of Fenics functions
        dt_old: double, timestep of u_list
        dt_new: double, desired timestep
        timestamps: timestamps associated to the entries in u_list
    Returns:
        interpolated u_list
        optionally: new timestamps
    '''
    #assumption: dt_old is the timestep of u_list
    Tf = dt_old*(len(u_list))

    N_new = np.int(Tf/dt_new+1)
    times_old = np.linspace(0, Tf, len(u_list))
    times_new = np.linspace(0, Tf, N_new)
    
    u_vec = [u.vector().get_local() for u in u_list]
    u_vec_new = [np.zeros(u_vec[0].size) for i in range(N_new)]

    velnodes = np.zeros(times_old.size)
    velnodes_new = np.zeros(times_new.size)

    from scipy.interpolate import interp1d
    for el in range(len(u_vec[0])): #iterating over all nodes
        for t in range(len(velnodes)): #iterating over original timesteps
            velnodes[t] = u_vec[t][el]
        inter_f = interp1d(times_old, velnodes, kind='cubic')
        velnodes_new = inter_f(times_new)
        for t in range(len(velnodes_new)):
            u_vec_new[t][el] = velnodes_new[t]
    
    u_list_new = []
    for t in range(len(velnodes_new)):
        u_new = Function(u_list[0].function_space())
        u_new.vector()[:] = u_vec_new[t][:]
        u_list_new.append(u_new)

    if timestamps:
        return u_list_new, times_new
    else:
        return u_list_new

def get_VENC(u_list, phase_contrast):
    '''
    computes the VENC as the percentage of Vmax given by phase_contrast
    Arguments:
        u_list: list of Fenics functions
        phase_contrast: integer
    Returns:
        VENC: float
    '''
    umax_vect = []
    for u in u_list:
        umax_vect.append(np.max(np.abs(u.vector()[:])))

    # VENC = np.floor(np.ceil(np.max(umax_vect))*phase_contrast/100)
    VENC = round(np.max(umax_vect)*phase_contrast/100, 2)
    info('VENC picked: {}'.format(VENC))
    return VENC

def get_std(signal,SNR):
    '''  Compute standard deviation from a given SNR

    Args:
        signal: numpy array
        SNR:    double with the SNR. 'inf' means no noise. 

    Returns:
        noise_std:       double
    '''
    if SNR=='inf':
        return 0
    else:
        Psignal = np.abs(signal)**2
        Psignal_av = np.mean(Psignal)
        Psignal_av_db = 10*np.log10(Psignal_av)
        Pnoise_av_db = Psignal_av_db - SNR
        Pnoise_av = 10**(Pnoise_av_db/10)
        noise_std = np.sqrt(Pnoise_av)
        info(' - standard deviation {}'.format(noise_std))
        return noise_std

def vel_from_mag(V, M1_list, M2_list, VENC):
    '''
    computes velocity from magnetization
    Args:
        V: intended function space; has to match magnetization function space
        M1_list: list of fenics functions containing background magnetization
        M2_list: list of fenics functions containing contrast magnetization
        VENC: integer
    Returns:
        u_list: list of Fenics functions
    '''
    info('computing velocity from magnetization...')
    info(' - VENC: {}'.format(VENC))
    u_list = [Function(V) for m in M1_list]
    for (M1, M2, u) in zip(M1_list, M2_list, u_list):
        #compute velocity for each entry
        u_vec = VENC/np.pi*np.angle(M2/M1)
        u.vector()[:] = u_vec
    info('done!')
    return u_list

def magnetization(u_list, VENC, phys_pars = None):
    '''
    computes magnetization from velocity input
    Args:
        u_list: list of Fenics functions containing velocity input
        VENC: float
        phys_pars: dictionary, optional; physical parameters for the scanner (gamma, TE, B0). If not set, defaults to standard values
    Returns:
        M1_list, M2_list: lists of numpy arrays containing complex magnetization
        Mmod_list: list of Fenics functions containing the module of the magnetization
    '''
    info('computing magnetization from velocity...')
    #get physical parameters
    if not phys_pars:
        gamma = 1
        B0 =   1.5 #  Tesla
        TE =   5e-3 #  Echo-time
        info( ' - using default physical parameters')
    else:
        gamma = phys_pars['gamma']
        B0 = phys_pars['B0']
        TE = phys_pars['TE']
        info(' - physical parameters: gamma = {}, B0 = {}, TE = {}'.format(gamma, B0, TE))

    #define function spaces and empty lists
    #V = u_list[0].function_space()
    
    M1_list = []
    M2_list = []

    for u in u_list:
        #info('- processing entry {} of {}'.format(i, len(u_list)))

        #compute phases
        if isinstance(u, dolfin.Function): 
            uvec = u.vector()[:]
        else: 
            uvec = u
        Phi1 = gamma*B0*TE + 0*uvec #background phase
        Phi2 = gamma*B0*TE + np.pi*uvec/VENC    #contrasted phase

        M1 = np.zeros_like(Phi1, dtype = 'complex_')
        M2 = np.zeros_like(Phi2, dtype = 'complex_')

        #compute magnetization from phases
        M1 = np.cos(Phi1) + 1j*np.sin(Phi1)
        M2 = np.cos(Phi2) + 1j*np.sin(Phi2)

        #compute magnitude of magnetization 
        #M_mod = np.real(np.sqrt(M2*np.conj(M2)))

        #Mmod_list[i].vector()[:] = M_mod
        M1_list.append(M1)
        M2_list.append(M2)
    info('done!')
    return M1_list, M2_list#, Mmod_list

def get_magnitude(M_list, V):
    if isinstance(M_list[0], dolfin.Function):
        Mmod_list = [Function(V) for i in range(len(M_list))]
        for i, m in enumerate(M_list):
            M_mod = np.real(np.sqrt(m*np.conj(m)))
            Mmod_list[i].vector()[:] = M_mod
    else:
        Mmod_list = [np.zeros_like(M_list[0]) for m in M_list]
        for i, m in enumerate(M_list):
            M_mod = np.real(np.sqrt(m*np.conj(m)))
            Mmod_list[i] = M_mod
    return Mmod_list

def get_lumen_indexes(u_list):
    '''
    get the indexes of voxels that have a non-zero velocity in at least one time step
    currently only works for 1D arrays, if a numpy array is given
    Args:
        u_list: list of input Fenics functions or list of numpy arrays
    Returns:
        lumen_indxs: list of indexes 
    '''
    #separate only lumen voxels here
    lumen_indxs = []
    if type(u_list[0]) is np.ndarray:
        return np.nonzero(u_list[0])
    else:
        for k in range(len(u_list[0].vector()[:])):
            for t in range(len(u_list)):
                #not-lumen voxels should always be zero in ground truth
                if u_list[t].vector()[k] != 0:
                    lumen_indxs.append(k)
                    break
    return lumen_indxs 

def apply_magnetization_noise(M1_list, M2_list, SNR, V, seed=None, lumen_indxs=None):
    '''
    applies a given level of gaussian noise to the magnetization of the data. Changes in place
    Args:
        M1_list: list of numpy arrays. Background magnetization. If not given, magnetization will be computed.
        M2_list: list of numpy arrays. Contrast magnetization. If not given, magnetization will be computed.
        SNR: signal-to-noise ratio. Used to compute the standard deviation of the noise
        lumen_indxs: list of integers. Indices where noise is applied
    Returns:
        M1_list, M2_list: list of numpy arrays
        Mmod_list: list of Fenics functions
    '''
    info('adding noise to the magnetization...')
    Phi0 = M1_list[0]

    #compute background magnetization
    Magnetization = np.cos(Phi0) + 1j*np.sin(Phi0)
    M_module = np.abs(Magnetization)
    Nsignal = M_module.shape
    #get standard deviation from the backgorund magnetization depending on SNR
    std = get_std(M_module, SNR)
    Mmod_list = [Function(V) for m in M1_list]

    #decide if applying noise only to lumen
    if not lumen_indxs is None:
        #Nsignal = len(lumen_indxs)
        Nsignal = M_module[lumen_indxs].shape

    for count, (M1, M2) in enumerate(zip(M1_list, M2_list)):
        #separate into real and imaginary parts
        M1_re = np.real(M1)
        M1_im = np.imag(M1)
        M2_re = np.real(M2)
        M2_im = np.imag(M2)

        #apply noise to each part separately with new seeds
        if seed:
            info('- seed = {}'.format(seed + 4*count))
            np.random.seed(seed + 4*count)
            if lumen_indxs is None:
                M1_re_new = M1_re + np.random.normal(0, std, Nsignal)
                np.random.seed(seed + 4*count + 1)
                M1_im_new = M1_im + np.random.normal(0, std, Nsignal)
                np.random.seed(seed + 4*count + 2)
                M2_re_new = M2_re + np.random.normal(0, std, Nsignal)
                np.random.seed(seed + 4*count + 3)
                M2_im_new = M2_im + np.random.normal(0, std, Nsignal)
            else:
                M1_re_new = M1_re[lumen_indxs] + np.random.normal(0, std, Nsignal)
                np.random.seed(seed + 4*count + 1)
                M1_im_new = M1_im[lumen_indxs] + np.random.normal(0, std, Nsignal)
                np.random.seed(seed + 4*count + 2)
                M2_re_new = M2_re[lumen_indxs] + np.random.normal(0, std, Nsignal)
                np.random.seed(seed + 4*count + 3)
                M2_im_new = M2_im[lumen_indxs] + np.random.normal(0, std, Nsignal)
        else:
            M1_re_new = M1_re[lumen_indxs] + np.random.normal(0, std, Nsignal)
            M1_im_new = M1_im[lumen_indxs] + np.random.normal(0, std, Nsignal)
            M2_re_new = M2_re[lumen_indxs] + np.random.normal(0, std, Nsignal)
            M2_im_new = M2_im[lumen_indxs] + np.random.normal(0, std, Nsignal)

        #assign noised data
        M1[lumen_indxs] = M1_re_new + 1j*M1_im_new
        M2[lumen_indxs] = M2_re_new + 1j*M2_im_new
        M1_list[count] = M1
        M2_list[count] = M2
        #compute new magnitude of magnetization
        #M_mod = np.real(np.sqrt(M2*np.conj(M2)))
        #Mmod_list[count].vector()[:] = M_mod
    info('done!')
    return M1_list, M2_list, Mmod_list

def get_neighbours(mesh):
    '''
    For a given mesh, get the IDs of neighbouring cells for each cell (without ID of itself).
    '''
    # source: https://fenicsproject.discourse.group/t/how-to-create-lists-of-cell-neighbours/4728/2
    # Init facet-cell connectivity
    tdim = mesh.topology().dim()
    mesh.init(tdim - 1, tdim)

    # For every cell, build a list of cells that are connected to its facets
    # but are not the iterated cell
    cell_neighbours = {}
    for cell in cells(mesh):
        index = cell.index()
        cell_neighbours[index] = []
        for facet in facets(cell):
            facet_cells = facet.entities(tdim)
            for facet_cell in facet_cells:
                if (index!=facet_cell):
                    cell_neighbours[index].append(facet_cell)

    return cell_neighbours

def get_neighbours_2D(mesh, component=0):
    '''
    For a given (3D hexahedral) mesh, get the IDs of 8 neighbouring cells (in a specified slice) for each cell.
    component (int): 0, 1, or 2, it says which coordinate should be the same for cells to be treated as neighbours. For example, 0 means that the neighbours a cell would lie in yz plane. 
    '''

    # For every cell, build a list of cells that are connected to its edges 
    # but are not the iterated cell
    cell_neighbours = {}

    # Init edge-cell connectivity
    mesh.init(1,3)

    for cell in cells(mesh):
        # get the index of the cell
        idx = cell.index()
        # create an empty list of its neighbours
        cell_neighbours[idx] = [] 
        # calculate the midpoint of the cell
        midpointx = Cell(mesh, idx).midpoint()[component]
        # for each edge of this particular cell
        for edge in edges(cell):
            # find all the cells that contain this edge and iterate over them
            for id_cell in edge.entities(3):
                if id_cell == idx: # exclude itself
                    continue
                # check if (the component of) the two midpoints match 
                # if so, we found one of the eight neighbours
                # if not, we found a neighbour in a different slice
                if Cell(mesh, id_cell).midpoint()[component] == midpointx and id_cell not in cell_neighbours[idx]:
                    cell_neighbours[idx].append(id_cell)

    return cell_neighbours

def compute_median(unwrapped_noisy, timestep, neighbour_inds):
    '''
    Compute median value over the neighbours of voxel k in one timestep.

    inputs:
    unwrapped_noisy.....a list of fenics functions - one for each timestep
    timestep............the timestep over which we wish to compute the median
    neighbour_inds......a list of neighbour indices of the cell

    outputs:
    median..............the median value corresponding to the neighbours of cell k
    '''

    # get the number of neighbours of cell k
    neighbour_values = []

    # append neighbour values by the neighbouring cells at each time
    for ind in neighbour_inds:
        neighbour_values.append(unwrapped_noisy[timestep].vector()[ind])

    # compute median
    median = np.median(neighbour_values)

    return median

def wavelet_find_indices(unwrapped_noisy, thresh, wavelet='db1'):
    '''
    This function takes unwrapped noisy fenics functions and does the following:
    - for each voxel, it computes the wavelet transform in time, 
    - filters out the small coefficients (small velocities), 
    - computes the wavelet reconstruction of these coefficients (small velocities are replaced by zeros),
    - find the indices of nonzero values that are believed to be the ones where artifacts occured.

    inputs:
    unwrapped_noisy.....a list of fenics functions - one for each timestep
    thresh..............(float) threshold for filtering wavelet coefficients
    wavelet.............(string) the name of wavelet

    output:
    indices.........a list of "suspicious" indices of the form [i, k],
                    where k is the position in fenics vector array and i is the number of timestep
    '''

    ntimesteps = len(unwrapped_noisy) # number of timesteps
    vec_length = len(unwrapped_noisy[0].vector()[:]) # length of each fenics function
    
    # # create empty 2D numpy arrays, where the timesteps are represented by columns and voxels by rows
    data = np.empty((ntimesteps, vec_length))
    artifacts = np.empty((ntimesteps, vec_length))
    # assign the data array by unwrapped noisy functions 
    for i in range(ntimesteps):
        for k in range(vec_length):
            data[i][k] = unwrapped_noisy[i].vector()[k]
    
    # iterate for each voxel (row of data array): compute the wavelet transform, filter out all coefficients below the threshold and reconstruct data called artifacts
    for k in range(np.size(data,axis=1)):

        # 1 level 1D Discrete Wavelet Transform of data 
        # (it could be even multilevel, but that would slow down the computation)
        coeffs = pywt.wavedec(data[:, k],wavelet=wavelet,level=1)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        # coeff_arr......Wavelet transform coefficient array
        # coeff_slices...List of slices corresponding to each coefficient

        # get boolean array saying whether values in coeff_arr are below the threshold
        ind = np.abs(coeff_arr) > thresh

        # Threshold small velocities
        coeff_arr_filt = coeff_arr * ind 

        # get coefficients from the filtered array and slices
        coeffs_filt = pywt.array_to_coeffs(coeff_arr_filt, coeff_slices, output_format='wavedec')

        # wavelet reconstruction
        artifacts[:, k] = pywt.waverec(coeffs_filt, wavelet=wavelet)

    # find all indices of nonzero values in the reconstructed data (artifacts),
    # these are the indices of the "suspicious" voxels
    indices = np.transpose(np.nonzero(artifacts))
    return indices

def find_indices_temp_differences(data, thresh):
    '''
    inputs:
    fenics funs........list of fenics functions, each item for one time step
    thresh.............threshold for the differences between velocities at two consecutive time steps
                       (artifacts must outline this threshold)

    outputs:
    indices.........a list of "suspicious" indices of the form [i, k],
                    where k is the ID of the voxel with jump in velocity above the threshold (in the forward direction) and i is the number of timestep in which the jump occured
    '''

    ntimesteps = len(data)
    vec_length = len(data[0].vector()[:])
    S = data[0].function_space()
    data.append(Function(S))

    # COMPUTE THE DIFFERENCES IN THE INPUT DATA between two consecutive time steps
    differences = np.empty((ntimesteps, vec_length))
    for i in range(ntimesteps):
        differences[i] = abs(data[i+1].vector()[:] - data[i].vector()[:])

    # get a boolean array which is true whenever we detect jump in the data
    inds_bool = differences > thresh
    # find those indices [i, k] that correspond to all jumps in velocity greater than threshold
    indices = np.transpose(np.nonzero(inds_bool))

    return indices

def temporal_wavelet(unwrapped_noisy, mesh, lowvenc_funs, lowvenc, thresh, wavelet='db1'):
    '''
    This function takes unwrapped noisy fenics functions together with the lowvenc measurement
    and outputs the denoised version of these fenics functions.

    inputs:
    unwrapped_noisy.....a list of fenics functions - one for each timestep
    mesh................hexahedral mesh (to compute the neighbours for each voxel)
    lowvenc_funs........a list of fenics functions - the output from artificial measurement with lower venc
    lowvenc.............(float) venc corresponding to the lowvenc_funs
    thresh..............(float) threshold for filtering wavelet coefficients
    wavelet.............(string) the name of wavelet 
    
    output:
    denoised_data.......a list of denoised fenics functions - one for each timestep
    '''

    # if the number of timesteps is odd, add the last timestep at the end
    if len(unwrapped_noisy) %2 == 1:
        unwrapped_noisy.append(unwrapped_noisy[-1])
        lowvenc_funs.append(lowvenc_funs[-1])
        additional_timestep = True
    else:
        additional_timestep = False

    # get the number of timesteps
    ntimesteps = len(unwrapped_noisy)

    # make a deep copy of unwrapped noisy fenics functions (these will be overwritten)
    denoised_data = [unwrapped_noisy[i].copy(deepcopy=True) for i in range(ntimesteps)]

    # get the IDs of neighbours for each cell
    cell_neighbours = get_neighbours(mesh)

    # get the indices [i, k] (position in the array, timestep) 
    # where the artifacts are likely to occur (the wavelet coefficients above the threshold)
    indices = wavelet_find_indices(unwrapped_noisy, thresh, wavelet=wavelet)
    
    # iterate over the "suspicious" indices, compute median over its neighbours
    # and if the velocity in this voxel is not near the median value, apply standard dual venc
    for index in indices:

        i = index[0]
        k = index[1]

        median = compute_median(denoised_data, i, cell_neighbours[k])
        # standard dual venc unwrapping is applied on the spoiled voxels (if the jump over two timesteps is bigger than the threshold)
        if i==0 or abs(unwrapped_noisy[i].vector()[k]-denoised_data[i-1].vector()[k])>thresh:
            diff = median - lowvenc_funs[i].vector()[k]
            n = round(diff / (2*lowvenc))
            denoised_data[i].vector()[k] = lowvenc_funs[i].vector()[k] + 2*n*lowvenc
    
    # remove the last (artificial) timestep if it was added at the top
    if additional_timestep:
        denoised_data.remove(denoised_data[-1])
        lowvenc_funs.remove(lowvenc_funs[-1])
        unwrapped_noisy.remove(unwrapped_noisy[-1])

    return denoised_data

def temporal_differences(unwrapped_noisy, mesh, lowvenc_funs, lowvenc, thresh):
    '''
    This function takes unwrapped noisy fenics functions together with the lowvenc measurement
    and outputs the denoised version of these fenics functions.

    inputs:
    unwrapped_noisy.....a list of fenics functions - one for each timestep
    mesh................hexahedral mesh (to compute the neighbours for each voxel)
    lowvenc_funs........a list of fenics functions - the output from artificial measurement with lower venc
    lowvenc.............(float) venc corresponding to the lowvenc_funs
    thresh..............(float) threshold for filtering wavelet coefficients

    output:
    denoised_data.......a list of denoised fenics functions - one for each timestep
    '''

    # get the number of timesteps
    ntimesteps = len(unwrapped_noisy)

    # make a deep copy of unwrapped noisy fenics functions (these will be overwritten)
    denoised_data = [unwrapped_noisy[i].copy(deepcopy=True) for i in range(ntimesteps)]

    # get the IDs of neighbours for each cell
    cell_neighbours = get_neighbours(mesh)

    # get the indices [i, k] (position in the array, timestep)
    # where the artifacts are likely to occur (the wavelet coefficients above the threshold)
    indices = find_indices_temp_differences(unwrapped_noisy, thresh)

    # iterate over the "suspicious" indices, compute median over its neighbours
    # and if the velocity in this voxel is not near the median value, apply standard dual venc
    for index in indices:

        i = index[0]
        k = index[1]

        median = compute_median(denoised_data, i, cell_neighbours[k])
        # standard dual venc unwrapping is performed in the spoiled voxels
        diff = median - lowvenc_funs[i].vector()[k]
        n = round(diff / (2*lowvenc))
        denoised_data[i].vector()[k] = lowvenc_funs[i].vector()[k] + 2*n*lowvenc

    return denoised_data
