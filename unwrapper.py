from __future__ import print_function
import numpy as np
import math
from fenics import *
from numpy.core.fromnumeric import shape
from numpy.lib.npyio import save
import scipy.signal
import scipy.io
from mritools import temporal_wavelet, get_neighbours_2D, temporal_differences

class Unwrapper:
    def __init__(self, options):
        self.savepath = options['io']['savepath']
        self.save_checkpoints = options['io']['write_checkpoints']
        self.save_xdmf = options['io']['write_xdmf']
        self.dt = options['dt']
        self.mesh_options = options['mesh']
        self.vencs = [infile['VENC'] for infile in options['input']]
        idx_highvenc = self.vencs.index(max(self.vencs))
        idx_lowvenc = self.vencs.index(min(self.vencs))
        self.venc = options['input'][idx_lowvenc]['VENC']
        self.in_file = options['input'][idx_highvenc]['in_file']

        #get method-specific parameters from the options
        mv_unwrapping = options.get('multi_venc_unwrapping', {})

        self.path_addons = {}

        for method in mv_unwrapping:
            self.path_addons[method] = options['multi_venc_unwrapping'][method]['path_addon']

        if 'omme_correction' in mv_unwrapping or 'omme_correction_threshold' in mv_unwrapping:
            if 'mask_path' in options['multi_venc_unwrapping']['omme_correction']:
                self.mask_path = options['multi_venc_unwrapping']['omme_correction']['mask_path']
            else:
                self.mask_path = ''

        #get mesh from the input file, with  boundaries
        self.mesh = Mesh()
        if options['mesh']['mesh_file'].endswith('h5'):
            h5 = HDF5File(self.mesh.mpi_comm(), options['mesh']['mesh_file'], 'r')
            h5.read(self.mesh, 'mesh', False)
            if h5.has_dataset('boundaries'):
                self.bnds = MeshFunction('size_t', self.mesh, self.mesh.topology().dim()-1)
                h5.read(self.bnds, 'boundaries')
        else:
            with XDMFFile(options['mesh']['mesh_file']) as xf:
                xf.read(self.mesh)
                print('read mesh')

        #assuming we only ever work in one function space
        self.mesh_type = self.mesh_options['mesh_type']
        if self.mesh_type == 'slice' or self.mesh_type == 'box':
            self.V = FunctionSpace(self.mesh, 'DG', 0)
        else:
            self.V = FunctionSpace(self.mesh, 'P', 1)

    def set_method(self, new_method):
        self.method = new_method

    def save(self, u_list, filelist, save_checkpoints = False, savepath = False, times = None):
        comm = self.mesh.mpi_comm()
        if not savepath:
            savepath = str(self.savepath)
        xdmf_u = XDMFFile(savepath + 'u.xdmf')
        u = Function(self.V)
        u.rename('velocity', 'u')
        for l in range(len(u_list)):
            # writing xdmf
            u.assign(u_list[l])
            if times is not None:
                xdmf_u.write(u, times[l])
            else:
                xdmf_u.write(u, l*self.dt)
            # writing checkpoint
            if save_checkpoints:
                if filelist:
                    print('saving checkpoint', l)
                    ckpath = savepath + filelist[l].split("/")[-1]
                    hdf = HDF5File(comm, ckpath, 'w')
                    if times is not None:
                        hdf.write(u, '/u', times[l])
                    else:
                        hdf.write(u, '/u', float(l*self.dt))
                    hdf.close()
                else:
                    print('saving checkpoint', l)
                    ckpath = savepath
                    if times is not None:
                        time = math.ceil(times[l]*1000)
                        name = f'/u{time}.h5'
                        hdf = HDF5File(comm, ckpath + name, 'w')
                        # print(f'saving at time {time}')
                        hdf.write(u, '/u', time)
                    else:
                        name = '/u{i}'.format(i = math.ceil(l*self.dt*1000)) + '.h5'
                        hdf = HDF5File(comm, ckpath + name, 'w')
                        hdf.write(u, '/u', float(l*self.dt))
                    hdf.close()

    def dual_venc(self, high_venc_files, low_venc_files, addon = False):
        print('unwrapping with dual venc')
        if not addon:
            addon = self.path_addons['lee']
        #read files with low venc
        t = 0
        times = []
        u = Function(self.V)
        low_venc = [Function(self.V) for i in range(len(low_venc_files))]
        for file in low_venc_files:
            h5u = HDF5File(self.mesh.mpi_comm(), file, 'r')
            h5u.read(u, 'u')
            # get time from the velocity file
            time = h5u.attributes('u/vector_0').to_dict()['timestamp']
            times.append(time)
            low_venc[t].assign(u)
            t +=1
        #read files with high venc
        t = 0
        u = Function(self.V)
        high_venc = [Function(self.V) for i in range(len(high_venc_files))]
        for file in high_venc_files:
            if isinstance(file, str):
                h5u = HDF5File(self.mesh.mpi_comm(), file, 'r')
                h5u.read(u, 'u')
                high_venc[t].assign(u)
            else:
                high_venc[t].assign(file)
            t +=1
        self.Lt = len(low_venc_files)
        u = low_venc
        for t in range(self.Lt):
            diff = high_venc[t].vector()[:] - low_venc[t].vector()[:]
            n = np.round(diff/(2*self.venc))
            u[t].vector()[:] += 2*n*self.venc
        if self.save_xdmf:
            self.save(u, False, self.save_checkpoints, self.savepath + addon, times=times)
        return u

    def omme(self, velocities, vencs):
        uw = Function(self.V)
        uw_vec = np.zeros(velocities[0].shape)
        if all([isinstance(venc, int) for venc in vencs]):
            venc_uw = np.lcm.reduce(vencs)
        else:
            venc_uw_int = [int(vencs[i]*1000) for i in range(len(vencs))]
            venc_uw = np.lcm.reduce(venc_uw_int)
            venc_uw = venc_uw/1000
        min_ind = np.argmin(vencs)
        min_venc = np.min(vencs)
        nb_samples = math.ceil(venc_uw/min_venc)+2
        range_u = np.arange(-nb_samples, nb_samples+1)*min_venc
        u = np.outer(velocities[min_ind][:],np.ones((1, len(range_u)))) + np.outer(np.ones((len(velocities[0][:]), 1)),range_u)
        for k in range(len(velocities[0])):
            J = 0
            u_k = u[k, abs(u[k,:]) <= venc_uw]
            for v_ind in range(len(vencs)):
                J = J - np.cos(np.pi * (velocities[v_ind][k] - u_k)/vencs[v_ind])
            ind_k = np.argmin(J)
            uw_vec[k] = u_k[ind_k]
        uw.vector()[:] = uw_vec
        return uw

    def omme_with_denoising(self, venc_files, vencs, addon = False, method = 'tempdiff'):
        '''
        method = 'wavelet' or 'tempdiff'; the denoising technique
        if False, the temporal difference method is applied twice in a row
        '''
        if not addon:
            addon = self.path_addons['omme_wavelet']
        print('unwrapping with %i vencs'%len(vencs))

        # get index of lowvenc and highvenc
        min_ind = np.argmin(vencs)
        max_ind = np.argmax(vencs)

        ntimesteps = len(venc_files[0])
        
        lowvenc_funs = [Function(self.V) for i in range(ntimesteps)]
        highvenc_funs = [Function(self.V) for i in range(ntimesteps)]

        times=[]
        uw = [] # unwrapped velocity

        # read velocities at each timestep into the list of fenics functions
        for i in range(ntimesteps):
            # read one timestep of lowvenc measurement
            with HDF5File(self.mesh.mpi_comm(), venc_files[min_ind][i], 'r') as hdf:
                hdf.read(lowvenc_funs[i], '/u/vector_0')
            # read one timestep of highvenc measurement
            with HDF5File(self.mesh.mpi_comm(), venc_files[max_ind][i], 'r') as hdf:
                hdf.read(highvenc_funs[i], '/u/vector_0')
                t = hdf.attributes('u/vector_0')['timestamp']
                times.append(t)

            # apply omme in this (i-th) timestep
            velocities = []
            velocities.append(highvenc_funs[i].vector()[:])
            velocities.append(lowvenc_funs[i].vector()[:])
            uw.append(self.omme(velocities, [vencs[max_ind], vencs[min_ind]]))
        
        # denoise the unwrapped (noisy) data using temporal wavelet filtering
        print(f'denoising with a lowvenc threshold = {vencs[min_ind]}')
        if method == 'wavelet':
            uw_denoised = temporal_wavelet(uw, self.mesh, lowvenc_funs, vencs[min_ind], thresh=vencs[min_ind], wavelet='db1')
        elif method == 'tempdiff':
            # apply temporal differences twice in a row
            uw_first = temporal_differences(uw, self.mesh, lowvenc_funs, vencs[min_ind], thresh=vencs[min_ind])
            uw_denoised = temporal_differences(uw_first, self.mesh, lowvenc_funs, vencs[min_ind], thresh=vencs[min_ind])
        else:
            raise Exception('Denoising method must be either wavelet or tempdiff.')

        if self.save_xdmf:
            self.save(uw_denoised, False, self.save_checkpoints, self.savepath + addon + 'denoised/', times=times)
        print('done!')
        return uw_denoised

    def odv_correction_voxel(self, velocities, vencs, voxel=0, negative=True):
        venc_uw_int = [int(vencs[i]*1000) for i in range(len(vencs))]
        venc_uw = np.lcm.reduce(venc_uw_int)
        venc_uw = venc_uw/1000
        min_venc = np.min(vencs)
        min_ind = np.argmin(vencs)
        du = min_venc/200  # Sampling of the cost functions

        u_k = np.arange(-venc_uw, venc_uw, du)  # search range

        # cost functional
        J = 0
        for v_ind in range(len(vencs)):
            J = J + 1 - np.cos(np.pi * (velocities[v_ind][voxel] - u_k)/vencs[v_ind])

        # find indices of local minimas of J
        inds = scipy.signal.argrelextrema(J, np.less)

        # find the local minima with the correct sign
        if negative:
            inds_negative = np.where(u_k[inds] <= 0)
            # If the noise level is very high, the minimum might not exist (within the range of effective venc)
            # Therefore, we use try except statement
            try:
                u_out = max(u_k[inds[0][inds_negative]])
            except:
                u_out = velocities[min_ind][voxel]
        else:
            inds_positive = np.where(u_k[inds] >= 0)
            try:
                u_out = min(u_k[inds[0][inds_positive]])
            except:
                u_out = velocities[min_ind][voxel]

        return u_out


    def odv_correction(self, unwrapped_noisy, venc_files, vencs, addon = False, threshold=0.0, component=2, mask=[]):
        '''
        This function takes unwrapped noisy fenics functions together with the lowvenc measurement
        and outputs the denoised version of these fenics functions.

        inputs:
        unwrapped_noisy.....a list of fenics functions - one for each timestep
        venc_files..........a list of of filenames; venc_files[0] contains all filenames of one measurement, venc_files[1] the second measurement
        vencs...............a list of vencs (floats) used for the artificial measurements
        threshold...........if the mean value over 8 neigbours is lower than the threshold, the voxel is not denoised
        component...........normal vector direction of our 2D slice (2 means that our hexahedral mesh lies in xy plane)
        
        output:
        denoised_data.......a list of denoised fenics functions - one for each timestep
        '''
        print("doing ODV correction (Pamela's algorithm)")
        if not addon:
            addon = self.path_addons['omme_correction']

        # get the lenght of the vector of each fenics function
        vec_length = len(unwrapped_noisy[0].vector()[:])
        # get the number of timesteps
        ntimesteps = len(unwrapped_noisy)

        # make a deep copy of unwrapped noisy fenics functions (these will be overwritten)
        denoised_data = [unwrapped_noisy[i].copy(deepcopy=True) for i in range(ntimesteps)]

        # get the neighbours, for example, cell_neighbours[13] outputs a list of ids of cells that have a common face with cell 13
        cell_neighbours = get_neighbours_2D(self.mesh,component=component) # component=2...xy plane

        # fenics functions in which the lowvenc and highvenc measurements will be stored
        lowvenc_funs = [Function(self.V) for i in range(ntimesteps)]
        highvenc_funs = [Function(self.V) for i in range(ntimesteps)]

        times=[]
        min_ind = np.argmin(vencs)
        max_ind = np.argmax(vencs)

        if self.mask_path != '':
            # read the mask array
            print(self.mask_path)
            f = scipy.io.loadmat(self.mask_path)
            mask_with_time = np.array(f.get("labels"))
        else:
            mask_array = mask

        for i in range(ntimesteps):

            if self.mask_path != '':
                # get one timestep of mask array
                mask_array = mask_with_time[:,:,i].flatten('F')

            # read velocities at each timestep into the list of fenics functions
            with HDF5File(self.mesh.mpi_comm(), venc_files[min_ind][i], 'r') as hdf:
                hdf.read(lowvenc_funs[i], '/u/vector_0')
            with HDF5File(self.mesh.mpi_comm(), venc_files[max_ind][i], 'r') as hdf:
                hdf.read(highvenc_funs[i], '/u/vector_0')
                t = hdf.attributes('u/vector_0')['timestamp']
                times.append(t)

            velocities = []
            velocities.append(highvenc_funs[i].vector()[:])
            velocities.append(lowvenc_funs[i].vector()[:])

            for k in range(vec_length):

                # skip if the voxel is not inside the domain of interest
                if mask_array[k] == 0:
                    continue

                voxel_value = unwrapped_noisy[i].vector()[k]

                # compute the mean value over the 8 neighbours of cell k
                neighbour_values = []
                for ind in cell_neighbours[k]:
                    neighbour_values.append(unwrapped_noisy[i].vector()[ind])

                mean_value = np.mean(neighbour_values)
                
                # if the sign of the central voxel is opposite than the sign of the mean value over its neighbours
                # and additionally, if the mean value of its neighbours is above the lower bound,
                # then the velocity value is replaced by the local minimum of J_{dual}(u) 
                # with the smallest velocity value of the same sign as the neighbourhood of the central pixel
                if abs(mean_value)>= threshold and np.sign(mean_value) > 0 and np.sign(voxel_value) < 0:
                    denoised_data[i].vector()[k] = self.odv_correction_voxel(velocities, vencs, voxel=k, negative=False)

                if abs(mean_value)>= threshold and np.sign(mean_value) < 0 and np.sign(voxel_value) > 0:
                    denoised_data[i].vector()[k] = self.odv_correction_voxel(velocities, vencs, voxel=k, negative=True)

        if self.save_xdmf:
            self.save(denoised_data, False, self.save_checkpoints, self.savepath + addon + 'denoised/', times=times)
        print('done!')
        return denoised_data

    def multi_venc(self, venc_files, vencs, addon = False):
        if not addon:
            addon = self.path_addons['omme']
        self.Lt = len(venc_files[0])
        print('unwrapping with %i vencs'%len(vencs))
        uw = []
        times = []
        for t in range(self.Lt):
            velocities = []
            for filelist in venc_files:
                if isinstance(filelist[t], str):
                    u = Function(self.V)
                    h5u = HDF5File(self.mesh.mpi_comm(), filelist[t], 'r')
                    h5u.read(u, 'u')
                    # get time from the velocity file
                    time = h5u.attributes('u/vector_0').to_dict()['timestamp']
                    velocities.append(u.vector()[:])
                    times.append(time) if time not in times else times
                else:
                    velocities.append(filelist[t].vector()[:])
            uw.append(self.omme(velocities, vencs))
        if self.save_xdmf:
            self.save(uw, False, self.save_checkpoints, self.savepath + addon, times=times)
        print('done!')
        return uw

    def get_lumen(self, ground_truth):
       #separate only lumen voxels here
        lumen_indxs = []
        self.Lt = len(ground_truth)
        for k in range(len(ground_truth[0].vector()[:])):
            for t in range(self.Lt):
                #not-lumen voxels should always be zero in ground truth
                # if ground_truth[t].vector()[k] != 0:
                if abs(ground_truth[t].vector()[k]) > 1e-12:
                    lumen_indxs.append(k)
                    break
        self.lumen_indxs = lumen_indxs
        return lumen_indxs

    def get_lumen_mask(self, ground_truth):
        lumen_indxs = []
        for k in range(len(ground_truth.vector()[:])):
            if abs(ground_truth.vector()[k]) > 1e-12:
                lumen_indxs.append(1)
            else:
                lumen_indxs.append(0)
        return lumen_indxs

    def get_mask(self, mask_path):
        # read the mask array
        f = scipy.io.loadmat(mask_path)
        mask_with_time = np.array(f.get("labels"))
        ntimesteps = mask_with_time.shape[2]
        nvoxels = mask_with_time.shape[0]*mask_with_time.shape[1]
        lumen_indxs = np.empty((ntimesteps,nvoxels))
        for i in range(ntimesteps):
            # get one timestep of mask array
            lumen_indxs[i] = mask_with_time[:,:,i].flatten('F')
        self.lumen_indxs = lumen_indxs
        return lumen_indxs

    def eval(self, wrapped, ground_truth, method = 'error', ts=None):

        if ts is not None:
            eval_one_timestep = True
            print(f"evaluating one timestep number {ts}")
        else:
            eval_one_timestep = False
            self.Lt = int((len(ground_truth))/2)

        if method == 'number':
            lumen_indxs = getattr(self, 'lumen_indxs', range(len(ground_truth[0].vector()[:])))
            n = 0
            if eval_one_timestep:
                for k in lumen_indxs:
                    if abs(wrapped[ts].vector()[k] - ground_truth[ts].vector()[k]) > 0.1:
                        n += 1
            else:
                for t in range(self.Lt):
                    for k in lumen_indxs:
                        if abs(wrapped[t].vector()[k] - ground_truth[t].vector()[k]) > 0.1:
                            n += 1
            print('number of wraps: ', n)
            print('percentage of wraps:', n / (self.Lt * len(lumen_indxs)) * 100)
            return n
        if method == 'error':
            lumen_indxs = getattr(self, 'lumen_indxs', range(len(ground_truth[0].vector()[:])))
            print(f"evaluating {self.Lt} timesteps")
            #concatenate all timesteps together
            length = len(ground_truth[0].vector()[lumen_indxs])

            if eval_one_timestep:
                ground = np.zeros((length))
                u = np.zeros((length))
                ground = ground_truth[ts].vector()[lumen_indxs]
                u = wrapped[ts].vector()[lumen_indxs]
            else:
                ground = np.zeros((self.Lt*length))
                u = np.zeros((self.Lt*length))
                for t in range(self.Lt):
                    ground[t*length:(t+1)*length] = ground_truth[t].vector()[lumen_indxs]
                    u[t*length:(t+1)*length] = wrapped[t].vector()[lumen_indxs]
            #compute norms
            error = np.linalg.norm(u-ground)/np.linalg.norm(ground)
            print('error: ', error)
            return error
        if method == 'error_mask':
            lumen_indxs = getattr(self, 'lumen_indxs')
            # create an empty array
            ground = np.array([])
            u = np.array([])
            if eval_one_timestep:
                mask_array = lumen_indxs[ts]
                for i, k in enumerate(mask_array):
                    if k==1:
                        ground = np.append(ground, ground_truth[ts].vector()[i])
                        u = np.append(u, wrapped[ts].vector()[i])
            else:
                self.Lt = len(ground_truth)
                print(f"evaluating {self.Lt} timesteps")
                for t in range(self.Lt):
                    mask_array = lumen_indxs[t]
                    for i, k in enumerate(mask_array):
                        if k==1:
                            ground = np.append(ground, ground_truth[t].vector()[i])
                            u = np.append(u, wrapped[t].vector()[i])
            #compute norms
            error = np.linalg.norm(u-ground)/np.linalg.norm(ground)
            print('error: ', error)
            return error
        return 0
