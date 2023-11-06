from __future__ import print_function
import numpy as np
import sys
from fenics import *
import glob
import statistics
from pathlib import Path
from unwrapper import Unwrapper

def read_parameters(infile):
    ''' Read in parameters yaml file.

    Args:
        infile      path to yaml file

    Return:
        prms        parameters dictionary
    '''
    import ruamel.yaml as yaml
    with open(infile, 'r+') as f:
        prms = yaml.load(f, Loader=yaml.Loader)
    return prms

def getfiles(path, dt, min_dt):
        #get the velocities from the input files
        files = glob.glob(path + '/**/[0-9]*/u.h5', recursive=True)
        if files == []:
            files = glob.glob(path + 'u[0-9]*.h5', recursive=True)
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        n = round(dt/min_dt)
        nfiles = files[:len(files):n]
        return nfiles

def max_velocity(velocities):
    length = len(velocities[0].vector()[:])
    u = np.zeros((len(velocities) * length))
    for t in range(len(velocities)):
        u[t*length:(t+1)*length] = velocities[t].vector()[:]
    return np.max(u)

def ROUTINES(options):
    ''' 

    All the process activated in the input file should be coded here
    
    '''
    unwrapper = Unwrapper(options)
    realizations = options['io'].get('realizations', 0)
    seeds = [val for key, val in options['io'].items() if key.startswith('init_seed')]
    seed_inc = options['io'].get('seed_increase', 1)
    dt = options['dt']
    min_dt = options['minimum_dt']
    vencs = [infile['VENC'] for infile in options['input']]
    idx_lowvenc = vencs.index(min(vencs))
    idx_highvenc = vencs.index(max(vencs))
    if realizations > 0:
        if options['eval']['save_all_errors'] == True:
                filename = Path(options['io']['savepath'] + "all_error.txt")
                dirname = Path(options['io']['savepath'])
                dirname.mkdir(parents=True, exist_ok=True)
                filename.touch(exist_ok=True)  # will create file, if it exists will do nothing
                
        total_errors = {}
        for k in range(realizations):
            info('processing realization ' +  str(k+1) + ' out of ' + str(realizations))

            savepath = options['io']['savepath'][0:-1] + '_s' + str(seeds[idx_highvenc] + k*seed_inc) + '_' + str(seeds[idx_lowvenc] + k*seed_inc) + '/'
            unwrapper.savepath = savepath
            vel_paths = [infile['in_file'][0:-1] + '_s' + str(seed + k*seed_inc) + '/' for seed, infile in zip(seeds, options['input'])]
            vs = {}
 
            mv_methods = options.get('multi_venc_unwrapping', False)
            if mv_methods:
                vels = [getfiles(path, dt, min_dt) for path in vel_paths]
                mv_methods = [method for method in mv_methods if mv_methods[method]['apply']]
                for method in mv_methods:
                    if method == 'omme':
                        v_omme = unwrapper.multi_venc(vels, vencs)
                        vs['omme'] = v_omme
                    if method == 'omme_wavelet':
                        v_omme_wavelet = unwrapper.omme_with_denoising(vels, vencs, method = 'wavelet', addon="ODV_wavelet/")
                        vs['omme_wavelet'] = v_omme_wavelet
                    if method == 'omme_tempdiff':
                        v_omme_tempdiff = unwrapper.omme_with_denoising(vels, vencs, method = 'tempdiff', addon="ODV_tempdiff/")
                        vs['omme_tempdiff'] = v_omme_tempdiff
                    if method == 'omme_correction':
                        threshold = options['multi_venc_unwrapping']['omme_correction']['threshold']
                        if options['eval']['lumen']:
                            ground_files = getfiles(options['eval']['groundtruth_file'], dt, min_dt)
                            mesh = Mesh()
                            if options['mesh']['mesh_file'][-2:] == 'h5':
                                h5 = HDF5File(mesh.mpi_comm(), options['mesh']['mesh_file'], 'r')
                                print('reading ' + options['mesh']['mesh_file'])
                                h5.read(mesh, '/mesh', False)
                                h5.close()
                            elif options['mesh']['mesh_file'][-4:] == 'xdmf':
                                with XDMFFile( options['mesh']['mesh_file']) as xf:
                                    xf.read(mesh)
                            V = FunctionSpace(mesh, 'DG', 0)
                            ground = Function(V)
                            h5u = HDF5File(mesh.mpi_comm(), ground_files[3], 'r')
                            h5u.read(ground, 'u')
                            lumen_mask = unwrapper.get_lumen_mask(ground)
                            v_omme_correction = unwrapper.odv_correction(v_omme, vels, vencs, threshold=threshold, mask=lumen_mask)
                        else:
                            v_omme_correction = unwrapper.odv_correction(v_omme, vels, vencs, threshold=threshold)
                        vs['omme_correction'] = v_omme_correction
                    if method == 'lee':
                        v_lee = unwrapper.dual_venc(vels[idx_highvenc], vels[idx_lowvenc])
                        vs['lee'] = v_lee

            if 'eval' in options and options['eval']['apply']:
                if options['mesh']['mesh_file'].endswith('.h5'):
                    mesh = Mesh()
                    h5 = HDF5File(mesh.mpi_comm(), options['mesh']['mesh_file'], 'r')
                    h5.read(mesh, 'mesh', False)
                else:
                    with XDMFFile(options['mesh']['mesh_file']) as xf:
                        xf.read(mesh)
                if options['mesh']['mesh_type'] == 'slice':
                    V = FunctionSpace(mesh, 'DG', 0)
                else:
                    V = FunctionSpace(mesh, 'P', 1)
                if options['eval']['omme_as_groundtruth'] and 'omme' in vs:
                    grounds = vs['omme']
                else:
                    ground_files = getfiles(options['eval']['groundtruth_file'], dt, min_dt)
                    t = 0
                    u = Function(V)
                    grounds = [Function(V) for i in range(len(ground_files))]
                    for file in ground_files:
                        h5u = HDF5File(mesh.mpi_comm(), file, 'r')
                        h5u.read(u, 'u')
                        grounds[t].assign(u)
                        t +=1

                t = 0
                u = Function(V)
                wraps = [Function(V) for i in range(len(vels[idx_lowvenc]))]
                for file in vels[idx_lowvenc]:
                    h5u = HDF5File(mesh.mpi_comm(), file, 'r')
                    h5u.read(u, 'u')
                    wraps[t].assign(u)
                    t += 1

                if options['eval']['lumen']:
                    unwrapper.get_lumen(grounds)

                errors = {}
                errors['wrapped'] = unwrapper.eval(wraps, grounds, options['eval']['method'])
                for key in vs:
                    info('evaluating ' + str(key) + ' method')
                    errors[key] = unwrapper.eval(vs[key], grounds, options['eval']['method'])
                
                for key in errors:
                    if not key in total_errors:
                        total_errors[key] = []
                    total_errors[key].append(errors[key])
        if options['eval']['save_all_errors']:
            with open(filename, 'w') as f:
                print(total_errors, file=f)

        #compute error average and standard deviation for error bars
        info('calculating means and std of errors')
        if options['eval']['apply']:
            means = {}
            stds = {}
            for key, val in total_errors.items():
                info('evaluating ' + str(key) + ' method')
                means[key] = sum(val)/len(val)
                info('average: ' + str(means[key]))
                stds[key] = statistics.stdev(val)
                info('standard deviation: ' + str(stds[key]))

            if options['eval']['save_errors']:
                filename = Path(options['io']['savepath'] + "error_means.txt")
                dirname = Path(options['io']['savepath'])
                dirname.mkdir(parents=True, exist_ok=True)
                filename.touch(exist_ok=True)  # will create file, if it exists will do nothing
                with open(filename, 'w') as f:
                    print(means, file=f)
                filename = Path(options['io']['savepath'] + "error_stds.txt")
                filename.touch(exist_ok=True)
                with open(filename, 'w') as f:
                    print(stds, file=f)
    else:
        vs = {}
        mv_methods = options.get('multi_venc_unwrapping', False)
        if mv_methods:
            savepath = options['io']['savepath'][0:-1] + '_s' + str(seeds[idx_highvenc]) + '_' + str(seeds[idx_lowvenc]) + '/'
            unwrapper.savepath = savepath
            vel_paths = [infile['in_file'][0:-1] + '_s' + str(seed) + '/' for seed, infile in zip(seeds, options['input'])]
            vels = [getfiles(path, dt, min_dt) for path in vel_paths]
            mv_methods = [method for method in mv_methods if mv_methods[method]['apply']]
            for method in mv_methods:
                if method == 'omme':
                    v_omme = unwrapper.multi_venc(vels, vencs)
                    vs['omme'] = v_omme
                if method == 'omme_wavelet':
                    v_omme_wavelet = unwrapper.omme_with_denoising(vels, vencs, method = 'wavelet', addon="ODV_wavelet/")
                    vs['omme_wavelet'] = v_omme_wavelet
                if method == 'omme_tempdiff':
                    v_omme_tempdiff = unwrapper.omme_with_denoising(vels, vencs, method = 'tempdiff', addon="ODV_tempdiff/")
                    vs['omme_tempdiff'] = v_omme_tempdiff
                if method == 'omme_correction':
                    threshold = options['multi_venc_unwrapping']['omme_correction']['threshold']
                    if options['eval']['lumen']:
                        ground_files = getfiles(options['eval']['groundtruth_file'], dt, min_dt)
                        mesh = Mesh()
                        if options['mesh']['mesh_file'][-2:] == 'h5':
                            h5 = HDF5File(mesh.mpi_comm(), options['mesh']['mesh_file'], 'r')
                            print('reading ' + options['mesh']['mesh_file'])
                            h5.read(mesh, '/mesh', False)
                            h5.close()
                        elif options['mesh']['mesh_file'][-4:] == 'xdmf':
                            with XDMFFile( options['mesh']['mesh_file']) as xf:
                                xf.read(mesh)
                        V = FunctionSpace(mesh, 'DG', 0)
                        ground = Function(V)
                        h5u = HDF5File(mesh.mpi_comm(), ground_files[3], 'r')
                        h5u.read(ground, 'u')
                        lumen_mask = unwrapper.get_lumen_mask(ground)
                        v_omme_correction = unwrapper.odv_correction(v_omme, vels, vencs, threshold=threshold, mask=lumen_mask)
                    else:
                        v_omme_correction = unwrapper.odv_correction(v_omme, vels, vencs, threshold=threshold)
                    vs['omme_correction'] = v_omme_correction
                if method == 'lee':
                    v_lee = unwrapper.dual_venc(vels[idx_highvenc], vels[idx_lowvenc])
                    vs['lee'] = v_lee

        if options['eval']['apply']:
            info('calculating errors...')
            mesh = Mesh()
            if options['mesh']['mesh_file'][-2:] == 'h5':
                h5 = HDF5File(mesh.mpi_comm(), options['mesh']['mesh_file'], 'r')
                print('reading ' + options['mesh']['mesh_file'])
                h5.read(mesh, '/mesh', False)
                h5.close()
            elif options['mesh']['mesh_file'][-4:] == 'xdmf':
                with XDMFFile( options['mesh']['mesh_file']) as xf:
                    xf.read(mesh)
                    print('read mesh')
            else:
                raise Exception('Unknown type of input mesh. Exiting.')
            if options['mesh']['mesh_type'] == 'slice':
                V = FunctionSpace(mesh, 'DG', 0)
            else:
                V = FunctionSpace(mesh, 'P', 1)
            if options['eval']['omme_as_groundtruth'] and 'omme' in vs:
                grounds = vs['omme']
            else:
                ground_files = getfiles(options['eval']['groundtruth_file'], dt, min_dt)
                t = 0
                u = Function(V)
                grounds = [Function(V) for i in range(len(ground_files))]
                for file in ground_files:
                    h5u = HDF5File(mesh.mpi_comm(), file, 'r')
                    h5u.read(u, 'u')
                    grounds[t].assign(u)
                    t +=1

            if options['eval']['lumen']:
                unwrapper.get_lumen(grounds)
            t = 0
            u = Function(V)
            wraps = [Function(V) for i in range(len(vels[idx_lowvenc]))]
            for file in vels[idx_lowvenc]:
                h5u = HDF5File(mesh.mpi_comm(), file, 'r')
                h5u.read(u, 'u')
                wraps[t].assign(u)
                t += 1
            errors = {}
            errors['wrapped'] = unwrapper.eval(wraps, grounds, options['eval']['method'])
            for key in vs:
                info('evaluating ' + str(key) + ' method')
                errors[key] = unwrapper.eval(vs[key], grounds, options['eval']['method'])

            # filename = Path(options['io']['savepath'] + "errors.pkl")
            filename = Path(options['io']['savepath'] + "errors.txt")
            dirname = Path(options['io']['savepath'])
            dirname.mkdir(parents=True, exist_ok=True)
            filename.touch(exist_ok=True)  # will create file, if it exists will do nothing
            with open(filename, 'w') as f:
                print(errors, file=f)

if __name__ == '__main__':

    # This lines are necessary in order to read the input file as well of the 'read_parameters' function.

    inputfile = sys.argv[1]
    info('Found input file ' + inputfile)

    # this part will be executed first and will call all the other process. 
    options =	read_parameters(inputfile)
    ROUTINES(options)

            
