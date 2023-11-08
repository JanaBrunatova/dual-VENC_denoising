import sys, os
from dolfin import *
import mritools
from pathlib import Path
import shutil
sys.path.append('../')
import ruamel.yaml as yaml

def get_files(in_file, mode):

    #get all the files according to checkpoint or all mode
    if mode == 'checkpoint':
        unsort_indexes = os.listdir(in_file + 'checkpoint/')
    else:
        unsort_indexes = os.listdir(in_file)
    unsort_indexes_c = []

    for un_ind in unsort_indexes:
        if '.yaml' in un_ind:
            unsort_indexes.remove(un_ind)
        if '.xdmf' in un_ind:
            L_str = len(un_ind)
            next_str = un_ind[0:L_str-4] + 'h5'
            unsort_indexes.remove(un_ind)
            unsort_indexes.remove(next_str)
    
    if 'postprocess' in unsort_indexes:
        unsort_indexes.remove('postprocess')

    if not mode == 'checkpoint':
        for un_ind in unsort_indexes:
            if 'h5' in un_ind and ('u' in un_ind):
                L_str = len(un_ind)
                unsort_indexes_c.append(un_ind[1:L_str-3])
        
        indexes = [int(x) for x in unsort_indexes_c]
        indexes.sort()
    else:
        indexes = [int(ff) for ff in unsort_indexes]
        indexes.sort()

    files_lst = []
    for ind in indexes:
        if not mode == 'checkpoint':
                file = in_file + 'u' + str(ind) + '.h5'
        else:
            file = in_file +  'checkpoint/' + str(ind) + '/u.h5'

        files_lst.append(file)

    return files_lst, indexes

def copy_inputfile(options, inputfile):
    ''' Copy input file of reference solution to measurements directory.

    Args:
        options (dict):   Options dictionary (from YAML input file)
        inputfile (str):  Path of input file to be copied
    '''


    outpath = options['write_path']

    lower_index = outpath[0:-1].rfind('/')
    folder_name = outpath[lower_index+1:len(outpath)-1]

    path = Path(outpath)
    #path = path.joinpath(folder_name + '_meas/')
    path.mkdir(parents=True, exist_ok=True)
    path = path.joinpath('input.yaml')
    shutil.copy2(str(inputfile), str(path))
    print('Copied input file to {}'.format(path))

def create_functionspaces(options, mesh):
    if mesh.ufl_cell() == hexahedron:
        if options['scalar']:
            V = FunctionSpace(mesh, 'DG', 0)
        else:
            V = VectorFunctionSpace(mesh, 'DG', 0)
    elif 'bubble' in options and options['bubble'] == True:
        U = FiniteElement('P', mesh.ufl_cell(), 1)
        B = FiniteElement("Bubble", mesh.ufl_cell(), 4)
        P1 = FiniteElement('P', mesh.ufl_cell(), 1)
        V1 = VectorElement(NodalEnrichedElement(U,B))
        V = FunctionSpace(mesh,V1)
        P = FunctionSpace(mesh,P1)
    else:    
        V1 = VectorElement('P', mesh.ufl_cell(), 1) 
        P1 = FiniteElement('P', mesh.ufl_cell(), 1)
        V = FunctionSpace(mesh, V1)
        P = FunctionSpace(mesh,P1)
    return V, P

def read_mesh(mesh_file):
    ''' Read HDF5 or DOLFIN XML mesh.

    Args:
        mesh_file       path to mesh file

    Returns:
        mesh            Mesh
        sd              subdomains
        bnd             boundaries
    '''
    tmp = mesh_file.split('.')
    file_type = tmp[-1]
    mesh_pref = '.'.join(tmp[0:-1])

    if file_type == 'xml':
        mesh = Mesh(mesh_file)

    elif file_type == 'h5':
        mesh = Mesh()
        with HDF5File(mesh.mpi_comm(), mesh_file, 'r') as hdf:
            hdf.read(mesh, '/mesh', False)

    elif file_type == 'xdmf':
        mesh = Mesh()
        with XDMFFile(mesh_file) as xf:
            xf.read(mesh)

    else:
        raise Exception('Mesh format not recognized. Try XDMF or HDF5 (or XML,'
                        ' deprecated)')

    return mesh

def ROUTINE(options):
    mesh = read_mesh(options['mesh_in'])
    V, P = create_functionspaces(options, mesh)
    
    #read data according to measurement or checkpoint mode 
    in_files, indexes = get_files(options['in_file'], options['mode'])
    path_lst = []
    if options['scalar'] and not mesh.ufl_cell() == hexahedron:
        F_space = P
    else:
        F_space = V
    u_list = []
    times = []

    #read data into list of functions
    for file in in_files:
        u = Function(F_space)
        hdf = HDF5File(mesh.mpi_comm(), file, 'r')
        hdf.read(u, 'u/vector_0')
        u_list.append(u)
        time = hdf.attributes('u/vector_0').to_dict()['timestamp']
        times.append(time)
        hdf.close()
    u_out = u_list.copy()
    u_out_lst = []
    artifacts = options['artifacts']
    venc = None
    V_aux = None
    moptions = None
    direction = None
    skip_sp = False #skip all later artifacts

    #apply artifacts
    if 'temporal_undersampling' in artifacts:

        #decide if we only need to undersample files or interpolate between measurements
        if isinstance(artifacts['temporal_undersampling'], int):
            indexes = indexes[::artifacts['temporal_undersampling']]
            u_out, times = mritools.TemporalUndersampling(u_list, artifacts['temporal_undersampling'], times)
        else:
            assert 'dt_old' in artifacts['temporal_undersampling']
            dt_old = artifacts['temporal_undersampling']['dt_old']
            if 'rate' in artifacts['temporal_undersampling']:
                dt_new = artifacts['temporal_undersampling']['rate']*dt_old
            else: 
                dt_new = artifacts['temporal_undersampling'].get('dt_new', 1*dt_old)
            u_out, times = mritools.TemporalInterpolation(u_list, dt_old, dt_new, times)
            indexes = [int(i*dt_new*1000) for i in range(len(times))]
    if 'space_interpolation' in artifacts and artifacts['space_interpolation']['apply'] and not skip_sp:
        spoptions = artifacts['space_interpolation']

        #find degree and element for function space
        mesh_out = read_mesh(spoptions['mesh_out'])
        if mesh_out.ufl_cell() == tetrahedron:
            element = 'P'
            degree = 1
        elif mesh_out.ufl_cell() == hexahedron:
            element = 'DG'
            degree = 0
        else:
            raise Exception('Unsupported mesh type')
        direction = None

        #define appropriate function space
        if 'velocity_direction' in spoptions and not options['scalar'] and not spoptions['velocity_direction'] is None:
            V_out = FunctionSpace(mesh_out, element, degree)
            V_aux = VectorFunctionSpace(mesh_out, element, degree)
            direction = spoptions['velocity_direction']
        elif options['scalar']:
            V_out = FunctionSpace(mesh_out, element, degree)
        else:
            V_out = VectorFunctionSpace(mesh_out, element, degree)

        #interpolate and update function space/mesh
        u_out = mritools.SpatialInterpolation(u_out, F_space, V_out, V_aux, direction)
        F_space = V_out
        mesh = mesh_out
    if 'magnetization' in artifacts:
        moptions = artifacts['magnetization']
        #find magnetization options
        if 'venc' in moptions:
            venc = moptions['venc']
        elif 'phase_contrast' in moptions:
            venc = mritools.get_VENC(u_out, moptions['phase_contrast'])
        else:
            raise Exception('VENC required for generating magnetization!')
        if 'model_parameters' in moptions:
            pars = artifacts['magnetization']['model_parameters']
        else:
            pars = None
        if 'lumen' in moptions and moptions['lumen']:
            lumen_indxs = mritools.get_lumen_indexes(u_out)
        else:
            lumen_indxs = None

        #compute magnetization    
        M1_list, M2_list = mritools.magnetization(u_out, venc, pars)

        #apply noise if desired
        if 'SNR' in moptions and not moptions['SNR'] == 'inf':
            Mmod_reals_lst = []
            if 'realizations' in artifacts and artifacts['realizations']['apply']:
                for i in range(max(artifacts['realizations']['total_number'], 1)):
                    #generate noise for each realization
                    seed_add = len(u_out)*i*4   #seed updated by four since noise is applied four times per timestep
                    seed = artifacts['realizations']['seed_init'] + seed_add
                    #save write paths per realization
                    info('processing realization ' +  str(i+1) + ' out of ' + str(artifacts['realizations']['total_number']))
                    write_path = options['write_path'][0:-1] + '_s' + str(seed) + '/'
                    path_lst.append(write_path)
                    #compute magnetization
                    M1_list, M2_list = mritools.magnetization(u_out, venc, pars)
                    #generate noise
                    M1_list, M2_list, Mmod_list = mritools.apply_magnetization_noise(M1_list, M2_list, moptions['SNR'], F_space, seed, lumen_indxs)
                    Mmod_reals_lst.append(Mmod_list)
                    #turn back to velocity
                    u_out_seed = mritools.vel_from_mag(F_space, M1_list, M2_list, venc)
                    u_out_lst.append(u_out_seed)
            else:
                #generate noise and turn back to velocity
                path_lst.append(options['write_path'])
                seed = None if not 'seed' in moptions else moptions['seed']
                M1_list, M2_list, Mmod_list = mritools.apply_magnetization_noise(M1_list, M2_list, moptions['SNR'], F_space, seed, lumen_indxs)
                Mmod_reals_lst.append(Mmod_list)
                u_out = mritools.vel_from_mag(F_space, M1_list, M2_list, venc)
                u_out_lst.append(u_out)
    if not moptions or not 'SNR' in moptions or moptions['SNR'] == 'inf':
        path_lst.append(options['write_path'])
        u_out_lst.append(u_out)
    
    print('done!')
            

    #save resulting measurements
    xdmfs = []
    if options['save_xdmf']:
        for path in path_lst:
            xdmf_path = options['write_path'] + 'u_all.xdmf'
            xdmf = XDMFFile(xdmf_path)
            xdmf.parameters['rewrite_function_mesh'] = False
            xdmfs.append(xdmf)
    if 'save_h5' in options and options['save_h5']:
        for i, (u_el, path) in enumerate(zip(u_out_lst, path_lst)):
            for u, t, ind in zip(u_el, times, indexes):
                u.rename('velocity', 'u')
                outpath = path + 'u' + str(ind) + '.h5'
                print('Writing checkpoint ',outpath)
                hdf_u = HDF5File(mesh.mpi_comm(), outpath, 'w')
                hdf_u.write(u, '/u', t)
                hdf_u.close()
                if options['save_xdmf']:
                    xdmfs[i].write(u, t)
            if 'magnetization' in artifacts and artifacts['magnetization']['save_module']:
                if not 'Mmod_reals_lst' in globals():
                    Mmod_reals_lst = mritools.get_magnitude(M2_list, F_space)
                for m, t, ind in zip(Mmod_reals_lst[i], times, indexes):
                    m.rename('module','M')
                    modpath = path + 'Module/M' + str(ind) + '.h5'
                    print('Writing magnetization module ', modpath)
                    hdf_mod = HDF5File(mesh.mpi_comm(), modpath, 'w')
                    hdf_mod.write(m, '/M', t)
                    hdf_mod.close()
            if options['save_xdmf']:
                xdmfs[i].close()      


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]):
            inputfile = sys.argv[1]
        else:
            raise Exception('Command line arg given but input file does not exist:'
                            ' {}'.format(sys.argv[1]))
    else:
        raise Exception('An input file is required as argument!')
        # Reading inputfile
    with open(inputfile, 'r+') as f:
        options = yaml.load(f, Loader=yaml.Loader)

    copy_inputfile(options, inputfile)
    ROUTINE(options)
