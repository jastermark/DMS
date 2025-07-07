import h5py


def _recursive_load_h5(f):
    '''Loads dictionary from hdf5 file'''
    dict_to_load = {}
    keys = list(f.keys())
    for key in keys:
        dict_to_load[key] = f[key][()]
    return dict_to_load


def _load_h5_from_filename(filename):
    '''Loads dictionary from hdf5 file'''
    with h5py.File(filename, 'r') as f:
        return _recursive_load_h5(f)
 

def load_h5(file):
    '''Loads dictionary from hdf5 file'''
    if isinstance(file, str):
        return _load_h5_from_filename(file)
    elif isinstance(file, h5py.File) or isinstance(file, h5py.Group):
        return _recursive_load_h5(file)
    else:
        raise ValueError("Invalid input type")


def recursive_write_h5(f, data):
    for key, value in data.items():
        key = str(key)

        if isinstance(value, dict):
            group = f.require_group(key)
            recursive_write_h5(group, value)
        elif isinstance(value, h5py.SoftLink):
            f[key] = value
        elif value is not None:
            # Base case
            f.create_dataset(key, data=value)


def write_h5(data, output_path, mode='w', verbose=True):
    '''Writes dictionary to hdf5 file
    Args:
        data: dictionary to write
        output_path: path to write hdf5 file
        mode: file mode ('w' for write, 'a' for append)
        verbose: print output_path'''
    if verbose:
            print(f'Writing to hdf5 file: {output_path}...')
    with h5py.File(output_path, mode) as f:
        recursive_write_h5(f, data)