from pathlib import Path

import h5py
import numpy as np
import typer
from typing_extensions import Annotated

from .utils import get_lw_angular_spectrum


def generic(path: Path):
    """
    Generic particle file.
    """
    args = []
    with h5py.File(path, "r") as f:
        for prop in  ['x', 'y', 'z', 'ux', 'uy', 'uz', 't']:
            args.append(f[prop][()])
            
    return args

def smilei(path: Path, name: str):
    """
    Generate a LW angular spectrum from a SMILEI simulation and save it to the specified file.
    """
    track_particle_disordered_file = path / f"TrackParticlesDisordered_{name}.h5"
    track_particle_file = path / f"TrackParticles_{name}.h5"
    
            
    if not track_particle_disordered_file.exists():
        raise FileNotFoundError(f"{track_particle_disordered_file} not found.")
    
    if not track_particle_file.exists():
        print(f"{track_particle_file} not found. maybe the particles are not sorted by happi.")
        print(f"try sorting with happi...")
        try:
            import happi
            sim = happi.Open(path)
            sim.TrackParticles(name, sort=True)
        except ImportError:
            raise Exception("happi is required to sort the particle files")
        except Exception as e:
            raise e

    with h5py.File(track_particle_disordered_file, "r") as f:
        ts = list(f['data'].keys())
        dtSI = f['data'][ts[0]].attrs['dt'] * f['data'][ts[0]].attrs['timeUnitSI']
        dxSI = f['data'][ts[0]]['particles'][name]['position']['x'].attrs['unitSI']
    
    
    with h5py.File(track_particle_file, "r") as f:
        x = f['x'] * dxSI
        
        args = [x, ]
        for prop in ['y', 'z']:
            try:
                args.append(f[prop] * dxSI)
            except Exception:
                args.append(np.zeros_like(x))
        for prop in ['px', 'py', 'pz']:
            try:
                args.append(f[prop][()])
            except Exception:
                args.append(np.zeros_like(x))
        
        t = f['Times']*dtSI
        args.append(t)
    print(f"shape of input trajectory: x.shape={x.shape}, t.shape={t.shape}")
        
    return args
        
def epoch(path: Path, name: str):
    raise NotImplementedError()

def main(
    path: Annotated[Path, typer.Argument()], 
    from_smilei: Annotated[bool, typer.Option("--smilei")]=False,
    from_epoch: Annotated[bool, typer.Option("--epoch")]=False,
    name: Annotated[str, typer.Argument(help='species or particle name, required for Smilei and EPOCH files')]=None,
    ek_max: Annotated[float, typer.Option(help="Maximum energy of axis [MeV]")]=1.0,
    nek: Annotated[int, typer.Option()]=128,
    theta_max: Annotated[float, typer.Option()]=0.5,
    ntheta: Annotated[int, typer.Option()]=128,
    direction: Annotated[str, typer.Option(help='x | y | z')]='x',
    theta_plane: Annotated[str, typer.Option(help='xy | xz | yz')]='xy',
    backend: Annotated[str, typer.Option(help="mt | cuda | None")]='mt',
    check_velocity: Annotated[bool, typer.Option()]=True,
    savefig: Annotated[bool, typer.Option()]=False,
):
    """
    Calculate angular radiation spectrum from relativistic particle trajectories using Lienard-Wiechert potentials.
    """
    if from_smilei:
        assert name is not None, "Name of particle must be specified for Smilei files."
        args = smilei(path, name)
    elif from_epoch:
        assert name is not None, "Name of particle must be specified for Epoch files."
        args = epoch(path, name)
    else:
        args = generic(path)
        
    try:
        from mpi4py.MPI import COMM_WORLD as comm
        rank = comm.Get_rank()
    except ImportError:
        comm = None
        rank = 0
        
    if backend == 'None':
        backend = None
        
    ek_axis = np.linspace(0, ek_max, nek)
    omega_axis = ek_axis*1e6/1.55*2.35e15
    theta_axis, spectrum = get_lw_angular_spectrum(
        *args, 
        omega_axis=omega_axis, 
        theta_max=theta_max, 
        ntheta=ntheta, 
        direction=direction, 
        theta_plane=theta_plane, 
        backend=backend, 
        check_velocity=check_velocity,
        comm=comm
    )
    if rank == 0:
        savepath = path if path.is_dir() else path.parent
        np.savetxt(savepath/'theta_axis.txt', theta_axis)
        np.savetxt(savepath/'spectrum.txt', spectrum)
        if savefig:
            try:
                import matplotlib.pyplot as plt
            except ImportError as e:
                raise e
            fig, ax = plt.subplots(layout='tight')
            h = ax.imshow(
                spectrum,
                extent = [ek_axis[0], ek_axis[-1], theta_axis[0], theta_axis[-1]],
                aspect='auto',
                origin='lower'
            )
            fig.colorbar(h, ax=ax)
            fig.savefig(savepath/'spectrum.png', dpi=300)