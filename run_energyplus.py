"""
Functions to call and run EnergyPlus models from python. Functions enable
parallel processing of idf files i.e. parallel simulations.

Works with python 2.7 and python 3 in both Windows and Linux.

@author: Carlos Duarte <cduarte@berkeley.edu>
"""


from os.path import join
import os
import numpy as np
import multiprocessing as mp
import subprocess


def save_sim_paths(idffiles, weatherfiles, sim_folder):
    """ Save idf and weather file paths to np object.
        Make it easier to resimulate in future.
    
    Parameters
    ----------
    idffiles: 
        list of paths of the idf files for the simulation
    weatherfiles: 
        list of paths to the weather files used in the simulation. 
        Must be the same length as idffiles because it matches them element wise.
    sim_folder: 
        path to where to save np object with file path names
    
    Returns
    -------
    File saved to sim_folder: np object
        np object created by np.savez
    """

    # convert lists to np arrays
    idffiles     = np.array(idffiles)
    weatherfiles = np.array(weatherfiles)

    np.savez(join(sim_folder, 'sim_paths'), idffiles=idffiles,
                                                           weatherfiles=weatherfiles)


def time_progress(tlt_files, processed_files, str_time, **kwargs):
    """Estimate the amount of time remaining to process files
    
    Parameters
    ----------
    tlt_files: int
        length of the total files to process`
    processed_files: int
        length of the total file already processed
    str_time: time.time() object
        start time of when the files started processing
    
    Returns
    -------
    Prints out the number of files left and estimated time left to processes
    """
    props = {"report_percentage": 0.1}
    props.update(kwargs)

    n_files = int(tlt_files*props["report_percentage"])

    # check to make sure number are not zero
    if n_files == 0: n_files = 1
    if processed_files == 0: processed_files = 1

    # determine when to send progress
    if (processed_files % n_files) == 0 and tlt_files > mp.cpu_count():
        # get elapsed_time
        elapsed_time = time.time() - str_time
        files_left = tlt_files - processed_files

        # get rate of change
        chg_rate = elapsed_time/processed_files

        # time left
        time_left = files_left*chg_rate
        hours_left = time_left / 3600
        seconds = time_left % 3600
        minutes_left = seconds / 60
        seconds_left = seconds % 60
        print("\n\n-----The is {:d} files left to process----\n".format(files_left))
        print("-----There is approximately {:.0f} hours, {:.0f} minutes, {:.0f} seconds left for processing to finish-----\n\n".format(hours_left, minutes_left, seconds_left))
        
        
def run_ep(idffiles, weatherfiles, outfolder, FMU=False, EP='EnergyPlusV8-4-0', args=None):
    """ This function runs simulation for input idf files; runs in parallel if more than 1 file
    
    Parameters
    ----------
    idffiles: 
        list of paths to the idf files for the simulation
    weatherfiles: 
        list of paths to the weather files used in the simulation. 
        Must be the same length as idffiles since it matches element wise with idf paths
    outfolder: 
        path to save energyplus simulation outputs
    FMU: 
        Set to true if fmu files are used
    EP: 
        energyplus version to run
    args: 
        insert additional arguments for command line run of energyplus
    
    Returns
    -------
    Simulated results to outfolder
    """
    # save idf and weather file paths in a numpy array
    save_sim_paths(idffiles, weatherfiles, outfolder)

    # identify the type of computer you are working with (pc or mac)
    if os.name == 'nt':
        EPexe = join("C:\\", EP, "energyplus.exe")
    elif os.name == 'posix':
        EP = EP.replace('V', '-')
        EPexe = join('/usr/local', EP, 'energyplus')
    else:
        raise NotImplementedError("OS not supported")

    # check number of cpus in computer
    core_num = mp.cpu_count()
    if core_num == 1:
        max_jobs = 1
    else:
        max_jobs = core_num - 1

    # save current project folder and get abolute paths for outfolder, idfs, and weather file
    project_folder = os.getcwd()
    outfolder = os.path.abspath(outfolder)
    idffiles = [os.path.abspath(fi) for fi in idffiles]
    weatherfiles = [os.path.abspath(we) for we in weatherfiles]

    jobs = set()

    if len(idffiles) == 1:
        idf_dir = os.path.dirname(idffiles[0])
        filename_noext = idffiles[0].split('.idf')[:-1][0]
        we = weatherfiles[0]

        if FMU:
            # change directory if you are running an FMU controller
            os.chdir(idf_dir)

        idfCmd = [EPexe,
                  '-w', we,                                 # define weather file
                  '-d', outfolder,                          # define output folder
                  '-p', os.path.basename(filename_noext),   # define output prefix
                  '-s', 'C',                                # define output suffix type
                  idffiles[0]]                              # define input idf

        # add addition arguments if needed
        if args is not None:
            if type(args) not in [list, tuple]:
                idfCmd.insert(-1, args)
            else:
                for a in args:
                    idfCmd.insert(-1, a)
        # run simulations
        p = subprocess.call(idfCmd)

    else:
        ii = 0
        start_time = time.time()
        last_eta = time.time()
        for fi, we in zip(idffiles, weatherfiles):
            idf_dir = os.path.dirname(fi)
            filename_noext = fi.split('.idf')[:-1][0]

            if FMU:
                # change directory if you are running an FMU controller
                os.chdir(idf_dir)

            idfCmd = [EPexe,
                      '-w', we,                # define weather file
                      '-d', outfolder,         # define output folder
                      '-p', os.path.basename(filename_noext),   # define output prefix
                      '-s', 'C',                                # define output suffix type
                      os.path.abspath(fi)]                      # define input idf

            # add addition arguments if needed
            if args is not None:
                if type(args) not in [list, tuple]:
                    idfCmd.insert(-1, args)
                else:
                    for a in args:
                        idfCmd.insert(-1, a)

            # run simulations
            jobs.add(subprocess.Popen(idfCmd))  # run simulations

            # add more runs as simulation are completing
            if len(jobs) >= max_jobs:
                [p.wait() for p in jobs]
                jobs.difference_update(
                    [p for p in jobs if p.poll() is not None]
                )
            ii += 1
            # # give status of models to perform every x simulations
            time_progress(len(idffiles), ii, start_time)
        # check if all child jobs were closed
        for p in jobs:
            if p.poll() is None:
                p.wait()
    os.chdir(project_folder)
    print('\n\n-----Processed ' + str(len(idffiles)) + ' simulation files.-----\n\n')


def one_run_rveso(eso, vrs, outfolder, EP='EnergyPlusV8-4-0', prefix=None, runall=False):
    """
    Gather variables from one eso file and create csv file for further processing

    Parameters
    ----------
    eso: 
        path to eso file from energyplus results
    vrs: 
        list of output variables to extract from eso file
    outfolder: 
        path to output folder to hold resulting csv files
    EP: 
        EnergyPlus version
    prefix: 
        String to append to csv file name
    runall: 
        Boolean asking if user want to run all rvi files in output folder
    
    Returns
    -------
    csv files to outfolder
    """

    if os.name == 'nt':
        RVIexe = join("C:\\", EP, "PostProcess\\ReadVarsESO.exe")
    elif os.name == 'posix':
        EP = EP.replace('V', '-')
        RVIexe = join('/usr/local', EP, 'PostProcess/ReadVarsESO')
    else:
        raise NotImplementedError("OS not supported")

    if prefix is None:
        prefix = '_output'

    filename_noext = os.path.basename(eso).split('.eso')[:-1][0]
    file_rvi = open(join(outfolder, filename_noext) + '.rvi', 'w')
    file_rvi.write(os.path.abspath(eso) + '\n')
    file_rvi.write(join(outfolder, os.path.basename(filename_noext)) + prefix + '.csv' + '\n')

    for j, v in enumerate(vrs):
        file_rvi.write(v + '\n')
    file_rvi.write('0')
    file_rvi.close()

    rvis = join(outfolder, filename_noext + '.rvi')

    # run energyplus rv eso
    subprocess.call([RVIexe, os.path.abspath(rvis)])


def unpack_run_rveso(tup):
    """ Unpacks tuples. Helper function for parallel processing
    """
    one_run_rveso(*tup)


def run_rveso(esos, vrs, outfolder, EP='EnergyPlusV8-4-0', prefix=None, pool_map=True):
    """Gather variables from multiple eso file and create csv files for further processing

    Parameters
    ----------
    eso: 
        list of paths to eso files from energyplus results
    vrs: 
        list of output variables to extract from eso file
    outfolder: 
        path to output folder to hold resulting csv files
    EP: 
        EnergyPlus version
    prefix: 
        String to append to csv file name
    pool_map: 
        Boolean asking if user wants to use pool_map for multiprocessing
    
    Returns
    -------
    csv files to outfolder
    """
    # check number of cpus in computer
    core_num = mp.cpu_count()
    if core_num == 1:
        max_jobs = 1
    else:
        max_jobs = core_num - 1

    jobs = set()

    if not pool_map:
        for eso in esos:
            prs = mp.Process(target=one_run_rveso, args=(eso, vrs, outfolder, EP, prefix))
            jobs.add(prs)
            prs.start()

            if len(jobs) >= max_jobs:
                [p.join() for p in jobs]
                jobs.difference_update(
                    [p for p in jobs if p.is_alive() is not True]
                )

        # check if all child jobs were closed
        for p in jobs:
            p.join()

    else:
        pool = mp.Pool(processes=max_jobs)
        pool.map(unpack_run_rveso, ((eso, vrs, outfolder, EP, prefix) for eso in esos))
        pool.close()
        pool.join()

 
def get_file(folder, keyterm, extension):
    """
    This function returns file(s) based on keyterm and extension
    :param folder: path to folder where to search keyterm in
    :param keyterm: term used to identify a particular file
    :param extension: extension of files to search keyterms
    :return: Returns file(s) based on keyterm
    """
    find_term = re.compile(keyterm.lower())
    files = glob.glob(join(folder, '*.' + extension))
    hits = [i for i, l in enumerate(files) for m in [find_term.search(l.lower())] if m]

    if len(hits) == 0:
        print('No files with search term: {} found in folder {}!'.format(keyterm, folder))
        f = None
        import pdb; pdb.set_trace()
    else:
        f = files[hits[0]]

    return f
