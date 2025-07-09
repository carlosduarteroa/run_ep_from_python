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
import re, glob, shutil
import time
import uuid
import collections.abc
import six


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

def make_folder(new_folder_path):
    """ Checks and makes new folder if it does not exist
    """
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

def run_ep_pre(idffiles, weatherfiles, EP='EnergyPlusV9-0-1', outfolder=None, idffolder=None, required_sim_files=None,
               mp_cores=None, rveso_col_max=250):
    """This function runs the EnergyPlus dot Bat file. Good to run all preprocessing
    functions
    """
    # identify the type of computer you are working with (pc or mac)
    if os.name == 'nt':
        EPexe = join("C:\\", EP, "Epl-run.bat")
    elif os.name == 'posix':
        EP = EP.replace('V', '-').replace('.', '-')
        EPexe = join('/usr/local', EP, 'energyplus')
    else:
        raise NotImplementedError("OS not supported")

    # check number of cpus in computer
    if mp_cores is None:
        core_num = mp.cpu_count()
        if core_num == 1:
            max_jobs = 1
        else:
            max_jobs = core_num - 1
    else:
        core_num = int(mp_cores)

    # check inputs to make sure they are iterable
    if not isinstance(idffiles, collections.abc.Iterable) and not isinstance(idffiles, six.string_types):
        idffiles = [idffiles]
    elif isinstance(idffiles, six.string_types):
        idffiles = [idffiles]

    # check that idffile is a list
    if not isinstance(weatherfiles, collections.abc.Iterable) and not isinstance(weatherfiles, six.string_types):
        weatherfiles = [weatherfiles]
    elif isinstance(weatherfiles, six.string_types):
        weatherfiles = [weatherfiles]

    # change integer to a string
    if not isinstance(rveso_col_max, str):
        rveso_col_max = str(rveso_col_max)

    # save current project folder and get abolute paths for outfolder, idfs, and weather file
    project_folder = os.getcwd()
    weather_files = [os.path.abspath(wea) for wea in weatherfiles]

    if idffolder is None:
        idf_files_in = [os.path.abspath(fi) if not os.path.isabs(fi) else fi for fi in idffiles]
    else:
        idf_files_in = [os.path.abspath(join(idffolder, os.path.basename(fi))) for fi in idffiles]

    idf_folder = [os.path.dirname(fi) for fi in idf_files_in]

    # define output folders
    if outfolder is None:
        out_folder = os.path.abspath(idf_folder[0])
        tmp_folder = os.path.abspath(join(out_folder, 'tmp_folder'))
    else:
        out_folder = os.path.abspath(outfolder)
        tmp_folder = os.path.abspath(join(out_folder, 'tmp_folder'))

    make_folder(tmp_folder)

    # copy idfs to output_idfs and make own folder
    tmp_folder_names = [str(uuid.uuid1()) for _ in range(len(idf_files_in))]
    [make_folder(join(tmp_folder, tp)) for tp in tmp_folder_names]
    tmp_idf_loc = [shutil.copy2(fi, join(tmp_folder, tp, os.path.basename(fi))) for fi, tp in zip(idf_files_in, tmp_folder_names)]

    # copy required simulation files to new directories
    if required_sim_files is not None:
        if not isinstance(required_sim_files, (list, tuple)):
            required_sim_files = [required_sim_files]

        req_sim_files = [join(idf_folder[0], rfi) if not os.path.isabs(rfi) else rfi for rfi in required_sim_files]
        tmp_req_files = [shutil.copy2(rfi, os.path.dirname(fi)) for rfi in req_sim_files for fi in tmp_idf_loc]

    orig_idf_loc = idf_files_in
    idf_files_in = tmp_idf_loc

    # set up parallel progress report
    jobs = set()
    ii = 0
    start_time = time.time()
    last_eta = time.time()

    for i, (fi, wea) in enumerate(zip(idf_files_in, weather_files)):
        idf_dir = os.path.dirname(fi)
        idf_noext = fi.split('.idf')[:-1][0]

        idf_out = os.path.abspath(join(out_folder, os.path.basename(idf_noext)))

        we = wea.split('.epw')[:-1][0]
        we_bn_noext = os.path.basename(we)

        # change idf directory
        if project_folder != idf_dir:
            os.chdir(idf_dir)

        idf_cmd = [EPexe, 
                  idf_noext,            # full idf path with no extension
                  idf_out,              # full output idf path with no extension
                  'idf',                # extension of file idf or imf
                  wea,                   # full path to weather file 
                  'EP',                 # to indicate if weather file is used
                  'N',                  # should pausing occur
                  rveso_col_max,        # number of columns to print for rv eso
                  'N',
                  'N'
                  ]

        # run simulations
        jobs.add(subprocess.Popen(idf_cmd))  # run simulations

        # add more runs as simulation are completing
        if len(jobs) >= max_jobs:
            [p.wait() for p in jobs]
            jobs.difference_update(
                [p for p in jobs if p.poll() is not None]
            )

        # change back to project folder if you need too
        if project_folder != idf_dir:
            os.chdir(project_folder)

        # # give status of models to perform every x simulations
        ii += 1
        time_progress(len(idf_files_in), ii, start_time)

    # check if all child jobs were closed
    for p in jobs:
        if p.poll() is None:
            p.wait()

    # remove tmp folder
    shutil.rmtree(tmp_folder)

    # print final progress
    print('\n\n-----Processed ' + str(len(idf_files_in)) + ' simulation files.-----\n\n')


def run_ep(idffiles, weatherfiles, outfolder, FMU=False, EP='EnergyPlusV8-4-0', save_paths=False, args=None):
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
    if save_paths:
        save_sim_paths(idffiles, weatherfiles, outfolder)

    # identify the type of computer you are working with (pc or mac)
    if os.name == 'nt':
        EPexe = join("C:\\", EP, "energyplus.exe")
    elif os.name == 'posix':
        EP = EP.replace('V', '-').replace('.', '-')
        EPexe = join('/usr/local', EP, 'energyplus')
    else:
        raise NotImplementedError("OS not supported")

    # check number of cpus in computer
    core_num = mp.cpu_count()
    if core_num == 1:
        max_jobs = 1
    else:
        max_jobs = core_num - 1

    # check that idffiles and outfolder are lists
    if not isinstance(idffiles, collections.abc.Iterable) and not isinstance(idffiles, six.string_types):
        idffiles = [idffiles]
    elif isinstance(idffiles, six.string_types):
        idffiles = [idffiles]

    if not isinstance(weatherfiles, collections.abc.Iterable) and not isinstance(weatherfiles, six.string_types):
        weatherfiles = [weatherfiles]
    elif isinstance(weatherfiles, six.string_types):
        weatherfiles = [weatherfiles]

    if not isinstance(outfolder, collections.abc.Iterable) and not isinstance(outfolder, six.string_types):
        outfolder = [outfolder]
    elif isinstance(outfolder, six.string_types):
        outfolder = [outfolder]

    # save current project folder and get absolute paths for outfolder, idfs, and weather file
    project_folder = os.getcwd()
    idffiles = [os.path.abspath(fi) for fi in idffiles]

    # input the same weather file for all simulation if only 1 is defined
    if len(weatherfiles) == 1 and len(idffiles) > 1:
        weatherfiles = [os.path.abspath(weatherfiles[0]) for _ in idffiles]
    elif len(weatherfiles) > 1:
        weatherfiles = [os.path.abspath(outf) for outf in weatherfiles]
    else:
        weatherfiles = [os.path.abspath(weatherfiles[0])]

    # input the same outfolder for all simulations if only 1 defined
    if len(outfolder) == 1 and len(idffiles) > 1:
        outfolders = [os.path.abspath(outfolder[0]) for _ in idffiles]
    elif len(outfolder) > 1:
        outfolders = [os.path.abspath(outf) for outf in outfolder]
    else:
        outfolders = [os.path.abspath(outfolder[0])]

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
                  '-d', outfolders[0],                          # define output folder
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
        for fi, we, outf in zip(idffiles, weatherfiles, outfolders):
            idf_dir = os.path.dirname(fi)
            filename_noext = fi.split('.idf')[:-1][0]

            if FMU:
                # change directory if you are running an FMU controller
                os.chdir(idf_dir)

            idfCmd = [EPexe,
                      '-w', we,                # define weather file
                      '-d', outf,         # define output folder
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
        EP = EP.replace('V', '-').replace('.', '-')
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

if __name__ == '__main__':

    # single file example
    one_idf = "F:/test_sims_ep/SF_Minnesota_Duluth.Intl.AP.727450_hp_slab_IECC_2006_Occupancy1_SP1_Thermostat5.idf"
    one_weather = "C:/EnergyPlusV9-1-0/WeatherData/USA_AK_Fairbanks.Intl.AP.702610_TMY3.epw"
    run_ep_pre(one_idf, one_weather, EP='EnergyPlusV9-0-1')

    import pdb; pdb.set_trace()
    # multi file example
    idffiles = glob.glob(join('F:/test_sims/*.idf'))
    weatherfiles = [one_weather]*len(idffiles)
    run_ep_pre(idffiles, weatherfiles, EP='EnergyPlusV9-0-1', outfolder= "F:/Test_new_folder", required_sim_files= "./schedules_all.csv")
    import pdb; pdb.set_trace()

